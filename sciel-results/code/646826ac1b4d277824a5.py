import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _autocorr(h):
    """Compute full autocorrelation (convolution of h with itself)."""
    return np.convolve(h, h)


def _ccp_iterate(h0, max_time=8.0, n_planes=12):
    """Cutting Plane Method: iteratively add linear constraints on autocorrelation.
    
    At each iteration, we solve an LP that minimizes the upper bound on max autocorrelation,
    subject to sum(h)=1 and linearized autocorrelation constraints from previous points.
    """
    t0 = time.time()
    h_curr = h0.copy()
    s = float(np.sum(h_curr))
    if abs(s) < 1e-12:
        return h_curr
    h_curr = h_curr / s  # normalize to sum=1
    
    # Accumulated cutting planes: each is (A_row, b_val) where A_row @ h <= b_val
    # The linearization of conv(h,h)[k] at point h0 is:
    # conv_lin[k] = 2 * sum_i h0[i] * h[k-i] - h0[k]^2 + h[k]^2
    # But for a *linear* constraint, we use the first-order Taylor:
    # conv(h,h)[k] >= 2 * sum_i h0[i] * h[k-i] - h0[k]^2 + h0[k]^2 = 2 * sum_i h0[i] * h[k-i]
    # Wait, that's not right. Let me think more carefully.
    #
    # f(h) = h^T R_k h where R_k is the circulant matrix for lag k
    # f(h) >= f(h0) + 2 * h0^T R_k (h - h0) = 2 * h0^T R_k h - h0^T R_k h0
    # So the cutting plane is: 2 * h0^T R_k h >= f(h0)
    # i.e., -2 * h0^T R_k h <= -f(h0)
    #
    # h0^T R_k h = sum_i h0[i] * (R_k h)[i] = sum_i h0[i] * sum_j R_k[i,j] h[j]
    # For autocorrelation at lag k: (R_k h)[i] = h[i-k] (with appropriate bounds)
    # So h0^T R_k h = sum_i h0[i] * h[i-k] (with bounds)
    #
    # Actually, let's just compute the gradient of each autocorrelation component.
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N else 0 (as derived earlier)
    # So the linearization at h0 is:
    # conv_lin[k] = conv(h0,h0)[k] + sum_j 2*h0[k-j]*(h[j]-h0[j]) for valid k-j
    # = conv(h0,h0)[k] - sum_j 2*h0[k-j]*h0[j] + sum_j 2*h0[k-j]*h[j]
    # Note: sum_j 2*h0[k-j]*h0[j] = 2*conv(h0,h0)[k] (by definition of conv)
    # Wait no: conv(h0,h0)[k] = sum_j h0[j]*h0[k-j]
    # So sum_j 2*h0[k-j]*h0[j] = 2*conv(h0,h0)[k]
    # Thus conv_lin[k] = conv(h0,h0)[k] - 2*conv(h0,h0)[k] + sum_j 2*h0[k-j]*h[j]
    # = -conv(h0,h0)[k] + sum_j 2*h0[k-j]*h[j]
    
    planes_A = []  # list of coefficient vectors for h
    planes_b = []  # list of RHS values
    
    best_h = h_curr.copy()
    best_val = _value(h_curr)
    
    for iteration in range(n_planes):
        if time.time() - t0 > max_time:
            break
        
        # Compute autocorrelation at current point
        conv_curr = _autocorr(h_curr)
        
        # For each lag k, compute the linearization coefficient vector
        # conv_lin[k] = -conv_curr[k] + sum_j 2*h_curr[k-j]*h[j] for valid j
        # The coefficient of h[j] in conv_lin[k] is: 2*h_curr[k-j] if 0<=k-j<N else 0
        
        # Add the most important cutting planes (at the max autocorrelation lags)
        # Sort by autocorrelation value, add top few
        sorted_lags = np.argsort(-conv_curr)
        
        for k in sorted_lags[:5]:
            # Coefficient vector for h
            coeff = np.zeros(N)
            for j in range(N):
                idx = k - j
                if 0 <= idx < N:
                    coeff[j] = 2.0 * h_curr[idx]
            # Linearization: conv_lin[k] = -conv_curr[k] + coeff @ h
            # Constraint: conv_lin[k] <= C (where C is our upper bound variable)
            # => coeff @ h - C <= conv_curr[k]
            # In standard form for linprog: -coeff @ h + C >= -conv_curr[k]
            # Or: coeff @ h - C <= conv_curr[k]
            
            # We'll handle this in the LP formulation below
            planes_A.append(coeff.copy())
            planes_b.append(conv_curr[k])
        
        # Solve LP: minimize C subject to sum(h)=1, h>=-M, h<=M
        # and for each plane: coeff_i @ h - C <= b_i
        # Variables: [h_0, ..., h_{N-1}, C]
        # Objective: minimize C => c = [0, 0, ..., 0, 1]
        
        c_vec = np.zeros(N + 1)
        c_vec[-1] = 1.0  # minimize C
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = np.zeros((len(planes_A), N + 1))
        b_ub = np.zeros(len(planes_A))
        for i, (coeff, b_val) in enumerate(zip(planes_A, planes_b)):
            A_ub[i, :N] = coeff
            A_ub[i, -1] = -1.0  # coeff @ h - C <= b_val
            b_ub[i] = b_val
        
        # Equality constraint: sum(h) = 1
        A_eq = np.zeros((1, N + 1))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: h_i in [-5, 5], C in [0, 10]
        bounds = [(-5.0, 5.0)] * N + [(0.0, 10.0)]
        
        try:
            result = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if result.success:
                h_new = result.x[:N]
                C_new = result.x[-1]
                
                if np.all(np.isfinite(h_new)):
                    s_new = float(np.sum(h_new))
                    if abs(s_new) > 1e-12:
                        h_new = h_new / s_new
                        v_new = _value(h_new)
                        if v_new < best_val:
                            best_val = v_new
                            best_h = h_new.copy()
                        h_curr = h_new.copy()
            else:
                break
        except Exception:
            break
    
    return best_h


def _smooth_objective(h, beta=30.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=30.0):
    """Gradient of the smooth objective using vectorized computation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    # d(soft_max)/d(h[j]) = 2 * sum_{k=j}^{j+N-1} weights[k] * h[k-j]
    # Use FFT for efficient computation
    # This is a cross-correlation between weights and h
    
    # Pad to avoid circular convolution
    size = 2 * N - 1
    w_padded = np.zeros(size)
    w_padded[:len(weights)] = weights
    
    # We need: for each j, sum_{m=0}^{N-1} weights[m+j] * h[m]
    # This is the correlation of weights with h at offset j
    # Use FFT: result[j] = sum_m weights[m+j] * h[m]
    # = sum_m weights[m+j] * h[m]
    
    # Let's use the fact that this is like a sliding dot product
    # Compute via FFT: correlate(weights, h)
    # np.correlate(weights, h, mode='valid') gives sum for each valid shift
    # But weights has length 2N-1 and h has length N
    # We want result[j] = sum_{m=0}^{N-1} weights[m+j] * h[m] for j=0..N-1
    
    # This is equivalent to: flip h, convolve with weights, take appropriate slice
    h_flipped = h[::-1]
    # Full convolution: conv_full[k] = sum_i weights[i] * h_flipped[k-i]
    # We want: for each j, sum_{m=0}^{N-1} weights[m+j] * h[m]
    # = sum_{m=0}^{N-1} weights[m+j] * h_flipped[N-1-m]
    # Let i = m+j, so m = i-j:
    # = sum_{i=j}^{j+N-1} weights[i] * h_flipped[N-1-(i-j)]
    # = sum_{i=j}^{j+N-1} weights[i] * h_flipped[N-1-j+i-j]... 
    
    # Simpler: just use np.correlate
    # np.correlate(weights, h, mode='full')[N-1:N-1+N] should give what we want
    corr = np.correlate(weights, h, mode='full')
    # The valid part starts at index N-1
    d_soft_max = 2.0 * corr[N-1:N-1+N]
    
    # Gradient of obj = |2N * soft_max / s^2|
    sign = np.sign(2.0 * N * soft_max / (s * s))
    if sign == 0:
        sign = 1.0
    
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    
    obj = abs(2.0 * N * soft_max / (s * s))
    return grad, obj


def _structured_search(max_time=20.0):
    """Search over structured patterns more thoroughly."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    
    x = np.arange(N)
    
    # Single frequency with phase
    for k in range(1, 80):
        if time.time() - t0 > max_time * 0.4:
            break
        for amp in np.linspace(0.2, 4.0, 15):
            for phase in [0, np.pi/4, np.pi/2, np.pi]:
                h = np.ones(N) / N + amp * np.cos(2 * np.pi * k * x / N + phase)
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                h = h / s
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Two frequency products
    if time.time() - t0 < max_time * 0.6:
        for k1 in range(1, 25):
            for k2 in range(k1 + 1, min(k1 + 15, 60)):
                if time.time() - t0 > max_time * 0.75:
                    break
                for amp in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
                    h = np.ones(N) / N + amp * np.cos(2*np.pi*k1*x/N) * np.cos(2*np.pi*k2*x/N)
                    s = np.sum(h)
                    if abs(s) < 1e-12:
                        continue
                    h = h / s
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    # Polynomial bases
    if time.time() - t0 < max_time * 0.8:
        t = x / (N - 1) - 0.5
        for deg in [2, 3, 4, 5, 6, 8]:
            for amp in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                h = np.ones(N) / N + amp * t ** deg
                s = np.sum(h)
                if abs(s) < 1e-12:
                    continue
                h = h / s
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Windowed patterns (Gaussian)
    if time.time() - t0 < max_time * 0.9:
        for sigma in [10, 20, 30, 50, 80]:
            for center in [N//4, N//2, 3*N//4]:
                for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    h = np.ones(N) / N + amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
                    s = np.sum(h)
                    if abs(s) < 1e-12:
                        continue
                    h = h / s
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=30.0):
    """Local refinement using smooth objective."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [20.0, 40.0, 80.0, 150.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=beta)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Phase 1: Structured search for good initial candidates
    h_struct, val_struct = _structured_search(max_time=12.0)
    
    # Also include the seed
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    
    if v_seed < val_struct:
        best_h = h_seed.copy()
        best_val = v_seed
    else:
        best_h = h_struct.copy()
        best_val = val_struct
    
    # Phase 2: Cutting Plane Method (structural change)
    if time_left() > 40:
        h_ccp, val_ccp = _ccp_iterate(best_h, max_time=min(30.0, time_left() - 20.0), n_planes=15)
        if val_ccp < best_val:
            best_val = val_ccp
            best_h = h_ccp.copy()
    
    # Phase 3: Local refinement from best candidate
    if time_left() > 35:
        best_h, best_val = _local_refine(best_h, max_time=min(35.0, time_left() - 15.0))
    
    # Phase 4: Another round of CCM from the refined point
    if time_left() > 25:
        h_ccp2, val_ccp2 = _ccp_iterate(best_h, max_time=min(20.0, time_left() - 10.0), n_planes=10)
        if val_ccp2 < best_val:
            best_val = val_ccp2
            best_h = h_ccp2.copy()
    
    # Phase 5: Final high-precision polish
    if time_left() > 12:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=200.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=200.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
