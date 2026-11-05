import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 138.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=40.0):
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


def _smooth_grad(h, beta=40.0):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    wsum = np.sum(weights)
    weights /= wsum
    soft_max = m + np.log(wsum) / beta
    val = 2.0 * N * soft_max / (s * s)
    obj = abs(val)
    
    # d(soft_max)/d(h[j]) = 2 * sum_k weights[k] * h[k-j] for valid k-j
    # Use FFT-based approach for speed
    # d_soft_max[j] = 2 * sum_{k=j}^{min(j+N-1, 2N-2)} weights[k] * h[k-j]
    # This is a cross-correlation of weights (reversed) with h, but only for the full conv part
    
    # Actually: conv[k] = sum_i h[i]*h[k-i] for valid indices
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] (if both valid)
    # But since we're doing full conv, d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, plus h[j-k] if 0<=j-k<N
    # Wait, for full convolution: conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] (if k-j in [0,N-1]) + h[j-k] (if j-k in [0,N-1])
    # But h[k-j] and h[j-k] are the same term when we swap, so actually:
    # d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, else 0, PLUS h[j-k] if 0<=j-k<N, else 0
    # But these are symmetric, so it's 2*h[k-j] when 0<=k-j<N (for k!=j case)
    # Actually no. Let me be more careful.
    
    # conv[k] = sum_i h[i] * h[k-i] where 0<=i<N and 0<=k-i<N
    # d(conv[k])/d(h[j]) = h[k-j] (from the first factor being h[j]) + h[j-k] (from the second factor being h[j])
    # But h[k-j] is only valid if 0<=k-j<N, and h[j-k] is only valid if 0<=j-k<N
    # Note that if 0<=k-j<N then k-j >= 0, and if 0<=j-k<N then j-k >= 0, these are mutually exclusive unless k=j
    # When k=j: d(conv[j])/d(h[j]) = h[0] + h[0] = 2*h[0]... wait no
    # When i=j in the sum: term is h[j]*h[k-j], derivative w.r.t h[j] is h[k-j]
    # When k-i=j, i.e., i=k-j: term is h[k-j]*h[j], derivative w.r.t h[j] is h[k-j]
    # So if j != k-j (i.e., k != 2j), we get two separate terms, each giving h[k-j] or h[j-k]
    # But h[k-j] when k-j is in range, and h[j-k] when j-k is in range
    # Since h is indexed 0..N-1, k-j in [0,N-1] means the first term exists
    # j-k in [0,N-1] means the second term exists
    # These can both be true only if k-j >= 0 and j-k >= 0, i.e., k=j, giving h[0]+h[0]=2h[0]
    # Wait no, when k=j: first term h[k-j]=h[0], second term h[j-k]=h[0], so total 2*h[0]
    # When k != j: only one of k-j or j-k is non-negative (or both could be if... no)
    # If k > j: k-j > 0, j-k < 0, so only first term: h[k-j] (if k-j < N)
    # If k < j: k-j < 0, j-k > 0, so only second term: h[j-k] (if j-k < N)
    
    # So d(conv[k])/d(h[j]) = h[k-j] if 0 <= k-j < N, else 0, PLUS h[j-k] if 0 <= j-k < N, else 0
    # But when k=j, this gives h[0] + h[0] = 2*h[0]
    
    # d(soft_max)/d(h[j]) = sum_k weights[k] * d(conv[k])/d(h[j])
    # = sum_k weights[k] * (h[k-j] * I[0<=k-j<N] + h[j-k] * I[0<=j-k<N])
    
    # For the first part: sum_k weights[k] * h[k-j] * I[0<=k-j<N]
    #   = sum_{d=0}^{N-1} weights[j+d] * h[d] (if j+d <= 2N-2)
    # For the second part: sum_k weights[k] * h[j-k] * I[0<=j-k<N]
    #   = sum_{d=0}^{N-1} weights[j-d] * h[d] (if j-d >= 0)
    
    # So d(soft_max)/d(h[j]) = sum_{d=0}^{min(N-1, 2N-2-j)} weights[j+d] * h[d] + sum_{d=0}^{min(N-1, j)} weights[j-d] * h[d]
    
    # This is a convolution/correlation. Let's use FFT.
    # Part 1: cross-correlation of weights with h, but shifted
    # Part 1[j] = sum_d weights[j+d] * h[d] for d=0..min(N-1, 2N-2-j)
    # Part 2[j] = sum_d weights[j-d] * h[d] for d=0..min(N-1, j)
    
    # Part 2 is just the standard cross-correlation (weights * h) at lag j
    # Part 1 is like a flipped cross-correlation
    
    # Let's compute using FFT for efficiency
    # Create padded arrays
    size = 2 * N  # power of 2 >= 2N-1
    size = 1 << (2 * N - 1).bit_length()
    
    H = np.fft.rfft(h, size)
    W = np.fft.rfft(weights, size)
    
    # Part 2: sum_d weights[j-d] * h[d] = (weights * h)[j] in correlation sense
    # This is np.correlate(weights, h, 'full')[N-1+j] or similar
    # Actually, let's think of it as: result2[j] = sum_d h[d] * weights[j-d]
    # This is the convolution of h and weights evaluated at j
    conv_hw = np.fft.irfft(H * W, size)
    # conv_hw[k] = sum_i h[i] * weights[k-i]
    # So part2[j] = conv_hw[j] for j in 0..N-1 (but we need weights[j-d] to be valid, i.e., 0<=j-d<N)
    # conv_hw[j] = sum_i h[i]*weights[j-i], and weights is only nonzero for indices 0..2N-2
    # So conv_hw[j] = sum_{i=0}^{N-1} h[i]*weights[j-i] where 0<=j-i<=2N-2
    # For j in 0..N-1, j-i >= 0 means i <= j, and j-i <= 2N-2 is always true for j<N
    # So part2[j] = sum_{i=0}^{j} h[i]*weights[j-i] = sum_{d=0}^{j} h[d]*weights[j-d] ✓
    
    # Part 1: sum_d weights[j+d] * h[d] for d=0..min(N-1, 2N-2-j)
    # Let's reverse weights: wr[k] = weights[2N-2-k]
    # Then weights[j+d] = wr[2N-2-j-d]
    # sum_d h[d] * wr[2N-2-j-d] = (h * wr)[2N-2-j]
    wr = weights[::-1]
    WR = np.fft.rfft(wr, size)
    conv_hwr = np.fft.irfft(H * WR, size)
    # conv_hwr[k] = sum_i h[i]*wr[k-i]
    # part1[j] = conv_hwr[2N-2-j]
    # But we need to check bounds: d from 0 to min(N-1, 2N-2-j)
    # conv_hwr[2N-2-j] = sum_{i=0}^{N-1} h[i]*wr[2N-2-j-i]
    # wr[2N-2-j-i] = weights[j+i] (since wr[k]=weights[2N-2-k])
    # So conv_hwr[2N-2-j] = sum_{i=0}^{N-1} h[i]*weights[j+i]
    # This is valid when 0<=j+i<=2N-2, i.e., -j<=i<=2N-2-j
    # For i in 0..N-1 and j in 0..N-1, j+i ranges from j to j+N-1, which is in [0, 2N-2]
    # So part1[j] = conv_hwr[2N-2-j] ✓
    
    d_soft_max = 2.0 * (conv_hw[:N] + conv_hwr[2*N-2:N-1:-1])
    
    # d(obj)/d(h[j]) = sign(val) * 2N * (d_soft_max[j]/s^2 - 2*soft_max/s^3)
    sign = 1.0 if val >= 0 else -1.0
    
    grad = sign * 2.0 * N * (d_soft_max / (s * s) - 2.0 * soft_max / (s ** 3))
    return grad, obj


def _diverse_init():
    """Generate diverse initial candidates and return the best one."""
    x = np.arange(N)
    candidates = []
    
    # Uniform baseline
    candidates.append(np.ones(N) / N)
    
    # Single frequency cosine
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            h = np.ones(N) / N + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Chirp patterns
    for freq in [20, 30, 50, 75, 100, 150]:
        for chirp_rate in [-10, -5, -1, 0, 1, 5, 10]:
            for dc_bias in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for amp in [1.0, 2.0, 3.0]:
                    h = dc_bias + amp * np.cos(2 * np.pi * (freq * x / N + chirp_rate * x * x / (N * N)))
                    s = np.sum(h)
                    if abs(s) > 1e-12:
                        candidates.append(h / s)
    
    # Two-frequency products
    for k1 in [1, 2, 3, 5, 7, 10]:
        for k2 in [11, 13, 17, 20, 25, 30, 40, 50]:
            for amp in [0.5, 1.0, 1.5, 2.0]:
                h = np.ones(N) / N + amp * np.cos(2*np.pi*k1*x/N) * np.cos(2*np.pi*k2*x/N)
                s = np.sum(h)
                if abs(s) > 1e-12:
                    candidates.append(h / s)
    
    # Anti-periodic patterns
    for period in [2, 4, 8, 16]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                h[i] = 1.0 if (i + phase) % period < period // 2 else -1.0
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Find best
    best_h = None
    best_val = 1e18
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=30.0):
    """Local refinement using smooth objective with L-BFGS-B."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    # Try different beta values
    for beta in [25.0, 40.0, 60.0, 100.0]:
        if time.time() - t0 > max_time * 0.75:
            break
        try:
            res = minimize(
                lambda h, b=beta: _smooth_objective(h, beta=b),
                best_h,
                jac=lambda h, b=beta: _smooth_grad(h, beta=b)[0],
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start_local(h_best, n_starts=12, max_time=40.0):
    """Multi-start local refinement with diverse perturbations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(12345)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.88:
            break
        
        scale = 0.03 + 0.06 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=40.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=40.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _chirp_grid_search(max_time=20.0):
    """Grid search over chirp parameters."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    x = np.arange(N)
    
    # Coarse grid
    for freq in range(5, 200, 5):
        if time.time() - t0 > max_time * 0.55:
            break
        for chirp_rate in np.linspace(-15, 15, 7):
            for dc_bias in np.linspace(-1.0, 1.0, 5):
                for amp in np.linspace(0.5, 3.0, 6):
                    h = dc_bias + amp * np.cos(2 * np.pi * (freq * x / N + chirp_rate * x * x / (N * N)))
                    s = np.sum(h)
                    if abs(s) < 1e-12:
                        continue
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = 2.0
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Diverse initialization
    h_div, val_div = _diverse_init()
    
    # Phase 2: Chirp grid search
    h_chirp, val_chirp = _chirp_grid_search(max_time=min(18.0, time_left() - 35.0))
    
    # Pick best
    if val_chirp <= val_div:
        best_h = h_chirp.copy()
        best_val = val_chirp
    else:
        best_h = h_div.copy()
        best_val = val_div
    
    # Include seed
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    # Phase 3: Local refinement
    if time_left() > 35:
        best_h, best_val = _local_refine(best_h, max_time=min(30.0, time_left() - 20.0))
    
    # Phase 4: Multi-start local refinement
    if time_left() > 25:
        best_h, best_val = _multi_start_local(best_h, n_starts=12, max_time=min(35.0, time_left() - 12.0))
    
    # Phase 5: Final high-precision polish with higher beta
    if time_left() > 12:
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=100.0),
                best_h,
                jac=lambda h: _smooth_grad(h, beta=100.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 600, 'maxfun': 1500, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead polish
    if time_left() > 6:
        try:
            res = minimize(
                _value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
