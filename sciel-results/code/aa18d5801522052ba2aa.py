"""Spectral smooth optimization for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0

def _compute_overlaps(h):
    """
    Compute O_k = N/2 - autocorr(k) for k=1..N-1.
    Uses FFT for efficient circular autocorrelation.
    """
    # autocorr[k] = sum_i h[i]*h[(i+k)%N]
    # Using FFT: autocorr = ifft(fft(h) * conj(fft(h))).real
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # We only care about k=1..N-1 for the "shifted" overlap
    # autocorr[0] is sum(h^2), which is not a shift
    # autocorr is circular, so autocorr[k] for k > N/2 corresponds to negative shifts
    # But overlap is symmetric: O_k = O_{N-k}
    # The task asks for overlap with "shifted 1-h". Usually implies all non-zero shifts.
    # The current program checks all k.
    # Let's return all k=1..N-1.
    return TARGET_SUM - autocorr[1:]

def _objective_and_grad(h, beta):
    """
    Compute smooth max of overlaps and its gradient.
    J(h) = (1/beta) * log(sum_k exp(beta * O_k))
    """
    overlaps = _compute_overlaps(h)
    
    # Numerical stability for log-sum-exp
    max_ov = np.max(overlaps)
    if max_ov > 500 / beta:
        # Scale to prevent overflow
        # Actually, if beta is large, beta*O_k can be large.
        # We can just work with O_k directly in the log-sum-exp formula if we are careful.
        # log(sum(exp(beta*O_k))) = beta*max_O + log(sum(exp(beta*(O_k - max_O))))
        # J = max_O + (1/beta)*log(sum(exp(beta*(O_k - max_O))))
        shifted = beta * (overlaps - max_ov)
        log_sum = np.log(np.sum(np.exp(shifted)))
        obj = max_ov + log_sum / beta
        
        # Weights w_k = exp(beta*O_k) / sum(exp(beta*O_k))
        # w_k = exp(shifted_k) / sum(exp(shifted_k))
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)
    else:
        shifted = beta * overlaps
        log_sum = np.log(np.sum(np.exp(shifted)))
        obj = log_sum / beta
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)

    # Gradient
    # dJ/dh_j = sum_k weights[k] * dO_k/dh_j
    # O_k = N/2 - sum_i h_i h_{i+k}
    # dO_k/dh_j = -(h_{j+k} + h_{j-k}) (circular)
    
    # We need to compute G_j = sum_{k=1}^{N-1} weights[k] * (h_{j+k} + h_{j-k})
    # Then grad_j = -G_j
    
    # Efficient computation using FFT?
    # G = ifft(fft(weights_padded) * fft(h_padded)) ... complex.
    # N=100 is small. Direct loop is fine and less error-prone.
    grad = np.zeros(N)
    # weights corresponds to k=1..N-1
    # Let's iterate k
    # For each k, w = weights[k-1]
    # Contribution to grad[j] is -w * (h[(j+k)%N] + h[(j-k)%N])
    
    # Vectorized approach:
    # Create a matrix W of size N x (N-1) where W[j, k-1] = weights[k-1]
    # Then G = W @ (shifts)
    # This might be memory heavy or slow if not done right.
    # Let's use the property that this is a weighted sum of circular shifts.
    
    # G_j = sum_k w_k h_{j+k} + sum_k w_k h_{j-k}
    # Let W be the weights vector (size N-1).
    # We can compute the first term: S1_j = sum_k w_k h_{j+k}
    # This is a circular convolution of h with w?
    # If we pad w to size N: w_padded = [w_1, ..., w_{N-1}, 0]
    # Then S1 = ifft(fft(h) * fft(w_padded)).real ?
    # Check: (h * w_padded)_j = sum_i h_i w_{j-i}
    # We want sum_k w_k h_{j+k}. Let i = j+k => k = i-j.
    # sum_i h_i w_{i-j}. This is correlation/convolution.
    
    # Let's just use loops for clarity and safety, N=100 is tiny.
    # But 100 iterations * 100 ops = 10k ops. Very fast.
    
    # Actually, we can do it with np.roll
    # Term 1: sum_k w_k h_{j+k}
    # For each k, add w_k * h shifted by -k (so h_{j+k} aligns with j)
    # h_shifted_k = np.roll(h, -k) -> h_shifted_k[j] = h[(j+k)%N]
    
    # To be faster, we can use FFT for the weighted sum.
    # w_full = np.zeros(N)
    # w_full[1:N] = weights  # k=1..N-1
    # Note: weights[k-1] corresponds to shift k.
    # So w_full[k] = weights[k-1] for k=1..N-1.
    
    w_full = np.zeros(N)
    w_full[1:] = weights
    
    # S1_j = sum_k w_full[k] h_{j+k}
    # This is the circular cross-correlation of h and w_full?
    # Let's verify:
    # conv(h, w_full)[j] = sum_i h[i] w_full[j-i]
    # We want sum_k w_full[k] h[j+k].
    # Let i = j+k => k = i-j.
    # sum_i h[i] w_full[i-j].
    # This is (w_full * h)_j if we define convolution as sum h[i] w[j-i]?
    # Standard convolution: (f*g)[n] = sum_k f[k] g[n-k].
    # Here we have sum_i h[i] w_full[i-j].
    # Let m = i-j => i = m+j.
    # sum_m h[m+j] w_full[m].
    # This is the correlation.
    # Using FFT:
    # corr = ifft(fft(h) * conj(fft(w_full))).real
    # Let's check indices.
    # If w_full is [0, w1, w2...], then corr[j] = sum_i h[i] w_full[(i-j)%N]
    # We want sum_k w_full[k] h[(j+k)%N].
    # Let i = j+k. sum_i w_full[(i-j)%N] h[i].
    # Yes, this matches.
    
    S1 = np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(w_full))).real
    
    # Term 2: S2_j = sum_k w_k h_{j-k}
    # Let i = j-k => k = j-i.
    # sum_i w_full[(j-i)%N] h[i].
    # This is the reverse correlation or convolution with reversed w.
    # w_rev = w_full[::-1] ... careful with circular.
    # w_full is [0, w1, ..., w_{N-1}].
    # We want sum_k w_full[k] h[j-k].
    # This is (h * w_full)[j] in standard convolution?
    # (h * w_full)[j] = sum_i h[i] w_full[j-i].
    # Yes.
    # So S2 = ifft(fft(h) * fft(w_full)).real
    
    S2 = np.fft.ifft(np.fft.fft(h) * np.fft.fft(w_full)).real
    
    grad = -(S1 + S2)
    
    return obj, grad

def _project(h):
    """Project h to sum=N/2 and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = TARGET_SUM - np.sum(h)
    # Iterative projection
    for _ in range(10):
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-9) & (h < 1 - 1e-9)
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
        diff = TARGET_SUM - np.sum(h)
    # Final exact adjustment
    if abs(np.sum(h) - TARGET_SUM) > 1e-10:
        # Distribute remaining error to all non-bound elements
        mask = (h > 1e-9) & (h < 1 - 1e-9)
        if np.any(mask):
            h[mask] += (TARGET_SUM - np.sum(h)) / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = float('inf')
    
    def _score(h):
        overlaps = _compute_overlaps(h)
        return np.max(overlaps)
    
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    
    # 1. Uniform
    restarts.append(np.full(N, 0.5))
    
    # 2. Random
    for _ in range(20):
        restarts.append(np.random.rand(N))
        
    # 3. Sinusoidal
    x = np.linspace(0, 2*np.pi, N)
    for freq in [1, 2, 3, 4, 5, 10, 20]:
        for amp in [0.2, 0.4, 0.49]:
            restarts.append(np.clip(0.5 + amp * np.sin(freq * x), 0, 1))
            restarts.append(np.clip(0.5 + amp * np.cos(freq * x), 0, 1))
            
    # 4. Block patterns (bang-bang approx)
    for block_size in [2, 5, 10, 20, 25, 50]:
        if block_size < N:
            pattern = np.tile([0.0, 1.0], N // block_size + 1)[:N]
            # Scale to have mean 0.5
            # pattern has 0.5 mean already if N is even and block_size divides N nicely
            # Otherwise project
            restarts.append(pattern)

    # Project all restarts
    projected_restarts = []
    for r in restarts:
        pr = _project(r.copy())
        projected_restarts.append(pr)
        
    # Multi-stage beta optimization
    betas = [5, 20, 50, 100, 200]
    
    for x0 in projected_restarts:
        if time.time() - start_time > deadline:
            break
            
        current_h = x0.copy()
        
        for beta in betas:
            if time.time() - start_time > deadline:
                break
                
            # Define objective and gradient for SLSQP
            def obj_fun(h):
                val, _ = _objective_and_grad(h, beta)
                return val
                
            def grad_fun(h):
                _, g = _objective_and_grad(h, beta)
                return g
                
            # Constraints
            # Equality: sum(h) - N/2 = 0
            def eq_constr(h):
                return np.sum(h) - TARGET_SUM
                
            def eq_constr_jac(h):
                return np.ones(N)
                
            # Bounds
            bounds = [(0, 1)] * N
            
            try:
                res = minimize(
                    obj_fun, 
                    current_h, 
                    jac=grad_fun, 
                    method='SLSQP', 
                    bounds=bounds,
                    constraints=[{'type': 'eq', 'fun': eq_constr, 'jac': eq_constr_jac}],
                    options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
                )
                
                if res.success:
                    current_h = res.x
                else:
                    # If it failed, keep previous or try to project
                    pass
                    
            except Exception:
                continue
                
            # Project to ensure constraints are met exactly
            current_h = _project(current_h)
            
            # Update best
            score = _score(current_h)
            if score < best_score:
                best_score = score
                best_h = current_h.copy()
                
    # Final verification and projection
    best_h = _project(best_h)
    
    return best_h
