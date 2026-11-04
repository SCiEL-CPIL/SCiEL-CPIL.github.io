import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _l2_autoconv_energy(x):
    """L2 norm of autoconvolution - smooth surrogate for optimization."""
    c = _autoconv_lin(x)
    return float(np.sum(c * c))

def _l2_grad(x):
    """Gradient of L2 autoconvolution energy w.r.t. x."""
    n = len(x)
    c = _autoconv_lin(x)
    # d/dx_i sum_k c_k^2 = 2 * sum_k c_k * dc_k/dx_i
    # dc_k/dx_i = x_{i-k} + x_{i+k} (with boundary handling)
    grad = np.zeros(n)
    for k in range(len(c)):
        ck = c[k]
        if abs(ck) < 1e-15:
            continue
        for i in range(n):
            contrib = 0.0
            if i >= k and i - k < n:
                contrib += x[i - k]
            if k > 0 and i + k < n:
                contrib += x[i + k]
            grad[i] += 2.0 * ck * contrib
    return grad

def _continuous_relaxation(x0, max_iters=200, lr0=0.01, deadline=None):
    """
    Continuous gradient descent on L2 autoconvolution energy.
    Returns the best continuous solution found.
    """
    x = x0.copy().astype(float)
    best_x = x.copy()
    best_energy = _l2_autoconv_energy(x)
    lr = lr0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient
        grad = _l2_grad(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Normalize gradient and take step
        x_new = x - lr * grad / grad_norm
        
        # Evaluate
        e_new = _l2_autoconv_energy(x_new)
        if e_new < best_energy:
            best_energy = e_new
            best_x = x_new.copy()
            x = x_new
            # Decay learning rate slowly
            lr = max(0.001, lr * 0.999)
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return best_x

def _project_to_binary(x):
    """Project continuous vector to binary {-1, 1} via thresholding."""
    return np.sign(x).astype(float)
    # Handle zeros
    # x[np.abs(x) < 1e-10] = 1.0
    # return x

def _binary_local_search(x, max_iters=500, deadline=None):
    """
    Discrete local search: flip individual bits if it improves the metric.
    """
    x = x.copy()
    current_val = _metric_std(x)
    n = len(x)
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random position
        i = rng.integers(0, n)
        x_trial = x.copy()
        x_trial[i] *= -1.0
        
        v_trial = _metric_std(x_trial)
        if v_trial < current_val:
            x = x_trial
            current_val = v_trial
    
    return x, current_val

def _spectral_flattening(x, n_iters=5):
    """
    Iteratively flatten the spectrum to reduce autocorrelation peaks.
    """
    x = x.copy()
    for _ in range(n_iters):
        fx = np.fft.fft(x)
        mag = np.abs(fx)
        phase = np.angle(fx)
        # Target: flat magnitude
        target_mag = np.ones_like(mag)
        # Blend to avoid abrupt changes
        alpha = 0.5
        new_mag = (1 - alpha) * mag + alpha * target_mag
        fx_new = new_mag * np.exp(1j * phase)
        x = np.real(np.fft.ifft(fx_new))
        # Project to binary
        x = np.sign(x)
        x[x == 0] = 1.0
    return x

def _generate_initial_candidates(rng, N):
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. Constant
    candidates.append(np.ones(N))
    
    # 2. Alternating
    x_alt = np.ones(N)
    x_alt[1::2] = -1.0
    candidates.append(x_alt)
    
    # 3. Random signs
    for _ in range(3):
        x_rand = rng.choice([-1.0, 1.0], size=N).astype(float)
        candidates.append(x_rand)
    
    # 4. Spectrally flattened random
    x_flat = _spectral_flattening(rng.choice([-1.0, 1.0], size=N).astype(float), n_iters=10)
    candidates.append(x_flat)
    
    # 5. Block patterns
    for block_size in [10, 20, 50]:
        n_blocks = N // block_size
        x_block = np.tile(rng.choice([-1.0, 1.0], size=n_blocks), N // block_size)
        if len(x_block) < N:
            x_block = np.concatenate([x_block, np.ones(N - len(x_block))])
        candidates.append(x_block[:N])
    
    return candidates

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(rng, N)
    
    best_x = None
    best_val = float('inf')
    
    # Phase 1: Evaluate initial candidates and pick best
    for cand in candidates:
        if time.time() > deadline - 60:
            break
        v = _metric_std(cand)
        if v < best_val:
            best_val = v
            best_x = cand.copy()
    
    if best_x is None:
        return np.ones(N)
    
    # Phase 2: For several top candidates, do continuous relaxation then binary search
    # Sort candidates by metric
    cand_scores = [(cand, _metric_std(cand)) for cand in candidates]
    cand_scores.sort(key=lambda x: x[1])
    
    # Take top 3 candidates for deeper optimization
    top_candidates = [c for c, v in cand_scores[:3]]
    
    for cand in top_candidates:
        if time.time() > deadline - 30:
            break
        
        # Continuous relaxation
        x_cont = _continuous_relaxation(cand, max_iters=150, lr0=0.01, deadline=deadline - 10)
        
        # Project to binary
        x_bin = _project_to_binary(x_cont)
        v_bin = _metric_std(x_bin)
        
        # Binary local search
        x_ls, v_ls = _binary_local_search(x_bin, max_iters=300, deadline=deadline - 5)
        
        if v_ls < best_val:
            best_val = v_ls
            best_x = x_ls
    
    # Phase 3: Also try direct binary local search from best candidate
    if time.time() < deadline - 10:
        x_ls2, v_ls2 = _binary_local_search(best_x, max_iters=500, deadline=deadline - 2)
        if v_ls2 < best_val:
            best_val = v_ls2
            best_x = x_ls2
    
    return best_x
