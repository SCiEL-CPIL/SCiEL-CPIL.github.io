import numpy as np
import time
from scipy.optimize import minimize

N = 100
WIDTH = 2.0 / N
TARGET_SUM = N / 2.0

def compute_overlaps(h):
    """
    Compute overlaps O_s for s=1..N-1.
    O_s = sum(h_i * (1 - h_{i+s})) * WIDTH
    """
    # Autocorrelation C_s = sum(h_i * h_{i+s})
    # Using FFT for speed
    fft_h = np.fft.fft(h, n=2 * N)
    corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
    # corr[s] is C_s
    # O_s = WIDTH * (TARGET_SUM - corr[s])
    overlaps = WIDTH * (TARGET_SUM - corr[1:])
    return overlaps

def smooth_max(overlaps, beta):
    """
    Compute smooth max and gradient weights.
    """
    m = np.max(overlaps)
    exps = np.exp(beta * (overlaps - m))
    sum_exps = np.sum(exps)
    val = m + np.log(sum_exps) / beta
    weights = exps / sum_exps
    return val, weights

def project_to_polytope(h, S):
    """
    Project h onto { x : sum(x) = S, 0 <= x_i <= 1 }
    """
    if S < 0 or S > N:
        return np.clip(h, 0, 1)
        
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h - mid, 0.0, 1.0))
        if val > S:
            lo = mid
        else:
            hi = mid
            
    lambda_star = (lo + hi) / 2.0
    return np.clip(h - lambda_star, 0.0, 1.0)

def grad_smooth_max(h, beta):
    """
    Compute gradient of smooth max of overlaps w.r.t h.
    """
    overlaps = compute_overlaps(h)
    val, weights = smooth_max(overlaps, beta)
    
    # Gradient of O_s w.r.t h_j:
    # O_s = WIDTH * (TARGET_SUM - sum_i h_i h_{i+s})
    # dO_s/dh_j = -WIDTH * (h_{j+s} + h_{j-s})
    
    # Total gradient = sum_s weights[s] * dO_s/dh
    # = -WIDTH * sum_s weights[s] * (h_{j+s} + h_{j-s})
    
    # Note: overlaps array corresponds to s=1..N-1.
    # weights is of length N-1.
    # We need to sum over s=1..N-1.
    
    grad = np.zeros(N)
    for s in range(1, N):
        w = weights[s-1]
        if w > 1e-10:
            # h_{j+s} is np.roll(h, -s)
            # h_{j-s} is np.roll(h, s)
            grad -= WIDTH * w * (np.roll(h, -s) + np.roll(h, s))
            
    return val, grad

def slsqp_objective(h):
    """
    Objective for SLSQP: max overlap.
    We minimize the negative of the max overlap? No, we minimize the max overlap.
    SLSQP minimizes. So we return the max overlap.
    To make it smooth, we use a high beta smooth max, but SLSQP can handle non-smooth if we are careful, 
    but better to use smooth max with high beta or just the actual max if we use subgradient? 
    Inspiration 1 used smooth max. Let's use smooth max with high beta for SLSQP.
    """
    overlaps = compute_overlaps(h)
    beta = 50.0
    val, _ = smooth_max(overlaps, beta)
    return val

def slsqp_gradient(h):
    """
    Gradient for SLSQP.
    """
    overlaps = compute_overlaps(h)
    beta = 50.0
    _, weights = smooth_max(overlaps, beta)
    
    grad = np.zeros(N)
    for s in range(1, N):
        w = weights[s-1]
        if w > 1e-10:
            grad -= WIDTH * w * (np.roll(h, -s) + np.roll(h, s))
    return grad

def run_pgd(h0, max_steps=200, beta_start=1.0, beta_end=50.0, alpha_start=0.05, alpha_end=0.005):
    """
    Run Projected Gradient Descent.
    """
    h = h0.copy()
    best_h = h.copy()
    best_score = np.max(compute_overlaps(h))
    
    for step in range(max_steps):
        # Anneal beta and alpha
        progress = step / max_steps
        beta = beta_start + (beta_end - beta_start) * progress
        alpha = alpha_start + (alpha_end - alpha_start) * progress
        
        val, grad = grad_smooth_max(h, beta)
        
        # Update
        h = h - alpha * grad
        h = project_to_polytope(h, TARGET_SUM)
        
        # Check actual score
        actual_score = np.max(compute_overlaps(h))
        if actual_score < best_score:
            best_score = actual_score
            best_h = h.copy()
            
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = np.max(compute_overlaps(best_h))
    
    rng = np.random.default_rng(42)
    
    # Generate initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Binary-like patterns (blocks)
    # Try different block sizes
    for block_size in [2, 4, 5, 10, 20, 25]:
        if N % block_size == 0:
            # Alternating blocks
            pattern = np.tile(np.array([1.0] * (block_size//2) + [0.0] * (block_size//2)), N // block_size)
            if np.sum(pattern) == TARGET_SUM:
                inits.append(pattern)
            else:
                # Adjust to meet sum
                pattern = project_to_polytope(pattern, TARGET_SUM)
                inits.append(pattern)
        
        # Random binary patterns
        for _ in range(3):
            p = rng.random(N)
            h_bin = (p > 0.5).astype(float)
            # Adjust sum
            h_bin = project_to_polytope(h_bin, TARGET_SUM)
            inits.append(h_bin)
            
    # 3. Sinusoidal patterns
    for freq in [1, 2, 3, 5, 7]:
        for amp in [0.3, 0.4, 0.45]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = project_to_polytope(h, TARGET_SUM)
            inits.append(h)
            
    # 4. Random perturbations
    for _ in range(10):
        h = 0.5 + 0.1 * rng.standard_normal(N)
        h = project_to_polytope(h, TARGET_SUM)
        inits.append(h)
        
    # Run PGD on each initialization
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        h_opt, score_opt = run_pgd(h0)
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
    # Polish the best result with SLSQP
    if time.time() < deadline - 10.0:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM}]
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(slsqp_objective, best_h, jac=slsqp_gradient, method='SLSQP', 
                           bounds=bounds, constraints=constraints, 
                           options={'maxiter': 200, 'ftol': 1e-9})
            if res.success:
                h_slsqp = res.x
                # Ensure constraints
                h_slsqp = project_to_polytope(h_slsqp, TARGET_SUM)
                score_slsqp = np.max(compute_overlaps(h_slsqp))
                if score_slsqp < best_score:
                    best_score = score_slsqp
                    best_h = h_slsqp.copy()
        except Exception:
            pass
            
    # Final check and return
    # Ensure sum is exactly TARGET_SUM
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-10:
        mask = (best_h > 0) & (best_h < 1)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
            
    return best_h
