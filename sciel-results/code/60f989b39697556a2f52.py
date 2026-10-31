import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50
DEADLINE_MARGIN = 10.0  # Return 10s before the 150s kill
SOFTMAX_TEMPERATURE = 0.01 # Temperature for smooth max approximation

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    The ratio is homogeneous of degree 0, so scaling h does not change the value.
    We assume h is non-negative.
    """
    h = np.asarray(h, dtype=np.float64)
    # Enforce non-negativity for safety, though optimizers should maintain it
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    # Normalize to sum=1 to improve numerical stability and simplify ratio
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    # Full convolution
    # Using np.convolve for exactness. N=50 is small.
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    
    if conv_inf < 1e-15:
        return 0.0
        
    return h2_sq / conv_inf

def smooth_max(x, temp=1.0):
    """
    Approximate max(x) using log-sum-exp for smoothness.
    max(x) ~ temp * log(sum(exp(x/temp)))
    """
    x_max = np.max(x)
    # Numerical stability: subtract max
    return x_max + temp * np.log(np.sum(np.exp((x - x_max) / temp)))

def neg_ratio_smooth(params, temp=SOFTMAX_TEMPERATURE):
    """
    Objective function for L-BFGS-B using a smooth surrogate for the max.
    We parameterize h using log-space to ensure positivity: h = exp(params).
    """
    h = np.exp(params)
    # Check for overflow
    if not np.all(np.isfinite(h)):
        return 1e10
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    
    # Use smooth max for the denominator to provide gradients
    conv_soft_max = smooth_max(conv, temp)
    
    if conv_soft_max < 1e-15:
        return 1e10
        
    return -(h2_sq / conv_soft_max)

def neg_ratio(params):
    """
    Objective function for L-BFGS-B.
    We parameterize h using log-space to ensure positivity: h = exp(params).
    Since the ratio is scale-invariant, we can just compute the ratio of exp(params).
    However, L-BFGS-B works better if the objective is somewhat well-scaled.
    Let's normalize inside the objective to keep values bounded.
    """
    h = np.exp(params)
    # Check for overflow
    if not np.all(np.isfinite(h)):
        return 1e10
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    h_norm = h / l1
    return -compute_ratio(h_norm)

def neg_ratio_normalized(x):
    """
    Objective for SLSQP on normalized vector x (sum=1, x>=0).
    """
    h = np.maximum(x, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    h_norm = h / s
    return -compute_ratio(h_norm)

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    def elapsed():
        return time.time() - t0

    # Initial best: single spike
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []
    
    # Helper to add candidate
    def add_candidate(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-15:
            return
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))
        
    # --- Phase 1: Generate Structured Candidates ---
    
    # 1. Single spike (already set)
    add_candidate(best_h)
    
    # 2. Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            add_candidate(h)
            if elapsed() > deadline - 5:
                return list(best_h)
                
    # 3. Three spikes
    # Sample a subset of triplets to save time
    # Focus on "spread out" triplets
    for i in range(0, N, 2):
        for j in range(i+2, N, 2):
            for k in range(j+2, N, 2):
                h = np.zeros(N)
                h[i] = 1.0/3
                h[j] = 1.0/3
                h[k] = 1.0/3
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 4. Sidon-like sets
    # A Sidon set ensures all pair sums are distinct, minimizing collisions in convolution.
    # This tends to lower ||h*h||_inf relative to ||h||_2^2.
    
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        """
        Greedy construction of a Sidon set.
        """
        A = []
        used_sums = set()
        # We want to pick indices from 0 to N-1
        for x in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x in used_sums:
                valid = False
            if valid:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
        return A

    # Try different steps and starts to find diverse Sidon sets
    for step in [1, 2, 3]:
        for start in range(0, 5, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 5. Random Sparse Sets
    rng = np.random.default_rng(42)
    # Increase number of random restarts and vary sparsity
    for _ in range(80):
        k = rng.integers(2, 20) # Wider range of sparsity
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        add_candidate(h)
        if elapsed() > deadline - 5:
            return list(best_h)

    # 6. Uniform
    add_candidate(np.ones(N))
    
    # 7. Gaussian
    x = np.arange(N)
    for sigma in [2, 5, 10, 20]:
        h = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        add_candidate(h)

    # 8. Structured Sparse Patterns (e.g., arithmetic progressions)
    for start in range(0, N, 5):
        for step in range(2, 12):
            indices = list(range(start, N, step))
            if len(indices) >= 2:
                h = np.zeros(N)
                h[indices] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 9. Quadratic residue-like patterns
    for base in [2, 3, 5, 7]:
        h = np.zeros(N)
        for i in range(N):
            if (i * i) % base == 0:
                h[i] = 1.0
        if np.sum(h) >= 2:
            add_candidate(h)
        if elapsed() > deadline - 5:
            return list(best_h)

    # 10. Fibonacci-like spacing
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    h = np.zeros(N)
    pos = 0
    for f in fibs:
        if pos < N:
            h[pos] = 1.0
            pos += f
    if np.sum(h) >= 2:
        add_candidate(h)
    if elapsed() > deadline - 5:
        return list(best_h)

    # Sort candidates by score, keep top few for refinement
    candidates.sort(key=lambda x: -x[0])
    
    # --- Phase 2: Refinement ---
    
    # We refine the top candidates.
    # We will use SLSQP on the normalized vector to handle the non-smooth max better.
    
    top_candidates = candidates[:20]
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 2:
            break
            
        # Normalize h_init
        h_norm_init = h_init / np.sum(h_init)
        
        # Constraints for SLSQP: sum(h) = 1, h >= 0
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 1)] * N
        
        try:
            # Use SLSQP with finite difference gradient
            # This is robust to the kink in the max function
            res = minimize(neg_ratio_normalized, h_norm_init, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12})
            
            if elapsed() > deadline - 1:
                break
                
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- Phase 3: Final Polish ---
    # If we have time, do one more high-precision pass on the best solution found.
    # We can try Nelder-Mead on the best solution for a final kick.
    if elapsed() < deadline - 1:
        h_norm_best = best_h / np.sum(best_h)
        try:
            res = minimize(neg_ratio_normalized, h_norm_best, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
