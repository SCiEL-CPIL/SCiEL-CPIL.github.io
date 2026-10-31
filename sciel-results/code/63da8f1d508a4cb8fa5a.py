import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0

def is_sidon(A):
    """Check if A is a Sidon set (all sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_set_greedy(n):
    """Greedy construction of a Sidon set."""
    A = []
    used_sums = set()
    for x in range(n):
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

def find_max_sidon_set(n, max_time=5.0):
    """Find a large Sidon set using greedy and random restarts."""
    start = time.time()
    best_A = find_sidon_set_greedy(n)
    best_size = len(best_A)
    
    rng = np.random.default_rng(42)
    
    # Try to improve with random restarts
    while time.time() - start < max_time:
        # Try random sets of size best_size + 1
        target_size = best_size + 1
        found = False
        # Limit tries per size to avoid long loops
        for _ in range(1000):
            if time.time() - start > max_time:
                break
            A_try = sorted(rng.choice(n, target_size, replace=False))
            if is_sidon(A_try):
                best_A = A_try
                best_size = target_size
                found = True
                break
        if not found:
            # Maybe try a different greedy strategy or stop
            # If we can't find size K+1, we might still find K+1 with different random seed?
            # But if 1000 random tries fail, it's unlikely.
            # Let's try one more time with a slightly different heuristic or break.
            # For now, break if we fail to increase size.
            break
            
    return best_A

def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return l2_sq / (l1 * hh_max)
    
    def neg_ratio(params, beta=1000.0):
        """Negative ratio with smooth max approximation for better gradient."""
        h = np.exp(params)
        h = h / np.sum(h)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        # Use log-sum-exp for smooth max
        hh_shifted = hh - np.max(hh)
        hh_max_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * hh_shifted))) + np.max(hh)
        if hh_max_smooth < 1e-300:
            return 0.0
        return -l2_sq / (l1 * hh_max_smooth)

    # 1. Find a large Sidon set
    A_sidon = find_max_sidon_set(N, max_time=5.0)
    
    # 2. Create initial candidate based on Sidon set
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    h_sidon = np.zeros(N)
    if len(A_sidon) > 0:
        h_sidon[A_sidon] = 1.0 / len(A_sidon)
        val = compute_ratio(h_sidon)
        if val > best_val:
            best_val = val
            best_h = h_sidon.copy()
    
    # 3. Also try some other structured candidates
    # Single spike
    h1 = np.zeros(N); h1[N//2] = 1.0
    val = compute_ratio(h1)
    if val > best_val:
        best_val = val
        best_h = h1.copy()
        
    # Two spikes
    h2 = np.zeros(N); h2[0] = 0.5; h2[N-1] = 0.5
    val = compute_ratio(h2)
    if val > best_val:
        best_val = val
        best_h = h2.copy()

    # 4. Refine the best candidate with L-BFGS-B
    # We will refine the Sidon-based candidate and the current best
    candidates_to_refine = []
    if len(A_sidon) > 0:
        candidates_to_refine.append(h_sidon)
    candidates_to_refine.append(best_h)
    
    # Deduplicate candidates roughly
    unique_candidates = []
    for h in candidates_to_refine:
        if not any(np.allclose(h, u) for u in unique_candidates):
            unique_candidates.append(h)
            
    for h_init in unique_candidates:
        if elapsed() > DEADLINE - 20:
            break
        
        h_pos = h_init + 1e-10
        x0 = np.log(h_pos)
        
        try:
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 2000})
            if elapsed() > DEADLINE - 10:
                break
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # 5. Final polish
    if elapsed() < DEADLINE - 5:
        try:
            h_pos = best_h + 1e-10
            x0 = np.log(h_pos)
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxfun': 5000})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
