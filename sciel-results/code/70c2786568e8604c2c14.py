import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin

# EVOLVE-BLOCK-START
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

    def get_support_metrics(S):
        """Calculate max multiplicity of sums for a support S."""
        sums = {}
        for i in S:
            for j in S:
                s = i + j
                sums[s] = sums.get(s, 0) + 1
        max_r = max(sums.values()) if sums else 0
        return max_r

    # 1. Generate candidate supports
    candidates = []
    
    # Greedy Sidon-like set
    S_greedy = [0]
    for i in range(1, N):
        S_test = S_greedy + [i]
        # Check if adding i creates too many collisions
        # Simple heuristic: if max_r increases by more than 1, skip?
        # Or just add and see.
        # Let's just add if it fits in N.
        S_greedy.append(i)
        if len(S_greedy) > 12: break # Cap size for now, optimize later
    
    # Actually, a proper greedy Sidon set construction:
    S_sidon = [0]
    for i in range(1, N):
        # Check if i can be added while keeping max_r low
        # For simplicity, let's just generate random sets and one greedy set
        pass

    # Random sparse sets
    rng = np.random.default_rng(42)
    for _ in range(200):
        K = rng.integers(3, 15)
        S = rng.choice(N, size=K, replace=False).tolist()
        candidates.append(S)
    
    # Specific structured sets
    candidates.append(list(range(10))) # Consecutive (bad for Sidon, but baseline)
    candidates.append([0, 1, 3, 7, 12, 18, 25, 33, 42]) # Quadratic growth (Sidon-like)
    candidates.append([0, 1, 4, 10, 19, 31, 46]) # Another Sidon-like
    
    # Evaluate and sort candidates by potential
    scored_candidates = []
    for S in candidates:
        if len(S) == 0: continue
        max_r = get_support_metrics(S)
        potential = len(S) / max_r
        scored_candidates.append((potential, S))
        
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_h = np.zeros(N)
    best_val = 0.0
    
    # 2. Optimize weights for top candidates
    n_to_try = min(20, len(scored_candidates))
    
    for i in range(n_to_try):
        if elapsed() > DEADLINE - 10:
            break
            
        S = scored_candidates[i][1]
        K = len(S)
        
        # Initial guess: uniform
        h0 = np.ones(K) / K
        
        # Define objective for optimization
        def neg_ratio_on_support(x):
            # x are the weights for indices in S
            h_full = np.zeros(N)
            for idx, val in zip(S, x):
                h_full[idx] = val
            return -compute_ratio(h_full)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(K)]
        
        try:
            res = minimize(neg_ratio_on_support, h0, method='SLSQP', 
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            if elapsed() > DEADLINE - 5:
                break
                
            h_opt_weights = res.x
            h_full = np.zeros(N)
            for idx, val in zip(S, h_opt_weights):
                h_full[idx] = val
            
            val = compute_ratio(h_full)
            if val > best_val:
                best_val = val
                best_h = h_full.copy()
                
        except Exception:
            # Fallback to uniform weights if optimization fails
            h_full = np.zeros(N)
            for idx in S:
                h_full[idx] = 1.0 / K
            val = compute_ratio(h_full)
            if val > best_val:
                best_val = val
                best_h = h_full.copy()

    # 3. Final check with a few random full-vector optimizations if time permits
    if elapsed() < DEADLINE - 20:
        # Try a few random initializations in the full space
        for _ in range(5):
            if elapsed() > DEADLINE - 10:
                break
            h_rand = np.random.dirichlet(np.ones(N) * 0.5)
            
            def neg_ratio_full(x):
                # Ensure non-negative and sum 1
                # Using softmax or just clip?
                # SLSQP with bounds is better
                return -compute_ratio(x)
            
            # We can't easily use SLSQP on full N=50 with 50 constraints?
            # It might be slow.
            # Instead, project to simplex?
            # Let's just use the best_h found so far as a starting point for a local search?
            # Or skip this if it's too complex.
            pass

    return list(best_h)
# EVOLVE-BLOCK-END
