import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Helper: Given a support set S, find optimal weights w on S
    # Maximize sum(w^2) / (sum(w) * max_conv(w)) subject to w >= 0
    # Since the ratio is scale invariant, we can fix sum(w) = 1.
    # Then we maximize sum(w^2) / max_conv(w).
    # This is still non-convex due to max_conv in denominator.
    # However, for small supports (|S| <= 10), we can use a grid search or local optimization on the weights.
    # Or, we can observe that for a fixed support, the max_conv is determined by the peak of the convolution.
    # If the support is a Sidon set, all pairwise differences are distinct, so the convolution peaks at 0 (self-convolution) 
    # and potentially at other distinct lags. 
    # Actually, if differences are distinct, the convolution at lag 0 is sum(w^2). 
    # The convolution at lag k is sum_{i-j=k} w_i w_j.
    # For a Sidon set, each pair (i,j) with i!=j has a unique difference. 
    # So the convolution values at non-zero lags are just w_i w_j for specific pairs.
    # The peak will be max(sum(w^2), max_{i!=j} w_i w_j).
    # Since w_i, w_j >= 0, w_i w_j <= (w_i^2 + w_j^2)/2.
    # So sum(w^2) is likely the peak if weights are concentrated.
    
    def optimize_weights_on_support(support):
        m = len(support)
        if m == 0:
            return np.zeros(N), 0.0
        if m == 1:
            h = np.zeros(N)
            h[support[0]] = 1.0
            return h, _score_raw(h)
        
        # We want to maximize R(w) = ||w||_2^2 / (||w||_1 * max_conv(w)) with sum(w)=1.
        # R(w) = ||w||_2^2 / max_conv(w).
        
        # Initialize weights uniformly
        w = np.ones(m) / m
        
        # Use local optimization on the weights
        # Variable: w (length m)
        # Constraint: sum(w) = 1, w >= 0
        
        def neg_obj(w):
            if np.any(w < -1e-9):
                return 0.0
            w = np.maximum(w, 0)
            if np.sum(w) == 0:
                return 0.0
            w_norm = w / np.sum(w)
            h_full = np.zeros(N)
            h_full[support] = w_norm
            # Calculate max conv
            # For small m, np.convolve is fast
            conv = np.convolve(h_full, h_full, mode='full')
            max_conv = np.max(conv)
            if max_conv == 0:
                return 0.0
            l2_sq = np.sum(w_norm**2)
            return -(l2_sq / max_conv)
            
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0)] * m
        
        best_w = w
        best_val = -neg_obj(w)
        
        # Try a few restarts for weight optimization
        for _ in range(3):
            w_init = np.random.dirichlet(np.ones(m))
            try:
                res = minimize(neg_obj, w_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
                if res.success:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_w = res.x
            except Exception:
                pass
        
        h_full = np.zeros(N)
        h_full[support] = best_w
        # Normalize to ensure sum=1 for score calculation consistency, though _score_raw handles it
        score = _score_raw(h_full)
        return h_full, score

    # Strategy: Find good supports.
    # 1. Sidon-like sets.
    # A Sidon set is a set of integers such that all pairwise sums are distinct.
    # For convolution, we care about differences. A set with distinct differences is a B_2 sequence or Sidon set.
    # If differences are distinct, the convolution terms don't "collide" (except for symmetric pairs i-j and j-i which are same difference magnitude but opposite sign, but for non-negative weights, w_i w_j = w_j w_i).
    # Wait, convolution at lag k is sum_{i-j=k} w_i w_j.
    # If all differences i-j are distinct for i!=j, then each non-zero lag has at most one term (or two if we consider i-j=k and j-i=-k, but we look at max over all lags).
    # Actually, if the set is symmetric or not, the structure matters.
    # Let's just use a greedy heuristic to find sets with "spread out" differences.
    
    # Greedy Sidon Set Construction
    def greedy_sidon_set(size):
        S = [0]
        for _ in range(size - 1):
            candidates = []
            for x in range(1, N):
                if x in S:
                    continue
                # Check if adding x creates a difference that already exists
                # Existing differences:
                diffs = set()
                for i in range(len(S)):
                    for j in range(len(S)):
                        if i != j:
                            diffs.add(S[i] - S[j])
                # New differences with x:
                new_diffs = []
                valid = True
                for s in S:
                    d1 = x - s
                    d2 = s - x
                    if d1 in diffs or d2 in diffs:
                        valid = False
                        break
                    new_diffs.append(d1)
                    new_diffs.append(d2)
                if valid:
                    # Score candidate by "sparsity" or just pick first
                    candidates.append(x)
            if not candidates:
                break
            # Pick the candidate that is furthest from existing points to maximize spread?
            # Or just pick the smallest? Smallest tends to be denser.
            # Let's pick the one that maximizes the minimum distance to existing points?
            best_cand = None
            best_min_dist = -1
            for c in candidates:
                min_d = min(abs(c - s) for s in S)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_cand = c
            if best_cand is not None:
                S.append(best_cand)
            else:
                break
        return sorted(S)
    
    # 2. Random sparse sets
    # 3. Structured sets (e.g., powers of 2, primes)
    
    supports_to_try = []
    
    # Add single spike
    supports_to_try.append([N//2])
    
    # Add pairs
    for i in range(N):
        for j in range(i+1, N):
            supports_to_try.append([i, j])
            
    # Add Sidon-like sets of size 3, 4, 5
    for size in range(3, 6):
        for _ in range(20):
            # Randomized greedy
            S = [np.random.randint(0, N)]
            for _ in range(size - 1):
                candidates = [x for x in range(N) if x not in S]
                if not candidates: break
                # Pick random candidate that doesn't create immediate collision? 
                # Just pick random for now, optimization will handle weights
                x = np.random.choice(candidates)
                S.append(x)
            supports_to_try.append(sorted(set(S)))
            
            # Also try greedy from random start
            S_g = [np.random.randint(0, N)]
            for _ in range(size - 1):
                candidates = []
                diffs = set()
                for i in range(len(S_g)):
                    for j in range(len(S_g)):
                        if i != j:
                            diffs.add(S_g[i] - S_g[j])
                for x in range(N):
                    if x in S_g: continue
                    valid = True
                    for s in S_g:
                        if (x - s) in diffs or (s - x) in diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                if not candidates: break
                x = np.random.choice(candidates)
                S_g.append(x)
            supports_to_try.append(sorted(set(S_g)))
            
    # Add structured sets
    # Powers of 2
    p2 = [1, 2, 4, 8, 16, 32]
    p2 = [x for x in p2 if x < N]
    supports_to_try.append(p2)
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    primes = [x for x in primes if x < N]
    supports_to_try.append(primes)
    
    # Random sparse sets
    for _ in range(50):
        k = np.random.randint(3, 8)
        idx = np.random.choice(N, k, replace=False)
        supports_to_try.append(sorted(idx))
        
    # Sort supports by size (smaller first, usually better ratios?)
    # Actually, larger supports can have better ratios if well-chosen.
    # Let's just iterate.
    
    # To save time, we will limit the number of supports we fully optimize.
    # We'll evaluate a quick heuristic score for each support first.
    
    def quick_heuristic_score(support):
        # Uniform weights
        m = len(support)
        h = np.zeros(N)
        h[support] = 1.0 / m
        return _score_raw(h)
        
    # Filter supports
    scored_supports = []
    for s in supports_to_try:
        if time.time() > deadline - 5:
            break
        s = sorted(set(s))
        if len(s) == 0: continue
        sc = quick_heuristic_score(s)
        scored_supports.append((sc, s))
        
    # Sort by heuristic score descending
    scored_supports.sort(key=lambda x: x[0], reverse=True)
    
    # Optimize weights for the top supports
    # Limit to top 50 supports to ensure time safety
    top_supports = scored_supports[:50]
    
    for sc, s in top_supports:
        if time.time() > deadline - 2:
            break
        try:
            h_opt, sc_opt = optimize_weights_on_support(s)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt
        except Exception:
            pass
            
    # Final polish on the best support found
    if time.time() < deadline - 5:
        # Extract support from best_h
        support = np.where(best_h > 1e-6)[0].tolist()
        if support:
            h_opt, sc_opt = optimize_weights_on_support(support)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt

    return list(best_h)
