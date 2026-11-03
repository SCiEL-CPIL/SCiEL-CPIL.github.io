import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """Compute the score for a given height vector h."""
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

def _is_sidon_like(indices, max_collisions=1):
    """
    Check if the set of indices has limited 'collisions' in the difference set.
    A Sidon set has all differences distinct (except trivial +/-).
    For our purpose, we want to minimize the number of pairs (a,b) and (c,d) 
    such that a-b = c-d, which corresponds to the peak of the autoconvolution 
    when weights are uniform.
    """
    diffs = {}
    n = len(indices)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = indices[i] - indices[j]
            if d in diffs:
                diffs[d] += 1
                if diffs[d] > max_collisions * 2: # Allow some slack
                    return False
            else:
                diffs[d] = 1
    return True

def _generate_sidon_candidates(k, n_total=N):
    """
    Generate candidate support sets of size k that are 'Sidon-like'.
    We use a greedy randomized construction to find sparse sets with low collision counts.
    """
    candidates = []
    # Try multiple random restarts to find good Sidon-like sets
    for _ in range(50):
        # Start with a random seed
        available = list(range(n_total))
        np.random.shuffle(available)
        current_set = [available[0]]
        
        for _ in range(k - 1):
            best_candidate = None
            best_collisions = float('inf')
            
            # Try adding each remaining element
            # To speed up, we can sample a subset of candidates if k is large
            # But for k <= 10, checking all is feasible
            for idx in available:
                if idx in current_set:
                    continue
                test_set = current_set + [idx]
                # Quick collision check
                diffs = {}
                for i in range(len(test_set)):
                    for j in range(len(test_set)):
                        if i == j: continue
                        d = test_set[i] - test_set[j]
                        if d in diffs:
                            diffs[d] += 1
                        else:
                            diffs[d] = 1
                max_coll = max(diffs.values()) if diffs else 0
                if max_coll < best_collisions:
                    best_collisions = max_coll
                    best_candidate = idx
            
            if best_candidate is not None:
                current_set.append(best_candidate)
                available.remove(best_candidate)
            
            # If we can't add any without too many collisions, break
            if best_collisions > 2: # Heuristic threshold
                break
        
        if len(current_set) == k:
            candidates.append(current_set)
    
    return candidates

def _optimize_weights_on_support(support_indices, deadline):
    """
    Given a fixed support set, optimize the weights on these indices.
    We restrict the optimization to only these variables.
    """
    k = len(support_indices)
    if k == 0:
        return None, 0.0
    
    # Map full vector to support vector
    def support_to_full(w):
        h = np.zeros(N)
        h[support_indices] = w
        return h
    
    def objective(w):
        h = support_to_full(w)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0.0
        return -(l2_sq / (l1 * max_conv))
    
    # Initial guess: uniform weights on support
    w0 = np.ones(k) / k
    
    # Bounds: weights must be non-negative
    bounds = [(0, 1)] * k
    
    # Constraint: sum of weights = 1 (normalization doesn't change ratio, but helps scaling)
    # Actually, the ratio is scale-invariant. 
    # R(c*h) = (c^2 ||h||^2) / (c ||h||_1 * c ||h*h||_inf) = R(h).
    # So we can fix sum(w) = 1 to simplify.
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    try:
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        if res.success:
            w_opt = res.x
            h_opt = support_to_full(w_opt)
            score = _score_raw(h_opt)
            return h_opt, score
    except Exception:
        pass
    
    # Fallback: uniform weights
    h_uniform = support_to_full(np.ones(k) / k)
    score = _score_raw(h_uniform)
    return h_uniform, score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Strategy: Search over different support sizes (sparsity levels)
    # Sparse supports (small k) tend to have lower autoconvolution peaks relative to L2^2
    # because the "collision" terms are fewer.
    
    # We'll try k from 5 to 15. 
    # k=1 is trivial (score 1). k=2 is better. Larger k needs good Sidon structure.
    
    for k in [5, 6, 7, 8, 9, 10, 12, 15]:
        if time.time() > deadline - 30:
            break
            
        # Generate candidate Sidon-like supports of size k
        # Limit the number of candidates to keep time under control
        candidates = _generate_sidon_candidates(k)
        
        # If we didn't find enough, just use random subsets
        if len(candidates) < 10:
            for _ in range(20):
                subset = np.random.choice(N, size=k, replace=False)
                candidates.append(list(subset))
        
        # Evaluate top candidates (based on a quick proxy metric or just iterate)
        # We'll iterate through them, but limit the number of optimizations per k
        max_evals_per_k = 10
        for i, support in enumerate(candidates[:max_evals_per_k]):
            if time.time() > deadline - 10:
                break
                
            h_opt, score = _optimize_weights_on_support(support, deadline)
            if h_opt is not None and score > best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Final polish: If we have time, do a local optimization on the full 50-dim space
    # starting from the best found sparse solution (filling zeros with small values or just optimizing the non-zeros)
    if time.time() < deadline - 5:
        # Check if the best solution is already sparse. If so, we've essentially done this.
        # If it's dense, try a quick SLSQP on the full vector from the best point.
        h_start = best_h.copy()
        l1 = np.sum(h_start)
        if l1 > 0:
            h_start /= l1
            
        def full_objective(h):
            l1 = np.sum(h)
            if l1 == 0: return 0.0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0.0
            return -(l2_sq / (l1 * max_conv))
        
        bounds = [(0, 1)] * N
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
        
        try:
            res = minimize(full_objective, h_start, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success:
                h_opt = res.x
                score = _score_raw(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-END
