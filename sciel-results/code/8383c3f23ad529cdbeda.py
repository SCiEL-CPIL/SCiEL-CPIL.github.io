import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time
import random

N = 50

def compute_ratio(h):
    """Compute ||h||^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -1.0
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_h = None
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
        
    # Strategy 1: Combinatorial Search over Sparse Supports
    # The ratio is scale invariant. We can fix ||h||_1 = 1.
    # If h has support S, h_i > 0 for i in S, 0 otherwise.
    # We optimize the weights w_i (sum to 1) for i in S.
    
    indices = list(range(N))
    
    # We will test supports of size K from 2 to 12
    # Larger K might be necessary if the optimal shape is wider, but small K is likely better for the ratio.
    max_k = 12
    
    # Pre-calculate some random permutations to diversify the subsets we pick if we can't check all
    # Number of subsets C(50, K) grows fast.
    # C(50, 2) = 1225
    # C(50, 3) = 19600
    # C(50, 4) = 230300
    # C(50, 5) = 2118760
    # We can afford to check all for K=2,3. For K=4,5 we need sampling.
    
    def optimize_on_support(S):
        """
        S: list of indices.
        Optimizes weights w (len S) such that sum(w)=1, w>=0.
        Returns best h (full length N) and score.
        """
        K = len(S)
        
        # Parameterization: use log-softmax to ensure positivity and sum 1?
        # Or just optimize w directly with constraints?
        # Let's use a simple gradient-free method on the simplex or just minimize on unconstrained space with penalty?
        # Better: Use minimize with method='Nelder-Mead' on log-weights or just direct weights with bounds.
        # Since K is small, we can use a dense grid or random restarts + local polish.
        
        # Objective function for weights w (sum to 1, w>=0)
        # We use a parameterization v = exp(z) / sum(exp(z)) to handle positivity automatically.
        
        def obj(z):
            v = np.exp(z - np.max(z))
            v = v / np.sum(v)
            # Construct full h
            h = np.zeros(N)
            for i, idx in enumerate(S):
                h[idx] = v[i]
            # Scale doesn't matter for ratio, but let's keep sum 1.
            return -compute_ratio(h)
        
        # Initial guesses
        best_local_score = 0.0
        best_local_v = None
        
        # Try a few random starts in z-space
        n_starts = 5
        for _ in range(n_starts):
            z0 = np.random.randn(K) * 0.5
            try:
                res = minimize(obj, z0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success or res.fun < 0: # res.fun is negative score
                    score = -res.fun
                    if score > best_local_score:
                        best_local_score = score
                        best_local_v = res.x
            except Exception:
                pass
        
        if best_local_v is not None:
            # Convert back to h
            v = np.exp(best_local_v - np.max(best_local_v))
            v = v / np.sum(v)
            h = np.zeros(N)
            for i, idx in enumerate(S):
                h[idx] = v[i]
            # Verify score
            final_score = compute_ratio(h)
            if final_score > best_local_score:
                best_local_score = final_score
            return h, best_local_score
        return None, 0.0

    # Iterate over K
    for K in range(2, max_k + 1):
        if time_left() < 5.0:
            break
            
        # Determine how many subsets to check
        # If total subsets < 500, check all.
        # Else, sample 200 random subsets.
        total_subsets = int(np.prod(np.array([N - K + i for i in range(K)]) / np.prod(np.arange(1, K + 1))))
        # Approximation for large numbers, but for K<=5 it's fine.
        
        if total_subsets <= 1000:
            subsets = list(combinations(indices, K))
        else:
            # Random sampling
            subsets = []
            seen = set()
            target_samples = 300
            while len(subsets) < target_samples and time_left() > 5.0:
                s = tuple(sorted(random.sample(indices, K)))
                if s not in seen:
                    seen.add(s)
                    subsets.append(s)
        
        for S in subsets:
            if time_left() < 2.0:
                break
                
            h, score = optimize_on_support(S)
            if h is not None and score > best_score:
                best_score = score
                best_h = h.copy()
                
        # If we found a good score with small K, we might skip larger K if time is tight?
        # No, larger K might be better. But if K=2 gives 1.24, K=3 might give 1.25.
        
    # Strategy 2: Refinement of the best sparse solution using full continuous optimization
    # Sometimes the optimal solution is not strictly sparse but "almost" sparse.
    # We can use the best_h as a starting point for a full N-dim optimizer.
    
    if best_h is not None and time_left() > 10.0:
        # Use L-BFGS-B on the full vector, but constrained to be non-negative.
        # We can use the log-softmax parameterization again, or just bounds.
        
        def full_obj(z):
            # z is unconstrained, v = softmax(z)
            v = np.exp(z - np.max(z))
            v = v / np.sum(v)
            return -compute_ratio(v)
        
        # Initialize z from best_h
        # If best_h has zeros, log(0) is -inf. We need to handle this.
        # We can replace zeros with a small epsilon before log.
        h_init = best_h.copy()
        h_init[h_init == 0] = 1e-8
        z_init = np.log(h_init)
        
        try:
            res = minimize(full_obj, z_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 0:
                v_opt = np.exp(res.x - np.max(res.x))
                v_opt = v_opt / np.sum(v_opt)
                score_opt = compute_ratio(v_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = v_opt
        except Exception:
            pass

    # Strategy 3: Random restarts with global optimization (Differential Evolution) on full space
    # This is expensive, so only if time permits and we haven't found a very high score.
    if time_left() > 30.0 and best_score < 1.3:
        def de_obj(z):
            v = np.exp(z - np.max(z))
            v = v / np.sum(v)
            return -compute_ratio(v)
            
        try:
            # Bounds for z: [-10, 10] is usually enough to cover dynamic range of weights
            bounds = [(-10, 10)] * N
            res = differential_evolution(de_obj, bounds, maxiter=20, popsize=15, tol=1e-8, polish=True, seed=42)
            if res.fun < 0:
                v_opt = np.exp(res.x - np.max(res.x))
                v_opt = v_opt / np.sum(v_opt)
                score_opt = compute_ratio(v_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = v_opt
        except Exception:
            pass

    # Final Output
    if best_h is not None:
        # Scale to avoid underflow/overflow issues in evaluator if any, though ratio is scale invariant.
        # The task asks for "nonnegative heights".
        # Let's return the normalized shape (sum 1) or scaled.
        # The seed program returned 1e-6 * v. Let's do the same for safety.
        h_out = best_h * 1e-6
        h_out = np.maximum(h_out, 0.0)
        return h_out.tolist()
    else:
        # Fallback
        h_out = np.ones(N) * 1e-6
        return h_out.tolist()
