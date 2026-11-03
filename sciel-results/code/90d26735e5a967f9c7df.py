import numpy as np
import time
import sys
from scipy.optimize import minimize
from itertools import combinations

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    all_indices = np.arange(N)
    
    # Helper: optimize amplitudes for a fixed support
    def optimize_amplitudes(support_indices):
        k = len(support_indices)
        if k < 2:
            return None, 0.0
        
        def neg_obj(x):
            # x are log-amplitudes
            a = np.exp(x)
            a_sum = np.sum(a)
            if a_sum == 0:
                return 0.0
            a = a / a_sum
            h = np.zeros(N)
            h[support_indices] = a
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        # Strategy: 
        # 1. Try exponential skew (common optimal for Sidon-like sets)
        # 2. Try local refinement
        
        best_local_h = None
        best_local_s = 0.0
        
        # Parametric exponential skew: h_i = exp(alpha * i)
        # We optimize alpha.
        def neg_obj_alpha(alpha):
            a = np.exp(alpha * np.arange(k))
            a_sum = np.sum(a)
            if a_sum == 0:
                return 0.0
            a = a / a_sum
            h = np.zeros(N)
            h[support_indices] = a
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        # Search for optimal alpha
        best_alpha = 0.0
        best_alpha_s = 0.0
        # Scan a range of alphas
        for alpha in np.linspace(-5, 5, 50):
            s_val = -neg_obj_alpha(alpha)
            if s_val > best_alpha_s:
                best_alpha_s = s_val
                best_alpha = alpha
        
        # Refine alpha
        try:
            res_alpha = minimize(neg_obj_alpha, best_alpha, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-10})
            if -res_alpha.fun > best_alpha_s:
                best_alpha_s = -res_alpha.fun
                best_alpha = res_alpha.x[0]
        except Exception:
            pass
        
        # Construct h from best alpha
        a = np.exp(best_alpha * np.arange(k))
        a = a / np.sum(a)
        h_candidate = np.zeros(N)
        h_candidate[support_indices] = a
        s_candidate = _score_raw(h_candidate)
        
        if s_candidate > best_local_s:
            best_local_s = s_candidate
            best_local_h = h_candidate.copy()
            
        # Multivariate refinement if time permits
        # Only do this if k is small enough
        if k <= 10 and time.time() < deadline - 5:
            x0 = best_alpha * np.arange(k)
            try:
                res = minimize(neg_obj, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12})
                if res.fun < 0:
                    a = np.exp(res.x)
                    a = a / np.sum(a)
                    h_test = np.zeros(N)
                    h_test[support_indices] = a
                    s_test = _score_raw(h_test)
                    if s_test > best_local_s:
                        best_local_s = s_test
                        best_local_h = h_test.copy()
            except Exception:
                pass
        
        return best_local_h, best_local_s

    # Phase 1: Generate Sidon-like supports
    # A Sidon set (B_2 sequence) has distinct pairwise sums (or differences).
    # This minimizes the autoconvolution peak.
    # We will generate sets using a greedy algorithm that maximizes the minimum difference
    # and penalizes repeated differences.
    
    def generate_sidon_support(size, seed_offset=0):
        # Start with a random starting point or 0
        points = []
        # Use a greedy approach: pick point that maximizes min distance to existing points
        # and has fewest difference collisions.
        
        # Initialize with 0
        points.append(0)
        
        for _ in range(size - 1):
            best_candidate = -1
            best_score_candidate = -1
            
            # Calculate current differences
            current_diffs = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    current_diffs.append(abs(points[i] - points[j]))
            
            for c in range(N):
                if c in points:
                    continue
                
                # Calculate new differences
                new_diffs = current_diffs.copy()
                min_dist = float('inf')
                for p in points:
                    d = abs(c - p)
                    new_diffs.append(d)
                    if d < min_dist:
                        min_dist = d
                
                # Score: We want large min_dist and few repeated differences
                # Unique diffs is good.
                unique_diffs = len(set(new_diffs))
                total_diffs = len(new_diffs)
                # Penalty for collisions
                collision_penalty = total_diffs - unique_diffs
                
                # Heuristic score
                score = min_dist * 10 - collision_penalty
                
                if score > best_score_candidate:
                    best_score_candidate = score
                    best_candidate = c
            
            if best_candidate != -1:
                points.append(best_candidate)
            else:
                break
        
        return sorted(points)

    # Phase 2: Search over support sizes and specific Sidon constructions
    support_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    
    for size in support_sizes:
        if time.time() > deadline:
            break
        
        candidates = []
        
        # 1. Deterministic Sidon-like sets
        # We can shift the set to different positions
        base_sidon = generate_sidon_support(size)
        if len(base_sidon) == size:
            # Shift the set
            for shift in range(0, N - size + 1, max(1, (N - size) // 5)):
                shifted = [x + shift for x in base_sidon]
                if max(shifted) < N:
                    candidates.append(tuple(shifted))
        
        # 2. Random Sidon-like attempts (Greedy with random start)
        for _ in range(50):
            if time.time() > deadline:
                break
            # Random start
            start = np.random.randint(0, N)
            points = [start]
            for _ in range(size - 1):
                best_candidate = -1
                best_score_candidate = -1
                current_diffs = []
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        current_diffs.append(abs(points[i] - points[j]))
                
                for c in range(N):
                    if c in points:
                        continue
                    new_diffs = current_diffs.copy()
                    min_dist = float('inf')
                    for p in points:
                        d = abs(c - p)
                        new_diffs.append(d)
                        if d < min_dist:
                            min_dist = d
                    unique_diffs = len(set(new_diffs))
                    total_diffs = len(new_diffs)
                    collision_penalty = total_diffs - unique_diffs
                    score = min_dist * 10 - collision_penalty + np.random.normal(0, 1) # Add noise
                    if score > best_score_candidate:
                        best_score_candidate = score
                        best_candidate = c
                if best_candidate != -1:
                    points.append(best_candidate)
                else:
                    break
            if len(points) == size:
                candidates.append(tuple(sorted(points)))

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        # Optimize for each candidate
        for support in unique_candidates:
            if time.time() > deadline:
                break
            h_candidate, s_candidate = optimize_amplitudes(list(support))
            if h_candidate is not None and s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()

    # Phase 3: Local Refinement of Best Support
    if time.time() < deadline - 10:
        active = np.where(best_h > 1e-6)[0]
        
        # Try swapping one element in the active support
        active_list = list(active)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(active_list)):
                if time.time() > deadline:
                    break
                for j in all_indices:
                    if j in active_list:
                        continue
                    new_support = active_list.copy()
                    new_support[i] = j
                    new_support = sorted(new_support)
                    h_candidate, s_candidate = optimize_amplitudes(new_support)
                    if h_candidate is not None and s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
                        active_list = new_support
                        improved = True
                        break # Restart loop with new best
                if improved:
                    break

    # Phase 4: Final Polish
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        h_start[h_start < 1e-8] = 1e-8
        x0 = np.log(h_start)
        
        def neg_obj_full(x):
            h = np.exp(x)
            h = h / np.sum(h)
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        try:
            res = minimize(neg_obj_full, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-15})
            if res.fun < 0:
                h_candidate = np.exp(res.x)
                h_candidate = h_candidate / np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
