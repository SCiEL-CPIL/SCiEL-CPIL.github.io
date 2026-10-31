import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def compute_score_numpy(U):
        """Faster score computation using numpy."""
        U = np.array(sorted(set(U)), dtype=np.int64)
        if len(U) < 2:
            return 0
        max_u = int(U.max())
        if max_u == 0:
            return 0
        # Sumset
        sums = U[:, None] + U[None, :]
        n_sum = len(np.unique(sums))
        # Diffset
        diffs = U[:, None] - U[None, :]
        n_diff = len(np.unique(diffs))
        if n_sum == 0:
            return -1
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def soft_cardinality(values, sigma=1.0):
        """
        Differentiable surrogate for |set(values)| using Gaussian binning.
        Returns a soft count of distinct values.
        """
        values = np.array(values, dtype=np.float64)
        # Soft cardinality: for each pair, 1 if "different" else 0
        # Use a smooth approximation: exp(-((vi - vj)/sigma)^2)
        # Cardinality ≈ sum_i (1 - (1/N) sum_{j≠i} exp(-((vi-vj)/sigma)^2))
        n = len(values)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0
        diff = values[:, None] - values[None, :]
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        # Soft "same" indicator
        same = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(same, 0.0)
        # Average similarity to others
        avg_sim = same.sum(axis=1) / (n - 1)
        # Soft distinctness
        distinct = 1.0 - avg_sim
        return float(distinct.sum())

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    def is_golomb(u):
        """Check if u is a Golomb ruler (all pairwise differences distinct)."""
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def local_search(base_u, time_limit):
        """Local search: try adding, removing, or swapping elements while maintaining Golomb property."""
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        improved = True
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_candidate = None
            best_candidate_score = best_score_local
            
            # Try adding a new element
            current_max = max(best_u)
            n = len(best_u)
            search_hi = current_max + n * n * 2
            for new_mark in range(current_max + 1, search_hi + 1):
                candidate = sorted(best_u + [new_mark])
                if not is_golomb(candidate):
                    continue
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
                if (time.time() - start_time) > time_limit:
                    break
            
            # Try removing an element (if size > 3)
            if best_u and len(best_u) > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # keep 0
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Try swapping: replace one element with another
            if best_u and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # don't swap 0
                    old_val = best_u[i]
                    # Try replacing with nearby values
                    for delta in range(1, max(10, len(best_u) * 3)):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0:
                                continue
                            if new_val in best_u:
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            if not is_golomb(candidate):
                                continue
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                        if (time.time() - start_time) > time_limit:
                            break
            
            if best_candidate is not None:
                best_u = best_candidate
                best_score_local = best_candidate_score
                improved = True
        
        return best_u, best_score_local

    def continuous_optimize(n_points, time_limit, sigma=1.0, lr=0.1, seed=42):
        """
        Continuous optimization over real-valued positions to maximize
        the soft ratio of difference set to sum set cardinalities.
        """
        rng = np.random.RandomState(seed)
        start_time = time.time()
        
        # Initialize positions: 0 fixed, others random in [0, max_range]
        max_range = 50
        x = np.zeros(n_points)
        x[1:] = rng.uniform(0, max_range, n_points - 1)
        
        best_x = x.copy()
        best_obj = -np.inf
        
        for iteration in range(10000):
            if (time.time() - start_time) > time_limit:
                break
            
            # Compute soft cardinalities
            sums = x[:, None] + x[None, :]
            diffs = x[:, None] - x[None, :]
            
            soft_sum = soft_cardinality(sums, sigma)
            soft_diff = soft_cardinality(diffs, sigma)
            
            # Objective: maximize log(soft_diff / soft_sum) / log(2*max(x)+1)
            # To make it differentiable, use soft max
            max_x = np.max(x)
            base = 2 * max_x + 1
            if soft_sum < 1e-10:
                obj = -1000.0
            else:
                ratio = soft_diff / soft_sum
                if ratio <= 1:
                    obj = 1.0
                else:
                    obj = 1 + math.log(ratio) / math.log(base)
            
            if obj > best_obj:
                best_obj = obj
                best_x = x.copy()
            
            # Compute gradient numerically (simple finite differences)
            grad = np.zeros(n_points)
            eps = 1e-4
            for i in range(1, n_points):  # skip 0
                x_plus = x.copy()
                x_plus[i] += eps
                sums_p = x_plus[:, None] + x_plus[None, :]
                diffs_p = x_plus[:, None] - x_plus[None, :]
                soft_sum_p = soft_cardinality(sums_p, sigma)
                soft_diff_p = soft_cardinality(diffs_p, sigma)
                max_x_p = np.max(x_plus)
                base_p = 2 * max_x_p + 1
                if soft_sum_p < 1e-10:
                    obj_p = -1000.0
                else:
                    ratio_p = soft_diff_p / soft_sum_p
                    obj_p = 1.0 if ratio_p <= 1 else 1 + math.log(ratio_p) / math.log(base_p)
                
                x_minus = x.copy()
                x_minus[i] -= eps
                if x_minus[i] < 0:
                    x_minus[i] = 0
                sums_m = x_minus[:, None] + x_minus[None, :]
                diffs_m = x_minus[:, None] - x_minus[None, :]
                soft_sum_m = soft_cardinality(sums_m, sigma)
                soft_diff_m = soft_cardinality(diffs_m, sigma)
                max_x_m = np.max(x_minus)
                base_m = 2 * max_x_m + 1
                if soft_sum_m < 1e-10:
                    obj_m = -1000.0
                else:
                    ratio_m = soft_diff_m / soft_sum_m
                    obj_m = 1.0 if ratio_m <= 1 else 1 + math.log(ratio_m) / math.log(base_m)
                
                grad[i] = (obj_p - obj_m) / (2 * eps)
            
            # Gradient ascent with step decay
            x[1:] += lr * grad[1:]
            # Keep non-negative
            x[1:] = np.maximum(x[1:], 0)
            # Decay learning rate
            lr *= 0.999
        
        # Round to integers and ensure uniqueness
        x_rounded = np.round(best_x).astype(int)
        # Ensure 0 is present and unique
        x_rounded[0] = 0
        # Remove duplicates
        unique_vals = sorted(set(x_rounded.tolist()))
        if 0 not in unique_vals:
            unique_vals.insert(0, 0)
        
        return unique_vals

    # Known verified Golomb rulers (optimal or near-optimal)
    known_golomb = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
        [0, 1, 4, 10, 12, 17, 20, 22, 24],
        [0, 1, 4, 10, 12, 17, 20, 22, 24, 26],
        [0, 1, 4, 10, 12, 17, 20, 22, 24, 26, 28],
        [0, 1, 4, 10, 12, 17, 20, 22, 24, 26, 28, 30],
    ]
    
    # First, evaluate known Golomb rulers
    for gu in known_golomb:
        try_set(gu)
    
    # Run continuous optimization for various sizes
    cont_sizes = [4, 5, 6, 7, 8]
    for n_pts in cont_sizes:
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time() - 30
        if remaining < 5:
            break
        time_for_cont = min(remaining * 0.3, 20)
        if time_for_cont < 3:
            time_for_cont = max(3, remaining * 0.5)
        
        # Try multiple seeds
        for seed in [42, 123, 999]:
            if time.time() > deadline - 30:
                break
            cand = continuous_optimize(n_pts, time_for_cont / 3, sigma=1.5, lr=0.5, seed=seed)
            if cand and len(cand) >= 3:
                # Polish with local search
                remaining2 = deadline - time.time() - 15
                if remaining2 > 5:
                    polished, _ = local_search(cand, min(remaining2 * 0.3, 10))
                    try_set(polished)
                try_set(cand)
    
    # Run local search from multiple starting points
    start_points = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12, 17, 20, 22, 24, 26],
    ]
    
    for sp in start_points:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        # Allocate time proportionally
        time_for_this = min(remaining * 0.2, 20)
        if time_for_this < 5:
            time_for_this = max(5, remaining * 0.5)
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            try_set(ext_u)
    
    # Exhaustive search for small sets to ensure we don't miss anything
    max_n = 6
    max_m = 30
    
    for n in range(2, max_n + 1):
        if time.time() > deadline - 15:
            break
        # Choose n-1 elements from 1..max_m
        for combo in combinations(range(1, max_m + 1), n - 1):
            if time.time() > deadline - 15:
                break
            cand = [0] + list(combo)
            try_set(cand)
    
    return list(best_set)
