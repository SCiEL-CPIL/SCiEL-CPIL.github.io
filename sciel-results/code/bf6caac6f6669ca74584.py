"""Dual-view construction: define U via difference set structure, minimize |U+U| while maximizing |U-U|/|U+U| ratio."""
import numpy as np
import math
import random

def construct():
    """
    Dual-view approach: 
    - View 1: U defines a difference set D = U-U
    - View 2: Given D, we want to minimize |U+U| 
    - Key insight: Sets with AP-like structure have small sumsets
    - Strategy: Build U greedily to maximize distinct differences while keeping sums compact
    """
    
    def compute_score(U):
        """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def fast_compute_score(U):
        """Faster score computation using numpy for larger sets"""
        if not U or 0 not in U:
            return 0.0
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_u = int(U_arr[-1])
        if max_u == 0:
            return 0.0
        # Compute all sums and differences
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        n_diff = len(np.unique(diffs))
        n_sum = len(np.unique(sums))
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def local_search(U, max_iters=300, max_val=400):
        """Local search with moves: single swap, 2-swap, add, remove"""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Move 1: Single element swap
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                for delta in range(-15, 16):
                    new_val = elem + delta
                    if new_val < 0 or new_val >= max_val:
                        continue
                    if new_val in U_set and new_val != elem:
                        continue
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            
            # Move 2: 2-swap
            if not improved and n >= 3:
                for i in range(1, min(n, 8)):
                    for j in range(i + 1, min(n, 10)):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-8, 9):
                            for d2 in range(-8, 9):
                                new_v1 = e1 + d1
                                new_v2 = e2 + d2
                                if new_v1 < 0 or new_v2 < 0 or new_v1 >= max_val or new_v2 >= max_val:
                                    continue
                                U_set_check = set(best_U)
                                if new_v1 in U_set_check or new_v2 in U_set_check:
                                    continue
                                if new_v1 == new_v2:
                                    continue
                                new_U = list(best_U)
                                new_U[i] = new_v1
                                new_U[j] = new_v2
                                new_U = sorted(set(new_U))
                                if 0 not in new_U or len(new_U) < 2:
                                    continue
                                new_score = compute_score(new_U)
                                if new_score > best_score + 1e-10:
                                    best_score = new_score
                                    best_U = new_U
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Move 3: Add element
            if not improved:
                U_set = set(best_U)
                # Try strategic candidates
                candidates = []
                for v in range(1, min(max_val, 150)):
                    if v not in U_set:
                        candidates.append(v)
                random.shuffle(candidates)
                for candidate in candidates[:40]:
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Move 4: Remove element
            if not improved:
                for i, elem in enumerate(best_U):
                    if elem == 0:
                        continue
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    def build_greedy_dual_view(target_size=12, max_val=350):
        """
        Greedy construction using dual-view:
        - Track difference set D to ensure many distinct differences
        - Track sum set S to minimize collisions
        - At each step, pick element that adds most new differences while minimizing new sums
        """
        U = [0]
        U_set = {0}
        diff_set = {0}
        sum_set = {0}
        
        for step in range(target_size - 1):
            best_candidate = None
            best_metric = float('-inf')
            
            for candidate in range(1, max_val + 1):
                if candidate in U_set:
                    continue
                
                # Count new differences
                new_diffs = 0
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    if d1 not in diff_set:
                        new_diffs += 1
                    if d2 not in diff_set:
                        new_diffs += 1
                
                # Count new sums
                new_sums = 0
                for a in U_set:
                    s = candidate + a
                    if s not in sum_set:
                        new_sums += 1
                
                # Metric: maximize new differences, minimize new sums
                # Also prefer candidates that keep max(U) reasonable
                current_max = max(U_set)
                new_max = max(current_max, candidate)
                
                # Score: new_diffs / (new_sums + 1) with penalty for large max
                metric = new_diffs / (new_sums + 1.0)
                # Penalize if candidate is much larger than current max
                if new_max > current_max * 1.5:
                    metric *= 0.7
                
                if metric > best_metric:
                    best_metric = metric
                    best_candidate = candidate
            
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
                # Update diff and sum sets
                for a in U_set:
                    d1 = best_candidate - a
                    d2 = a - best_candidate
                    s = best_candidate + a
                    diff_set.add(d1)
                    diff_set.add(d2)
                    sum_set.add(s)
        
        return sorted(U)
    
    def build_ap_based(start=0, step=1, length=10, perturb=0):
        """Build set based on arithmetic progression with perturbation"""
        U = [start + i * step for i in range(length)]
        if perturb > 0:
            random.seed(perturb)
            for i in range(1, len(U)):
                U[i] += random.randint(-perturb, perturb)
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        return U
    
    def build_mian_chowla(n=12):
        """Mian-Chowla sequence: greedy Sidon sequence"""
        U = [0, 1]
        U_set = {0, 1}
        diff_set = {0, 1, -1}
        
        for _ in range(n - 2):
            candidate = 2
            while True:
                valid = True
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    if d1 in diff_set or d2 in diff_set:
                        valid = False
                        break
                if valid:
                    break
                candidate += 1
            
            U.append(candidate)
            U_set.add(candidate)
            for a in U_set:
                d1 = candidate - a
                d2 = a - candidate
                diff_set.add(d1)
                diff_set.add(d2)
        
        return sorted(U)
    
    # Generate candidates using multiple strategies
    candidates = []
    
    # Strategy 1: Greedy dual-view construction with different parameters
    for target_size in [10, 11, 12, 13, 14]:
        for max_val in [250, 300, 350, 400]:
            U = build_greedy_dual_view(target_size=target_size, max_val=max_val)
            if len(U) >= 2:
                candidates.append(U)
    
    # Strategy 2: Mian-Chowla sequences (maximize distinct differences)
    for n in [8, 10, 12, 14]:
        U = build_mian_chowla(n=n)
        candidates.append(U)
    
    # Strategy 3: AP-based with perturbation
    for step in [3, 4, 5, 6, 7, 8]:
        for length in [8, 10, 12]:
            for perturb in [0, 1, 2, 3]:
                U = build_ap_based(step=step, length=length, perturb=perturb)
                if len(U) >= 2:
                    candidates.append(U)
    
    # Strategy 4: Hybrid - start with AP, add Sidon-like elements
    for base_step in [4, 5, 6]:
        base_len = 6
        U_base = build_ap_based(step=base_step, length=base_len, perturb=0)
        U_set = set(U_base)
        diff_set = set()
        sum_set = set()
        for a in U_set:
            for b in U_set:
                diff_set.add(a - b)
                sum_set.add(a + b)
        
        # Add Sidon-like elements
        for _ in range(4):
            best_cand = None
            best_score = float('-inf')
            for cand in range(max(U_base) + 1, max(U_base) + 100):
                if cand in U_set:
                    continue
                new_diffs = 0
                for a in U_set:
                    if cand - a not in diff_set:
                        new_diffs += 1
                    if a - cand not in diff_set:
                        new_diffs += 1
                new_sums = 0
                for a in U_set:
                    if cand + a not in sum_set:
                        new_sums += 1
                score = new_diffs / (new_sums + 1.0)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand is not None:
                U_base.append(best_cand)
                U_set.add(best_cand)
                for a in U_set:
                    diff_set.add(best_cand - a)
                    diff_set.add(a - best_cand)
                    sum_set.add(best_cand + a)
        
        candidates.append(sorted(U_base))
    
    # Strategy 5: Random Sidon-biased constructions
    random.seed(123)
    for trial in range(15):
        n = random.randint(8, 14)
        max_val = random.randint(200, 400)
        U = [0]
        U_set = {0}
        diff_set = {0}
        sum_set = {0}
        
        for _ in range(n - 1):
            best_cand = None
            best_metric = float('-inf')
            for _ in range(50):
                cand = random.randint(1, max_val)
                if cand in U_set:
                    continue
                new_diffs = 0
                for a in U_set:
                    if cand - a not in diff_set:
                        new_diffs += 1
                    if a - cand not in diff_set:
                        new_diffs += 1
                new_sums = 0
                for a in U_set:
                    if cand + a not in sum_set:
                        new_sums += 1
                metric = new_diffs / (new_sums + 1.0)
                if metric > best_metric:
                    best_metric = metric
                    best_cand = cand
            if best_cand is not None:
                U.append(best_cand)
                U_set.add(best_cand)
                for a in U_set:
                    diff_set.add(best_cand - a)
                    diff_set.add(a - best_cand)
                    sum_set.add(best_cand + a)
        
        U = sorted(U)
        if len(U) >= 2:
            candidates.append(U)
    
    # Run local search on each candidate and keep the best
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        
        searched_U, searched_score = local_search(U, max_iters=250, max_val=350)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
