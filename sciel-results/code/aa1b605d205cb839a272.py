"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with Sidon-like structures (maximize distinct differences)
    - Use local search to minimize |U+U| while preserving |U-U|
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
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
    
    def compute_score_fast(U):
        """Faster score computation using numpy for larger sets."""
        if not U or 0 not in U:
            return 0.0
        U_arr = np.array(sorted(U), dtype=np.int64)
        max_u = U_arr[-1]
        if max_u == 0:
            return 0.0
        
        # Compute all sums and differences
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return 0.0
        
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def local_search(U, max_iters=300, max_val=400):
        """Enhanced local search with multiple move types."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score_fast(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try single-element moves (shift)
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                for delta in range(-15, 16):
                    new_val = elem + delta
                    if new_val < 0 or new_val in U_set or new_val > max_val:
                        continue
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    new_score = compute_score_fast(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            
            # Try 2-swap moves
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-8, 9):
                            for d2 in range(-8, 9):
                                new_v1 = e1 + d1
                                new_v2 = e2 + d2
                                if new_v1 < 0 or new_v2 < 0 or new_v1 > max_val or new_v2 > max_val:
                                    continue
                                U_set_check = set(best_U)
                                if new_v1 in U_set_check or new_v2 in U_set_check or new_v1 == new_v2:
                                    continue
                                new_U = list(best_U)
                                new_U[i] = new_v1
                                new_U[j] = new_v2
                                new_U = sorted(set(new_U))
                                if 0 not in new_U or len(new_U) < 2:
                                    continue
                                new_score = compute_score_fast(new_U)
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
            
            # Try adding a new element
            if not improved:
                U_set = set(best_U)
                candidates = list(range(1, min(max_val + 1, 150)))
                random.shuffle(candidates)
                for candidate in candidates[:30]:
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score_fast(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Try removing an element
            if not improved:
                for i, elem in enumerate(best_U):
                    if elem == 0:
                        continue
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score_fast(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    def construct_from_difference_structure():
        """
        Dual-view: construct U from a difference structure.
        Start with a set of target differences D, then find U such that
        U-U covers D efficiently while minimizing U+U.
        """
        candidates = []
        
        # Strategy 1: Start with arithmetic progression base, add structured perturbations
        for base_step in [3, 4, 5, 6, 7, 8]:
            for n in range(5, 12):
                U = [i * base_step for i in range(n)]
                # Add structured perturbations to break symmetry
                for k in range(1, len(U)):
                    U[k] += (k % 3) * 2 + (k % 2)
                U = sorted(set(U))
                if 0 in U:
                    candidates.append(U)
        
        # Strategy 2: Geometric-like growth with controlled differences
        for ratio in [1.5, 1.7, 2.0, 2.3]:
            U = [0]
            current = 1
            for _ in range(10):
                U.append(int(current))
                current *= ratio
            U = sorted(set(U))
            candidates.append(U)
        
        # Strategy 3: Mixed linear + quadratic
        for a in [1, 2, 3]:
            for b in [1, 2, 3, 4]:
                U = [a * i + b * i * i for i in range(12)]
                U = sorted(set(U))
                if 0 in U:
                    candidates.append(U)
        
        # Strategy 4: Random with Sidon bias (minimize difference collisions)
        random.seed(123)
        for trial in range(15):
            n = random.randint(8, 14)
            max_val = random.randint(150, 350)
            U = [0]
            U_set = {0}
            diff_set = set()
            
            for _ in range(n - 1):
                best_candidate = None
                best_score = float('-inf')
                
                for _ in range(80):
                    candidate = random.randint(1, max_val)
                    if candidate in U_set:
                        continue
                    
                    # Count new differences and their collisions
                    new_diffs = set()
                    collisions = 0
                    for a in U_set:
                        d1 = candidate - a
                        d2 = a - candidate
                        new_diffs.add(d1)
                        new_diffs.add(d2)
                        if d1 in diff_set:
                            collisions += 1
                        if d2 in diff_set:
                            collisions += 1
                    
                    # Score: prefer fewer collisions, moderate values
                    score = -collisions * 5 + candidate * 0.05
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    U.append(best_candidate)
                    U_set.add(best_candidate)
                    for a in U_set:
                        diff_set.add(best_candidate - a)
                        diff_set.add(a - best_candidate)
            
            U = sorted(U)
            candidates.append(U)
        
        # Strategy 5: Greedy construction maximizing ratio
        random.seed(456)
        for trial in range(8):
            max_val = random.randint(200, 350)
            U = [0]
            U_set = {0}
            
            for _ in range(10):
                best_candidate = None
                best_ratio = float('-inf')
                
                for candidate in range(1, max_val + 1, 2):
                    if candidate in U_set:
                        continue
                    new_U = sorted(U + [candidate])
                    score = compute_score_fast(new_U)
                    if score > best_ratio:
                        best_ratio = score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    U.append(best_candidate)
                    U_set.add(best_candidate)
            
            U = sorted(U)
            candidates.append(U)
        
        return candidates
    
    # Generate diverse initial candidates
    candidates = construct_from_difference_structure()
    
    # Run local search on each candidate
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
