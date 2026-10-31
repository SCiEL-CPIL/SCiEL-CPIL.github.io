"""Dual-view construction: define U via its difference set D, minimize |U+U| while keeping |U-U| large."""
import numpy as np
import math
import random

def construct():
    """
    Dual-view approach:
    - Generate candidate difference sets D (Sidon-like)
    - For each D, find U such that U-U ⊆ D and |U-U| is maximized
    - Optimize U to minimize |U+U| given the constraint on U-U
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
        """Compute the score for a set U."""
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
        U_arr = np.array(sorted(set(U)))
        if len(U_arr) < 2 or U_arr[-1] == 0:
            return 0.0
        max_u = U_arr[-1]
        
        # Compute all sums and differences
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(set(sums.flatten().tolist()))
        n_diff = len(set(diffs.flatten().tolist()))
        
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def local_search(U, max_iters=300, max_val=400):
        """Local search with focus on minimizing |U+U| while preserving |U-U|."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try single element moves with wider range
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                # Try a range of deltas, biased toward values that might reduce sumset
                for delta in list(range(-15, 16)):
                    new_val = elem + delta
                    if new_val < 0 or new_val in U_set or new_val > max_val:
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
            
            # Try 2-element swaps
            if not improved and n >= 3:
                found = False
                for i in range(1, min(n, 6)):
                    for j in range(i + 1, min(n, 6)):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-8, 9):
                            for d2 in range(-8, 9):
                                if found:
                                    break
                                new_v1 = e1 + d1
                                new_v2 = e2 + d2
                                if new_v1 < 0 or new_v2 < 0:
                                    continue
                                if new_v1 > max_val or new_v2 > max_val:
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
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    continue
            
            # Try adding elements that maximize difference set growth while minimizing sumset growth
            if not improved:
                U_set = set(best_U)
                # Sample candidate positions
                candidates = list(range(1, min(max_val + 1, 150)))
                random.shuffle(candidates)
                for candidate in candidates[:30]:
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Try removing elements
            if not improved:
                for i in range(len(best_U) - 1, 0, -1):  # Remove from end first
                    elem = best_U[i]
                    if elem == 0:
                        continue
                    new_U = best_U[:i] + best_U[i+1:]
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
    
    # Generate initial candidates using dual-view strategy
    candidates = []
    
    # Strategy 1: Start with AP-like structures (small sumsets) and perturb to increase diffsets
    random.seed(123)
    for base in range(3, 15):
        for n in range(4, 12):
            # Base AP
            U_base = list(range(0, n * base, base))
            if 0 not in U_base:
                U_base = [0] + U_base
            
            # Perturb to increase differences while keeping sums small
            U = list(U_base)
            for i in range(1, len(U)):
                # Small random perturbations
                if random.random() < 0.7:
                    U[i] += random.randint(-2, 2)
                    U[i] = max(0, U[i])
            
            U = sorted(set(U))
            if 0 in U and len(U) >= 3:
                candidates.append(U)
    
    # Strategy 2: Sidon sequences (large diffsets) with compression
    # Mian-Chowla style
    def mian_chowla(n, start=0):
        """Generate Mian-Chowla Sidon sequence."""
        seq = [start]
        for _ in range(n - 1):
            candidate = seq[-1] + 1
            while True:
                # Check if candidate creates duplicate differences
                valid = True
                for i in range(len(seq)):
                    d1 = candidate - seq[i]
                    d2 = seq[i] - candidate
                    for j in range(i + 1, len(seq)):
                        if d1 == seq[j] - seq[i] or d1 == seq[i] - seq[j]:
                            valid = False
                            break
                        if d2 == seq[j] - seq[i] or d2 == seq[i] - seq[j]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                candidate += 1
            seq.append(candidate)
        return seq
    
    for n in range(5, 14):
        seq = mian_chowla(n)
        # Compress by scaling down
        for scale in [1, 2, 3]:
            U = [x * scale for x in seq]
            candidates.append(U)
    
    # Strategy 3: Hybrid - combine AP structure with Sidon properties
    for trial in range(20):
        n = random.randint(5, 12)
        # Start with a scaled AP
        base = random.randint(4, 10)
        U = list(range(0, n * base, base))
        
        # Apply Sidon-like perturbation: ensure differences are mostly unique
        random.shuffle(U[1:])
        for i in range(1, len(U)):
            # Nudge to avoid difference collisions
            for _ in range(5):
                # Check for difference collisions
                diff_counts = {}
                for a in U:
                    for b in U:
                        d = a - b
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                max_count = max(diff_counts.values())
                if max_count <= 2:
                    break
                # Find element with most collisions and nudge it
                elem_diffs = {}
                for idx, elem in enumerate(U):
                    if idx == 0:
                        continue
                    count = 0
                    for other in U:
                        d1 = elem - other
                        d2 = other - elem
                        if d1 in diff_counts and diff_counts[d1] > 1:
                            count += 1
                        if d2 in diff_counts and diff_counts[d2] > 1:
                            count += 1
                    elem_diffs[idx] = count
                
                if elem_diffs:
                    worst_idx = max(elem_diffs, key=elem_diffs.get)
                    U[worst_idx] += random.randint(-3, 3)
                    U[worst_idx] = max(0, U[worst_idx])
        
        U = sorted(set(U))
        if 0 in U and len(U) >= 3:
            candidates.append(U)
    
    # Strategy 4: Greedy construction optimizing the actual score
    random.seed(456)
    for trial in range(15):
        max_val = random.randint(150, 350)
        U = [0]
        U_set = {0}
        
        for _ in range(random.randint(6, 12)):
            best_candidate = None
            best_score_val = float('-inf')
            
            # Try a sample of candidates
            candidates_to_try = list(range(1, max_val + 1, random.randint(1, 4)))
            random.shuffle(candidates_to_try)
            
            for candidate in candidates_to_try[:40]:
                if candidate in U_set:
                    continue
                new_U = sorted(U + [candidate])
                new_score = compute_score(new_U)
                if new_score > best_score_val:
                    best_score_val = new_score
                    best_candidate = candidate
            
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
        
        U = sorted(U)
        candidates.append(U)
    
    # Strategy 5: Known good structures with variations
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115, 131, 148])
    candidates.append([0, 1, 3, 7, 12, 20, 30, 44, 60, 79, 101, 125, 151])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86])
    candidates.append([0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77])
    
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
