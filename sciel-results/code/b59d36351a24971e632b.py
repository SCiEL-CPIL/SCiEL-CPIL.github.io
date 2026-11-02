"""Combined program for ternary covering code K_3(7,1) using vectorized greedy + local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses a vectorized greedy set-cover approach followed by local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    # Fallback: direct sum construction for general case
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    """Direct sum construction as fallback."""
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    # For q=3, use tetracode-based construction
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        # General fallback: all words
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    """Solve K_3(7,1) using vectorized greedy set cover + local search."""
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute ball membership matrix: ball_matrix[i, j] = 1 if word j is within distance 1 of word i
    # Use efficient computation via digit comparison
    # For each word i, its ball contains i itself and all words differing in exactly one position
    ball_matrix = np.zeros((N, N), dtype=np.int8)
    
    # Build index from word tuple to index
    word_to_idx = {}
    for i in range(N):
        word_to_idx[tuple(all_words[i])] = i
    
    # For each word, compute its ball
    for i in range(N):
        w = all_words[i]
        ball_matrix[i, i] = 1
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w[pos] = val
                    j = word_to_idx[tuple(w)]
                    ball_matrix[i, j] = 1
                    w[pos] = orig
    
    # Precompute coverers: for each point, which codewords cover it
    # coverers[p] is a boolean array of length N
    coverers = ball_matrix.T  # coverers[p, i] = 1 if codeword i covers point p
    
    # 1. Vectorized Greedy Set Cover
    covered = np.zeros(N, dtype=np.int8)  # 1 if point is covered
    is_chosen = np.zeros(N, dtype=bool)
    codewords = []
    
    while covered.sum() < N:
        if time.time() > deadline - 40:
            break
        
        # Compute gain for each candidate: number of newly covered points
        # gain[i] = sum of ball_matrix[i, :] where covered == 0
        uncovered_mask = (covered == 0)
        gains = ball_matrix @ uncovered_mask  # Vectorized: for each i, count uncovered points in its ball
        
        # Mask out already chosen
        gains[is_chosen] = -1
        
        best_i = int(np.argmax(gains))
        if gains[best_i] <= 0:
            break
        
        codewords.append(best_i)
        is_chosen[best_i] = True
        covered |= ball_matrix[best_i]
    
    # Ensure full coverage
    if covered.sum() < N:
        uncovered_indices = np.where(covered == 0)[0]
        for pt in uncovered_indices:
            if time.time() > deadline - 10:
                break
            # Find any codeword that covers this point
            candidates = np.where(coverers[pt] & ~is_chosen)[0]
            if len(candidates) > 0:
                cw = int(candidates[0])
            else:
                cw = int(np.random.randint(0, N))
            if not is_chosen[cw]:
                codewords.append(cw)
                is_chosen[cw] = True
            covered |= ball_matrix[cw]
    
    # 2. Local Search Setup
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        cover_count += ball_matrix[cw]
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Helper: Try to remove redundant codewords (vectorized check)
    def try_remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 15:
                break
            cw = codewords[i]
            # Check if all points in ball(cw) have count > 1
            if np.min(cover_count[ball_matrix[cw] == 1]) > 1:
                cover_count -= ball_matrix[cw]
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue
        
        # 2. Try Swaps
        if codewords:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find critical points: points in ball(old_cw) with count == 1
            b_old_mask = ball_matrix[old_cw] == 1
            critical_points = np.where(b_old_mask & (cover_count == 1))[0]
            
            if len(critical_points) == 0:
                # Redundant - remove it
                cover_count -= ball_matrix[old_cw]
                codewords.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
            
            # Find candidates that cover ALL critical points
            # Start with coverers of first critical point
            candidate_mask = coverers[critical_points[0]].copy()
            for p in critical_points[1:]:
                candidate_mask &= coverers[p]
            
            # Remove old_cw and existing codewords
            candidate_mask[old_cw] = False
            for cw in codewords:
                candidate_mask[cw] = False
            
            candidates = np.where(candidate_mask)[0]
            
            if len(candidates) > 0:
                # Pick best candidate: minimize sum of cover_count in its ball
                # (less redundancy = better)
                if len(candidates) > 15:
                    random.shuffle(candidates.tolist())
                    candidates = candidates[:15]
                
                best_new_cw = -1
                min_cost = float('inf')
                
                for new_cw in candidates:
                    b_new_mask = ball_matrix[new_cw] == 1
                    cost = int(np.sum(cover_count[b_new_mask]))
                    if cost < min_cost:
                        min_cost = cost
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    # Perform swap
                    cover_count -= ball_matrix[old_cw]
                    cover_count += ball_matrix[best_new_cw]
                    
                    # Check validity
                    if np.min(cover_count) == 0:
                        # Revert
                        cover_count -= ball_matrix[best_new_cw]
                        cover_count += ball_matrix[old_cw]
                    else:
                        codewords[idx] = best_new_cw
                        improved = True
                        no_improvement_count = 0
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                no_improvement_count = 0

    # Convert indices back to lists of integers
    result = [list(all_words[cw]) for cw in best_codewords]
    return result
