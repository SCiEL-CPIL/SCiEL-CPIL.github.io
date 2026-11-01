"""Combined program for ternary covering code K_3(7,1) using optimized greedy set cover + fast local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses a greedy set-cover approach followed by local search optimization.
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
    """Solve K_3(7,1) using greedy set cover + fast local search."""
    q, n, radius = 3, 7, 1
    
    # 1. Precompute all words and their balls (neighbors within radius 1)
    all_words = list(product(range(3), repeat=7))
    N = len(all_words) # 2187
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls: for each index, list of indices covered by that codeword
    # Ball(c) = {c} U {words at dist 1}
    balls = []
    for w in all_words:
        covered_indices = [word_to_idx[w]]
        for i in range(7):
            for val in range(3):
                if val != w[i]:
                    new_word = list(w)
                    new_word[i] = val
                    new_word = tuple(new_word)
                    covered_indices.append(word_to_idx[new_word])
        balls.append(covered_indices)
    
    # Convert balls to numpy arrays for faster indexing
    ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
    
    # Precompute set representations for fast intersection/difference checks in local search
    # We store balls as sets of integers for O(1) membership and fast set operations if needed,
    # but since we use numpy arrays for counts, we can also just use boolean masks.
    # For N=2187, boolean masks are 2187 bytes. Creating them on the fly is cheap.
    # However, for the swap check, we need Ball(A) \ Ball(B).
    
    # 2. Greedy Set Cover to get initial solution
    covered = np.zeros(N, dtype=bool)
    codewords = []
    
    # Build adjacency matrix A where A[i, j] = 1 if j in balls[i]
    # This allows vectorized calculation of "new coverage" for all candidates
    A = np.zeros((N, N), dtype=np.uint8)
    for i, b in enumerate(balls):
        A[i, b] = 1
    
    while not covered.all():
        # Calculate how many new points each candidate covers
        # uncovered = ~covered
        # counts = A @ uncovered
        uncovered_mask = ~covered
        counts = A @ uncovered_mask
        
        best_count = np.max(counts)
        if best_count == 0:
            break
            
        best_cw = np.argmax(counts)
        codewords.append(best_cw)
        covered[ball_arrays[best_cw]] = True
        
        if time.time() > deadline - 30:
            break

    if not covered.all():
        # Fallback to finish quickly if greedy was cut short
        while not covered.all():
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            cw = random.choice(uncovered_indices.tolist())
            codewords.append(cw)
            covered[ball_arrays[cw]] = True
            if time.time() > deadline - 5:
                break

    # 3. Initialize coverage counts for Local Search
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        coverage_count[ball_arrays[cw]] += 1
        
    # Helper to remove codeword
    def try_remove(cw_idx):
        b = ball_arrays[cw_idx]
        # Check if any word in b has count 1
        if np.min(coverage_count[b]) == 1:
            return False
        # Safe to remove
        coverage_count[b] -= 1
        return True

    # 4. Redundancy Removal (Greedy Pass)
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
        if time.time() > deadline - 20:
            break

    # 5. Local Search
    # Strategy: Random Removals and Swaps with FAST validity check
    random.seed(42)
    
    # We maintain 'coverage_count'.
    # To check if a swap from A to B is valid:
    # 1. Points in Ball(A) but NOT in Ball(B) will have their count decremented.
    #    If any of these had count 1, they become 0 -> INVALID.
    # 2. Points in Ball(B) but NOT in Ball(A) will have their count incremented.
    #    If any of these had count 0, they become 1 -> Still valid (was 0, now 1).
    #    But we start from a valid state (no 0s), so this is safe.
    # 3. Points in Intersection have net 0 change.
    
    # So the only check needed is: min(coverage_count[Ball(A) \ Ball(B)]) > 1
    
    while time.time() < deadline - 2:
        # Try to remove a random codeword
        if codewords:
            idx_to_try = random.randint(0, len(codewords) - 1)
            cw = codewords[idx_to_try]
            if try_remove(cw):
                codewords.pop(idx_to_try)
                continue
            
        # Try to swap
        if codewords:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Pick a random new codeword
            cw_new = random.randint(0, N - 1)
            
            if cw_new == cw_old:
                continue
            
            b_old = ball_arrays[cw_old]
            b_new = ball_arrays[cw_new]
            
            # Calculate set difference: b_old \ b_new
            # We can use numpy boolean mask for efficiency
            # mask = is_in_old & ~is_in_new
            # But creating full masks is O(N). 
            # Since balls are small (size 16), it's faster to just iterate or use set logic?
            # Let's try a hybrid: use numpy where but only on the subset of indices in b_old.
            
            # Indices in b_old that are NOT in b_new
            # We can check membership of each element in b_old against b_new.
            # Since b_new is a small array, we can use a set for O(1) lookup?
            # Or just use numpy:
            # in_new = np.isin(b_old, b_new)
            # diff = b_old[~in_new]
            
            # np.isin is quite fast for small arrays.
            in_new_mask = np.isin(b_old, b_new)
            diff_indices = b_old[~in_new_mask]
            
            if len(diff_indices) == 0:
                # B covers everything A covers. Swap is definitely valid (B adds more).
                # Actually, if diff is empty, A \ B is empty. No counts decrease.
                # So it's valid.
                codewords[idx_old] = cw_new
                coverage_count[b_old] -= 1
                coverage_count[b_new] += 1
                continue
            
            # Check if any count in diff_indices is 1
            if np.min(coverage_count[diff_indices]) == 1:
                # Invalid swap
                continue
            else:
                # Valid swap
                codewords[idx_old] = cw_new
                coverage_count[b_old] -= 1
                coverage_count[b_new] += 1

    # Convert indices back to lists of integers
    result = [list(all_words[cw]) for cw in codewords]
    return result
