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
    """Solve K_3(7,1) using greedy set cover + fast local search with 3-for-2 swaps."""
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
    
    # Precompute ball sets for O(1) membership checks
    ball_sets = [set(b) for b in balls]
    
    # 2. Greedy Set Cover to get initial solution
    covered = np.zeros(N, dtype=bool)
    codewords = []
    
    # Build adjacency matrix A where A[i, j] = 1 if j in balls[i]
    A = np.zeros((N, N), dtype=np.uint8)
    for i, b in enumerate(balls):
        A[i, b] = 1
    
    while not covered.all():
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
        if np.min(coverage_count[b]) == 1:
            return False
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

    # 5. Local Search with 1-for-1 swaps AND 3-for-2 swaps
    random.seed(42)
    
    # Precompute for each codeword, the set of words it covers (as frozenset for hashing)
    # Actually, we'll compute on the fly since we need dynamic coverage_count
    
    def try_swap(old_idx, new_cw):
        """Try to swap codeword at old_idx with new_cw. Returns True if successful."""
        cw_old = codewords[old_idx]
        if cw_old == new_cw:
            return False
        
        b_old = ball_arrays[cw_old]
        b_new = ball_arrays[new_cw]
        
        # Points in Ball(A) but NOT in Ball(B) will have their count decremented
        # Check if any of these had count 1
        in_new_mask = np.isin(b_old, b_new)
        diff_indices = b_old[~in_new_mask]
        
        if len(diff_indices) == 0:
            # B covers everything A covers. Valid.
            codewords[old_idx] = new_cw
            coverage_count[b_old] -= 1
            coverage_count[b_new] += 1
            return True
        
        if np.min(coverage_count[diff_indices]) == 1:
            return False
        
        codewords[old_idx] = new_cw
        coverage_count[b_old] -= 1
        coverage_count[b_new] += 1
        return True
    
    def try_3_for_2_swap(idx1, idx2, new_cw):
        """
        Try to replace codewords at idx1 and idx2 with new_cw.
        This removes 2 codewords and adds 1, reducing size by 1.
        Valid if:
        - No word in (Ball(A) \ Ball(C)) has count 1
        - No word in (Ball(B) \ Ball(C)) has count 1
        - No word in (Ball(A) ∩ Ball(B) \ Ball(C)) has count 2 (would drop to 1... wait)
        
        Actually, let's think carefully:
        Current counts include contributions from A, B, and all others.
        After removing A and B and adding C:
        - Words in Ball(A) \\ Ball(B) \\ Ball(C): count decreases by 1. Must have count >= 2.
        - Words in Ball(B) \\ Ball(A) \\ Ball(C): count decreases by 1. Must have count >= 2.
        - Words in Ball(A) ∩ Ball(B) \\ Ball(C): count decreases by 2. Must have count >= 3.
        - Words in Ball(C) \\ (Ball(A) ∪ Ball(B)): count increases by 1. No issue.
        - Words in Ball(C) ∩ (Ball(A) ∪ Ball(B)): net change depends, but no decrease.
        
        So we need to check:
        For each word w:
          delta = -[w in Ball(A)] - [w in Ball(B)] + [w in Ball(C)]
          If delta < 0, we need coverage_count[w] + delta >= 1, i.e., coverage_count[w] >= 1 - delta
        
        Since delta is at most -2, we need:
        - If w in A∩B∩~C: count >= 3
        - If w in A\\B∩~C or B\\A∩~C: count >= 2
        """
        cw1 = codewords[idx1]
        cw2 = codewords[idx2]
        
        if new_cw == cw1 or new_cw == cw2:
            return False
        
        b1 = ball_arrays[cw1]
        b2 = ball_arrays[cw2]
        bc = ball_arrays[new_cw]
        
        # Get the union of words affected
        # Words that lose coverage: in (A ∪ B) \ C
        # We need to check counts for these
        
        # Efficient approach: check each word in (A ∪ B) that is not in C
        set_a = ball_sets[cw1]
        set_b = ball_sets[cw2]
        set_c = ball_sets[new_cw]
        
        # Words only in A (not in B, not in C)
        only_a = set_a - set_b - set_c
        # Words only in B (not in A, not in C)
        only_b = set_b - set_a - set_c
        # Words in both A and B but not in C
        both_not_c = (set_a & set_b) - set_c
        
        # Check counts
        if only_a:
            if np.min(coverage_count[list(only_a)]) < 2:
                return False
        if only_b:
            if np.min(coverage_count[list(only_b)]) < 2:
                return False
        if both_not_c:
            if np.min(coverage_count[list(both_not_c)]) < 3:
                return False
        
        # Perform the swap
        coverage_count[b1] -= 1
        coverage_count[b2] -= 1
        coverage_count[bc] += 1
        
        # Remove the two codewords and add the new one
        # Be careful with indices if idx1 > idx2
        if idx1 < idx2:
            codewords.pop(idx2)
            codewords.pop(idx1)
        else:
            codewords.pop(idx1)
            codewords.pop(idx2)
        codewords.append(new_cw)
        
        return True
    
    # Main local search loop
    iterations = 0
    while time.time() < deadline - 2:
        iterations += 1
        
        # Try to remove a random codeword
        if codewords and random.random() < 0.3:
            idx_to_try = random.randint(0, len(codewords) - 1)
            cw = codewords[idx_to_try]
            if try_remove(cw):
                codewords.pop(idx_to_try)
                continue
        
        # Try 1-for-1 swap
        if codewords and random.random() < 0.6:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Pick a random new codeword, bias towards uncovered areas
            cw_new = random.randint(0, N - 1)
            
            if try_swap(idx_old, cw_new):
                continue
        
        # Try 3-for-2 swap (remove 2, add 1)
        if len(codewords) >= 2 and random.random() < 0.15:
            idx1 = random.randint(0, len(codewords) - 1)
            idx2 = random.randint(0, len(codewords) - 1)
            if idx1 == idx2:
                continue
            
            # Pick a new codeword
            cw_new = random.randint(0, N - 1)
            
            if try_3_for_2_swap(idx1, idx2, cw_new):
                continue
        
        # Occasionally do a greedy re-optimization pass
        if iterations % 500 == 0 and time.time() < deadline - 10:
            # Try to remove redundant codewords
            i = 0
            while i < len(codewords):
                if try_remove(codewords[i]):
                    codewords.pop(i)
                else:
                    i += 1

    # Convert indices back to lists of integers
    result = [list(all_words[cw]) for cw in codewords]
    return result
