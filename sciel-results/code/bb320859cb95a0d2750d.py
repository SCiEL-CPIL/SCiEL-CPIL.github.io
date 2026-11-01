"""Combined program for ternary covering code K_3(7,1) using greedy set cover + fast local search."""
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
    deadline = start_time + 110  # Safety margin before 120s limit

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
    """Solve K_3(7,1) using greedy set cover + local search."""
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
    
    # 2. Greedy Set Cover to get initial solution
    covered = np.zeros(N, dtype=bool)
    codewords = []
    
    while not covered.all():
        best_cw = -1
        best_count = -1
        
        # Optimization: We can sample or use a heuristic, but N is small.
        # Scanning all candidates is feasible if done efficiently.
        # However, to be safe on time, we can limit candidates or just scan all.
        # Let's scan all but break early if we find a perfect cover? Unlikely.
        # N=2187, 15 neighbors each. 2187 * 15 = 32k ops per candidate.
        # 2187 candidates * 32k = 70M ops per step. Too slow in Python loop.
        
        # Better: Use numpy vectorization for the counting part.
        # But we need to do it for each candidate.
        # Let's try a different heuristic: Randomized Greedy
        # Pick a random subset of candidates, find the best among them.
        
        # Actually, let's just iterate through all but optimize the inner loop.
        # Or, we can maintain a list of uncovered indices and check membership.
        # Membership in a set is O(1).
        
        uncovered_set = set(np.where(~covered)[0])
        
        # To speed up, we can sample candidates.
        # But for the first few steps, scanning all might be okay if we use numpy.
        # Let's try to just scan all candidates but use a fast check.
        
        # Alternative: Just pick the first candidate that covers the most?
        # No, greedy needs the max.
        
        # Let's use a randomized approach to avoid worst-case slowness.
        # Sample 1000 candidates, find the best.
        
        candidates = list(range(N))
        random.shuffle(candidates)
        # If we have many uncovered, sample more.
        sample_size = min(N, 500)
        candidates = candidates[:sample_size]
        
        for i in candidates:
            b = balls[i]
            # Count how many are in uncovered_set
            count = 0
            for nb in b:
                if nb in uncovered_set:
                    count += 1
            if count > best_count:
                best_count = count
                best_cw = i
        
        if best_cw == -1 or best_count == 0:
            # Fallback: pick a random uncovered one
            if uncovered_set:
                best_cw = random.choice(list(uncovered_set))
            else:
                break
                
        codewords.append(best_cw)
        covered[balls[best_cw]] = True
        
        if time.time() > deadline - 30:
            break

    if not covered.all():
        # If greedy didn't finish, we need to finish it quickly or fallback
        # Let's just finish it with random picks to ensure validity
        while not covered.all():
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            # Pick a random uncovered word as a codeword
            cw = random.choice(uncovered_indices.tolist())
            codewords.append(cw)
            covered[balls[cw]] = True
            if time.time() > deadline - 5:
                break

    # 3. Initialize coverage counts for Local Search
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        coverage_count[balls[cw]] += 1
        
    cnt0 = np.sum(coverage_count == 0)
    
    # Helper to check validity
    def is_valid():
        return cnt0 == 0

    # Helper to remove codeword
    def try_remove(cw_idx):
        nonlocal cnt0
        b = balls[cw_idx]
        # Check if any word in b has count 1
        if np.min(coverage_count[b]) == 1:
            return False
        # Safe to remove
        coverage_count[b] -= 1
        # Update cnt0
        new_zeros = np.sum(coverage_count[b] == 0)
        cnt0 += new_zeros
        return True

    # Helper to add codeword
    def try_add(cw_idx):
        nonlocal cnt0
        b = balls[cw_idx]
        old_zeros = np.sum(coverage_count[b] == 0)
        coverage_count[b] += 1
        cnt0 -= old_zeros
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
    # Strategy: Random Removals and Swaps
    random.seed(42)
    
    while time.time() < deadline - 2:
        improvement = False
        
        # Try to remove a random codeword
        if codewords:
            idx_to_try = random.randint(0, len(codewords) - 1)
            cw = codewords[idx_to_try]
            if try_remove(cw):
                codewords.pop(idx_to_try)
                improvement = True
                continue
            
        # Try to swap
        if codewords:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Pick a random new codeword
            cw_new = random.randint(0, N - 1)
            
            if cw_new == cw_old:
                continue
            
            b_old = balls[cw_old]
            b_new = balls[cw_new]
            
            # Simulate swap
            # Decrement old, Increment new
            coverage_count[b_old] -= 1
            coverage_count[b_new] += 1
            
            # Check validity
            if np.min(coverage_count) == 0:
                # Invalid, revert
                coverage_count[b_old] += 1
                coverage_count[b_new] -= 1
            else:
                # Valid swap, accept it (even if it doesn't reduce size immediately,
                # it might enable future removals)
                codewords[idx_old] = cw_new
                improvement = True
            
        # If no improvement for a while, we could try a more aggressive search,
        # but for now, random swaps are a good stochastic local search.
        
    # Convert indices back to lists of integers
    result = [list(all_words[cw]) for cw in codewords]
    return result
