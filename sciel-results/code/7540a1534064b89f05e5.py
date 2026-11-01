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
    
    # Convert to numpy arrays for fast indexing
    balls_arr = np.array(balls, dtype=np.int32)
    
    # 2. Greedy Set Cover to get initial solution
    covered = np.zeros(N, dtype=bool)
    codewords = []
    
    while not covered.all():
        # Find the candidate that covers the most uncovered words
        # Use numpy to count uncovered neighbors for all candidates at once
        # uncovered_mask is boolean array of size N
        # For each candidate i, count = sum(uncovered_mask[balls[i]])
        # This is equivalent to balls_arr @ uncovered_mask (as int)
        
        # Optimization: Only consider candidates that cover at least 1 uncovered word
        # We can compute scores for all candidates efficiently
        scores = balls_arr @ covered.astype(np.int8) # This gives covered count, we want uncovered
        # Actually, we want uncovered count:
        # uncovered_count[i] = sum(1 for nb in balls[i] if not covered[nb])
        # = len(balls[i]) - sum(1 for nb in balls[i] if covered[nb])
        # = 15 - (balls_arr @ covered)
        
        # Let's just compute directly using the uncovered set for clarity and speed
        # Since N is small, let's try the matrix multiplication approach
        # covered is bool, so covered.astype(int) works
        covered_counts = balls_arr @ covered.astype(np.int32)
        uncovered_counts = 15 - covered_counts
        
        # Find the candidate with max uncovered count
        # But we only care about candidates that are not already in codewords?
        # Actually, adding a duplicate is useless, but greedy usually avoids it implicitly
        # if we track covered. However, a codeword might cover new things if others are removed?
        # No, in greedy construction, we just add.
        # We should pick the one with max uncovered_counts.
        
        # If all are 0, we are stuck (should not happen if covered is not all True)
        if uncovered_counts.max() == 0:
            # Fallback: pick a random uncovered word
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) > 0:
                best_cw = random.choice(uncovered_indices.tolist())
            else:
                break
        else:
            # Among those with max score, pick one. If tie, pick random?
            # For determinism in greedy, pick first max. For diversity, pick random.
            max_score = uncovered_counts.max()
            candidates = np.where(uncovered_counts == max_score)[0]
            # Pick random from candidates to allow different runs
            best_cw = random.choice(candidates.tolist())
        
        codewords.append(best_cw)
        covered[balls_arr[best_cw]] = True
        
        if time.time() > deadline - 30:
            break

    # Ensure full coverage
    if not covered.all():
        while not covered.all():
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            # Pick a random uncovered word as a codeword
            cw = random.choice(uncovered_indices.tolist())
            codewords.append(cw)
            covered[balls_arr[cw]] = True
            if time.time() > deadline - 5:
                break

    # 3. Initialize coverage counts for Local Search
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        coverage_count[balls_arr[cw]] += 1
        
    # Helper to check validity
    def is_valid():
        return np.min(coverage_count) >= 0 # And specifically no zeros if we want strict cover?
        # Actually, valid means min >= 1.
        # But we track cnt0 for efficiency.
    
    cnt0 = np.sum(coverage_count == 0)
    
    # Helper to remove codeword
    def try_remove(cw_idx):
        nonlocal cnt0
        b = balls_arr[cw_idx]
        # Check if any word in b has count 1
        if np.min(coverage_count[b]) == 1:
            return False
        # Safe to remove
        old_counts = coverage_count[b]
        coverage_count[b] -= 1
        # Update cnt0
        # Words that were 1 become 0
        new_zeros = np.sum(coverage_count[b] == 0)
        # Words that were 2 become 1 (no change in zero count)
        # Words that were >2 become >1 (no change)
        # So we just add the new zeros
        cnt0 += new_zeros
        return True

    # Helper to add codeword
    def try_add(cw_idx):
        nonlocal cnt0
        b = balls_arr[cw_idx]
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

    # 5. Local Search with Simulated Annealing
    # We want to reduce the size of codewords.
    # Moves:
    # 1. Remove a codeword (if valid, i.e., no word becomes uncovered)
    # 2. Swap a codeword (remove one, add one)
    # 3. Add a codeword (only if it allows removing more later? Or just accept worse states)
    
    # Since we want to minimize size, "cost" is len(codewords).
    # We accept worse states (larger size) with probability exp(-delta/T) to escape local minima.
    
    current_code = list(codewords)
    current_size = len(current_code)
    best_code = list(current_code)
    best_size = current_size
    
    # Re-initialize coverage for the current best
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in current_code:
        coverage_count[balls_arr[cw]] += 1
    cnt0 = np.sum(coverage_count == 0)
    
    T_start = 1.0
    T_end = 0.01
    n_steps = 20000
    rng = random.Random(42)
    
    for step in range(n_steps):
        if time.time() > deadline - 2:
            break
        
        alpha = step / n_steps
        T = T_start * (T_end / T_start) ** alpha
        
        # Choose a move type
        # 1. Try to remove a random codeword
        # 2. Try to swap a random codeword with a random new one
        # 3. (Optional) Try to add a random codeword (to allow larger sets temporarily)
        
        move_type = rng.random()
        
        if move_type < 0.4 and current_size > 0:
            # Try remove
            idx_to_remove = rng.randrange(current_size)
            cw = current_code[idx_to_remove]
            
            # Check if removable
            b = balls_arr[cw]
            if np.min(coverage_count[b]) > 1:
                # Safe to remove
                coverage_count[b] -= 1
                # Update cnt0 (should not increase if min was > 1, but just in case)
                # Actually if min > 1, no zeros are created.
                current_code.pop(idx_to_remove)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = list(current_code)
            else:
                # Cannot remove, reject move (or accept with prob? No, it's invalid)
                pass
                
        elif move_type < 0.9 and current_size > 0:
            # Try swap
            idx_old = rng.randrange(current_size)
            cw_old = current_code[idx_old]
            
            # Pick a random new codeword
            cw_new = rng.randrange(N)
            
            if cw_new in current_code:
                continue
            
            b_old = balls_arr[cw_old]
            b_new = balls_arr[cw_new]
            
            # Simulate swap
            # Decrement old, Increment new
            coverage_count[b_old] -= 1
            coverage_count[b_new] += 1
            
            # Check validity
            if np.min(coverage_count) >= 1:
                # Valid swap
                # Size remains same
                current_code[idx_old] = cw_new
                # No size change, so no best update unless we consider diversity?
                # But maybe this leads to better future removals.
            else:
                # Invalid swap, revert
                coverage_count[b_old] += 1
                coverage_count[b_new] -= 1
                
        else:
            # Try add (to escape local minima where we are stuck at a size that can't be reduced by swap/remove)
            cw_new = rng.randrange(N)
            if cw_new in current_code:
                continue
            
            b_new = balls_arr[cw_new]
            coverage_count[b_new] += 1
            current_code.append(cw_new)
            current_size += 1
            
            # This increases size, so it's a worse state.
            # We accept it always in this branch to explore, but we don't update best.
            # Actually, we should accept with prob exp(-1/T) to control it.
            # But since we're in the "else" branch, we already decided to try it.
            # Let's revert if it doesn't help immediately? No, SA keeps state.
            # However, if we just add, we might get stuck in large sets.
            # It's better to only add if it allows a subsequent removal in the same step?
            # That's complex. Let's just allow it and rely on the cooling to reduce T.
            # But wait, if we add, size increases. We should only keep it if it's "good" or with prob.
            # Standard SA: accept worse with prob exp(-delta/T).
            # delta = 1 (size increase).
            if rng.random() > np.exp(-1.0 / T):
                # Reject the addition
                coverage_count[b_new] -= 1
                current_code.pop()
                current_size -= 1

    # Final redundancy removal on best_code
    # Re-initialize coverage for best_code
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in best_code:
        coverage_count[balls_arr[cw]] += 1
    
    i = 0
    while i < len(best_code):
        b = balls_arr[best_code[i]]
        if np.min(coverage_count[b]) > 1:
            coverage_count[b] -= 1
            best_code.pop(i)
        else:
            i += 1
        if time.time() > deadline - 1:
            break

    # Convert indices back to words
    return [list(all_words[i]) for i in best_code]
