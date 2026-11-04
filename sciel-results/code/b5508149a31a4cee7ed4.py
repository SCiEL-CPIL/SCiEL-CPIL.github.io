import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    # Precompute neighbors for all 2187 words
    # We will use numpy arrays for faster operations
    # idx_to_tuple is still needed for initial construction logic potentially, but we can generate indices directly
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast intersection checks
    # neighbors[i] is an array of indices
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    # Initial Solution: Direct Sum (243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_indices.append(tuple_to_idx(h + list(tail)))
    
    init_indices = list(set(init_indices))
    
    # State
    code = init_indices
    coverage = np.zeros(N, dtype=np.int32)
    
    # Update coverage
    for c in code:
        coverage[neighbors_list[c]] += 1

    # Helper: Check if a candidate c is redundant given current coverage
    # c is redundant if min(coverage[neighbors[c]]) > 1
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    # Helper: Greedy Repair
    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        
        # Optimization: If the set is small, we can just pick words that cover the most.
        # Since sphere size is 15, and we start with 1 removed codeword, uncovered is <= 15.
        # Actually, when removing 1 codeword, the uncovered set is exactly the words that ONLY it covered.
        # Size is <= 15.
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # We can restrict candidates to those that are NOT in current code? 
            # Or just any candidate.
            # To be fast, we can check candidates that are neighbors of the uncovered words?
            # Or just iterate all N? N=2187. 
            # Let's try iterating all N but using numpy for count.
            # Actually, for small sets, a simple loop might be faster due to overhead.
            # But let's try a vectorized approach for counting.
            
            # Convert temp_uncovered to a boolean mask or set for fast lookup?
            # Using a set is O(1) lookup.
            # Let's stick to a simple loop over candidates but optimize the inner count.
            # Since temp_uncovered is small, we can just check if any neighbor is in temp_uncovered.
            
            # Better: Pre-compute for each candidate c, how many of temp_uncovered it covers?
            # That's expensive.
            
            # Let's just iterate candidates.
            # To speed up, we can prioritize candidates that are in the "vicinity" of the uncovered words.
            # Or just random order? No, greedy needs best.
            
            # Let's use a simple heuristic: 
            # For each w in temp_uncovered, its neighbors are candidates.
            # The best candidate must be a neighbor of at least one w in temp_uncovered.
            # So we can build a candidate pool from neighbors of temp_uncovered.
            
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_list[w].tolist())
            
            # Now iterate only over candidate_pool
            for c in candidate_pool:
                if c in code:
                    # We can add codewords that are already in the code? No, we removed one.
                    # If c is in code, it's already there. We are adding new ones.
                    # Wait, if c is in code, it's already counted in coverage.
                    # We are trying to repair a gap created by removal.
                    # The removed word is not in code anymore.
                    # Other words in code are already active.
                    # So we should only consider c NOT in code.
                    continue
                
                # Count intersection
                # neighbors_list[c] is a numpy array
                # temp_uncovered is a set
                # Count how many elements of neighbors_list[c] are in temp_uncovered
                # This is a bit slow in Python.
                # Let's try: sum(1 for x in neighbors_list[c] if x in temp_uncovered)
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                        # Early break if we cover all?
                        if best_count == len(temp_uncovered):
                            break
                
                if best_count == len(temp_uncovered):
                    break

            if best_c == -1:
                # Should not happen
                break

            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)

        return added

    def try_remove():
        nonlocal code, coverage
        removed = False
        # Iterate through a copy
        for c in list(code):
            if is_redundant(c):
                code.remove(c)
                coverage[neighbors_list[c]] -= 1
                removed = True
        return removed

    def kick():
        nonlocal code, coverage
        if not code:
            return False
            
        old_size = len(code)
        
        # Pick a random codeword to remove
        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]
        
        # Simulate removal
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        # Find uncovered words (where coverage becomes 0)
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # It was redundant, but try_remove should have caught it.
            # Just remove it.
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        
        # Greedy repair
        added = greedy_repair(uncovered_set)
        
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            # Accept
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return True
        else:
            return False

    # Main Loop
    last_improvement = time.time()
    
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = try_remove()
        
        # 2. Kick
        if time.time() > deadline:
            break
        
        # Try a few kicks
        kicks_tried = 0
        success = False
        # Increase kick attempts if we are fast
        max_kicks = 20
        while kicks_tried < max_kicks:
            if kick():
                success = True
                kicks_tried = 0
            else:
                kicks_tried += 1
            if time.time() > deadline:
                break
        
        if not success:
            # Stuck in local minimum.
            # Try a "2-kick" or "swap"?
            # For now, just continue. The random nature of kick selection might help eventually.
            # Or we can try to force a swap: Remove 2, Repair.
            # Let's implement a simple 2-removal kick.
            
            if len(code) > 2:
                # Pick 2 random distinct
                i1 = random.randint(0, len(code) - 1)
                i2 = random.randint(0, len(code) - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % len(code)
                
                c1 = code[i1]
                c2 = code[i2]
                
                # Simulate removal
                temp_coverage = coverage.copy()
                temp_coverage[neighbors_list[c1]] -= 1
                temp_coverage[neighbors_list[c2]] -= 1
                
                uncovered = np.where(temp_coverage == 0)[0]
                
                if len(uncovered) == 0:
                    # Both redundant? Unlikely if try_remove passed.
                    # Just remove one?
                    pass
                else:
                    uncovered_set = set(uncovered.tolist())
                    added = greedy_repair(uncovered_set)
                    
                    new_size = len(code) - 2 + len(added)
                    
                    if new_size < len(code):
                        # Accept
                        code.remove(c1)
                        code.remove(c2)
                        coverage[neighbors_list[c1]] -= 1
                        coverage[neighbors_list[c2]] -= 1
                        for c in added:
                            if c not in code: # Ensure distinct
                                code.append(c)
                                coverage[neighbors_list[c]] += 1
                        # Success!
                        # Reset kicks counter?
                        pass

    # Convert indices back to lists of ints
    result = [idx_to_tuple(c) for c in code]
    return result
