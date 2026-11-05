import time
import numpy as np
import random
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for radius=1
    # neighbors[idx] = list of indices within Hamming distance <= 1
    space_size = q ** n
    
    # Powers of q for fast digit access
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    neighbors = np.empty(space_size, dtype=object)
    
    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n-1-pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)
    
    # 1. Generate an initial valid code using greedy set cover with heap
    uncovered = np.ones(space_size, dtype=bool)
    initial_code = []
    
    # Precompute cover matrix for fast greedy updates
    cover_masks = np.zeros((space_size, space_size), dtype=bool)
    for i in range(space_size):
        cover_masks[i, neighbors[i]] = True
        
    # Greedy Set Cover with heap: pick word covering most uncovered
    # Use a max-heap (negate counts for Python's min-heap)
    # We'll periodically rebuild the heap to avoid stale entries
    
    # Initial heap: all candidates with their coverage count
    heap = []
    counts = cover_masks @ uncovered.astype(np.int8)
    for i in range(space_size):
        heap.append((-counts[i], i))
    heapq.heapify(heap)
    
    heap_version = 0
    last_heap_build = time.time()
    
    while uncovered.any():
        # Check if we need to rebuild the heap (stale entries)
        if time.time() - last_heap_build > 0.5 or len(heap) < 100:
            counts = cover_masks @ uncovered.astype(np.int8)
            heap = [(-counts[i], i) for i in range(space_size) if counts[i] > 0]
            heapq.heapify(heap)
            last_heap_build = time.time()
        
        # Pop until we find a valid entry
        found = False
        while heap:
            neg_count, idx = heapq.heappop(heap)
            count = -neg_count
            # Check if this is still valid (idx not in code and count matches)
            if idx not in set(initial_code) and uncovered[neighbors[idx]].any():
                # Verify count is still accurate
                actual_count = np.sum(cover_masks[idx] & uncovered)
                if actual_count == count:
                    found = True
                    break
                else:
                    # Stale entry, continue
                    continue
            else:
                # Invalid or already selected
                continue
        
        if not found:
            # Heap is empty or all entries are stale
            # Rebuild
            counts = cover_masks @ uncovered.astype(np.int8)
            heap = [(-counts[i], i) for i in range(space_size) if counts[i] > 0]
            heapq.heapify(heap)
            last_heap_build = time.time()
            
            if not heap:
                break
            
            # Try again
            neg_count, best_idx = heapq.heappop(heap)
            count = -neg_count
            if count == 0:
                break
        else:
            best_idx = idx
        
        initial_code.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        if time.time() - t_start > deadline - 5:
            break

    # Ensure we have a valid code
    if not uncovered.all():
        while not uncovered.all():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            w_idx = uncovered_indices[0]
            initial_code.append(w_idx)
            uncovered[neighbors[w_idx]] = False
            if time.time() - t_start > deadline - 5:
                break

    # 2. Fixed-Size ILS with Swap and Re-insertion
    # State: 
    # - code: list of codeword indices (fixed size K)
    # - cov: array of length space_size, cov[i] = number of codewords covering i
    # - in_set: boolean array, in_set[i] = True if i is in current_code
    
    def try_remove(code, cov, in_set_mask, c_idx):
        """Try to remove codeword c_idx. Returns (success, new_code, new_cov, new_in_set)."""
        c = code[c_idx]
        # Check if c is redundant: all its neighbors have cov > 1
        # If any neighbor has cov == 1, removing c makes it uncovered (cov=0)
        # So we need min(cov[neighbors[c]]) > 1
        if np.min(cov[neighbors[c]]) > 1:
            # Safe to remove
            new_code = code[:]
            new_code.pop(c_idx)
            
            new_cov = cov.copy()
            new_cov[neighbors[c]] -= 1
            
            new_in_set = in_set_mask.copy()
            new_in_set[c] = False
            
            return True, new_code, new_cov, new_in_set
        return False, None, None, None

    def try_swap(code, cov, in_set_mask, c_idx, new_c):
        """Try to swap codeword at c_idx with new_c. Returns (success, new_code, new_cov, new_in_set)."""
        if in_set_mask[new_c]:
            return False, None, None, None
            
        c = code[c_idx]
        if c == new_c:
            return False, None, None, None
            
        # Simulate removal of c and addition of new_c
        # We need to ensure that after removal, all neighbors of c are still covered
        # OR they become covered by new_c? No, swap is atomic.
        # Condition: For all x in neighbors[c], if cov[x] == 1, then x MUST be in neighbors[new_c].
        # Basically, the set of neighbors covered ONLY by c must be a subset of neighbors[new_c].
        
        # Let's check:
        # critical_nbs_c = [x for x in neighbors[c] if cov[x] == 1]
        # If any x in critical_nbs_c is NOT in neighbors[new_c], swap is invalid.
        
        # Optimization: 
        # We can compute the "exclusive" coverage of c.
        # Actually, just check:
        # For each x in neighbors[c]:
        #    if cov[x] == 1:
        #        if not cover_masks[new_c, x]: return False
        
        # This is O(deg(c) * check). deg is 15. Very fast.
        
        for x in neighbors[c]:
            if cov[x] == 1:
                if not cover_masks[new_c, x]:
                    return False, None, None, None
        
        # Swap is valid
        new_code = code[:]
        new_code[c_idx] = new_c
        
        new_cov = cov.copy()
        new_cov[neighbors[c]] -= 1
        new_cov[neighbors[new_c]] += 1
        
        new_in_set = in_set_mask.copy()
        new_in_set[c] = False
        new_in_set[new_c] = True
        
        return True, new_code, new_cov, new_in_set

    def try_insert(code, cov, in_set_mask, new_c):
        """Try to insert new_c into code. Returns (success, new_code, new_cov, new_in_set)."""
        if in_set_mask[new_c]:
            return False, None, None, None
            
        new_code = code + [new_c]
        new_cov = cov.copy()
        new_cov[neighbors[new_c]] += 1
        
        new_in_set = in_set_mask.copy()
        new_in_set[new_c] = True
        
        return True, new_code, new_cov, new_in_set

    def try_double_swap(code, cov, in_set_mask, c_idx1, c_idx2, new_c1, new_c2):
        """Try to swap two codewords with two new ones. Returns (success, new_code, new_cov, new_in_set)."""
        if in_set_mask[new_c1] or in_set_mask[new_c2]:
            return False, None, None, None
        if new_c1 == new_c2:
            return False, None, None, None
            
        c1 = code[c_idx1]
        c2 = code[c_idx2]
        if c1 == new_c1 or c1 == new_c2 or c2 == new_c1 or c2 == new_c2:
            return False, None, None, None
            
        # Check validity: for each of c1, c2, all neighbors with cov==1 must be covered by new_c1 or new_c2
        # Build a set of neighbors covered by new_c1 or new_c2
        new_covers = set()
        for x in neighbors[new_c1]:
            new_covers.add(x)
        for x in neighbors[new_c2]:
            new_covers.add(x)
        
        for x in neighbors[c1]:
            if cov[x] == 1 and x not in new_covers:
                return False, None, None, None
        for x in neighbors[c2]:
            if cov[x] == 1 and x not in new_covers:
                return False, None, None, None
        
        # Double swap is valid
        new_code = code[:]
        new_code[c_idx1] = new_c1
        new_code[c_idx2] = new_c2
        
        new_cov = cov.copy()
        new_cov[neighbors[c1]] -= 1
        new_cov[neighbors[c2]] -= 1
        new_cov[neighbors[new_c1]] += 1
        new_cov[neighbors[new_c2]] += 1
        
        new_in_set = in_set_mask.copy()
        new_in_set[c1] = False
        new_in_set[c2] = False
        new_in_set[new_c1] = True
        new_in_set[new_c2] = True
        
        return True, new_code, new_cov, new_in_set

    # Initialize state
    current_code = initial_code[:]
    current_cov = np.zeros(space_size, dtype=np.int32)
    current_in_set = np.zeros(space_size, dtype=bool)
    for c in current_code:
        current_cov[neighbors[c]] += 1
        current_in_set[c] = True
        
    # Initial Pruning (Remove redundant words)
    changed = True
    while changed:
        changed = False
        for i in range(len(current_code)):
            success, new_code, new_cov, new_in_set = try_remove(current_code, current_cov, current_in_set, i)
            if success:
                current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                changed = True
                break # Restart loop after modification

    best_code = current_code[:]
    best_size = len(current_code)
    
    rng = np.random.default_rng(42)
    
    # ILS Loop with enhanced moves
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - t_start < deadline - 5:
        # 1. Try to remove a random word
        if len(current_code) > 0:
            i = rng.integers(0, len(current_code))
            success, new_code, new_cov, new_in_set = try_remove(current_code, current_cov, current_in_set, i)
            if success:
                # Improvement!
                current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                if len(current_code) < best_size:
                    best_code = current_code[:]
                    best_size = len(current_code)
                no_improve_count = 0
                continue
        
        # 2. Try to swap a random word with a random candidate
        if len(current_code) > 0:
            i = rng.integers(0, len(current_code))
            new_c = rng.integers(0, space_size)
            success, new_code, new_cov, new_in_set = try_swap(current_code, current_cov, current_in_set, i, new_c)
            if success:
                # Same size. Accept with some probability to escape local minima
                if rng.random() < 0.15:
                    current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                no_improve_count += 1
                continue
        
        # 3. Try double swap when stuck
        if no_improve_count > 15 and len(current_code) >= 2:
            i1 = rng.integers(0, len(current_code))
            i2 = rng.integers(0, len(current_code))
            if i1 != i2:
                new_c1 = rng.integers(0, space_size)
                new_c2 = rng.integers(0, space_size)
                success, new_code, new_cov, new_in_set = try_double_swap(
                    current_code, current_cov, current_in_set, i1, i2, new_c1, new_c2
                )
                if success and rng.random() < 0.2:
                    current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                    no_improve_count = 0
                    continue
        
        # 4. Try to insert a random word when very stuck
        if no_improve_count > max_no_improve:
            new_c = rng.integers(0, space_size)
            success, new_code, new_cov, new_in_set = try_insert(current_code, current_cov, current_in_set, new_c)
            if success:
                current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                no_improve_count = 0
                continue
        
        no_improve_count += 1
        
        # Prune step every 15 no-improvements
        if no_improve_count % 15 == 0:
            changed = True
            while changed:
                changed = False
                for i in range(len(current_code)):
                    success, new_code, new_cov, new_in_set = try_remove(current_code, current_cov, current_in_set, i)
                    if success:
                        current_code, current_cov, current_in_set = new_code, new_cov, new_in_set
                        changed = True
                        break
                if changed:
                    if len(current_code) < best_size:
                        best_code = current_code[:]
                        best_size = len(current_code)
                    no_improve_count = 0

    # Final check and return
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
        
    if np.any(final_cov == 0):
        # Fallback to initial_code
        return [idx_to_word(c) for c in initial_code]
    else:
        return [idx_to_word(c) for c in best_code]
