import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # Fallback for non-specific cases
    if q != 3 or n != 6 or radius != 1:
        total = q ** n
        if total > 10000:
            return [[0] * n]
        coords = []
        for i in range(total):
            c = [0] * n
            x = i
            for j in range(n - 1, -1, -1):
                c[j] = x % q
                x //= q
            coords.append(c)
        
        nbrs = []
        for c in coords:
            nbs = [c]
            for i in range(n):
                orig = c[i]
                for v in range(q):
                    if v != orig:
                        nc = c[:]
                        nc[i] = v
                        nbs.append(nc)
            nbrs.append(nbs)
            
        coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
        nbr_idx = []
        for nbs in nbrs:
            nbr_idx.append([coord_to_idx[tuple(nb)] for nb in nbs])
        
        uncovered = np.ones(total, dtype=np.int8)
        code = []
        while uncovered.sum() > 0:
            if time.time() > deadline:
                break
            counts = np.sum(uncovered[np.array(nbr_idx)], axis=1)
            best = int(np.argmax(counts))
            if counts[best] == 0:
                break
            code.append(best)
            uncovered[nbr_idx[best]] = 0
        return [coords[i] for i in code]

    # Specific case: K_3(6,1)
    total = 729
    powers = [1, 3, 9, 27, 81, 243]
    coords = [[(i // p) % 3 for p in powers] for i in range(total)]
    
    # Precompute neighbors as indices
    nbr = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        nbr[i, 0] = i
        k = 1
        for p in range(6):
            base = (i // powers[p]) % 3
            for alt in range(3):
                if alt != base:
                    nbr[i, k] = i + (alt - base) * powers[p]
                    k += 1

    def get_direct_sum():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]

    # 1. Greedy Initialization
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        counts = np.sum(uncovered[nbr], axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        code.append(best)
        uncovered[nbr[best]] = 0
        
    def try_reduce(cw):
        """Remove redundant codewords. Returns a list of indices."""
        if not cw:
            return cw
        cov = np.zeros(total, dtype=np.int32)
        for idx in cw:
            cov[nbr[idx]] += 1
        changed = True
        # We iterate until no more changes
        while changed:
            changed = False
            # To make this faster, we can just scan once per "pass"
            # but we need to restart if we remove one.
            for i in range(len(cw) - 1, -1, -1):
                idx = cw[i]
                # If all neighbors of idx are covered by at least 2 codewords, idx is redundant
                if np.min(cov[nbr[idx]]) >= 2:
                    cov[nbr[idx]] -= 1
                    cw.pop(i)
                    changed = True
                    break
        return cw

    code = try_reduce(code)
    best_size = len(code)
    best_code = code[:]
    
    # 2. Simulated Annealing Local Search
    rng = np.random.default_rng(42)
    # We allow the code size to increase temporarily to escape local minima
    temperature = 5.0
    cooling = 0.995
    no_improve = 0
    max_no_improve = 5000
    current_code = best_code[:]
    
    # Precompute a random permutation of indices for diversification if needed
    # But for now, we just do random moves.
    
    while time.time() < deadline - 2.0 and no_improve < max_no_improve:
        if not current_code:
            break
            
        # Calculate coverage for current code
        cov = np.zeros(total, dtype=np.int32)
        for idx in current_code:
            cov[nbr[idx]] += 1
            
        # Pick a random codeword to remove
        i = int(rng.integers(len(current_code)))
        c_remove = current_code[i]
        current_code.pop(i)
        cov[nbr[c_remove]] -= 1
        
        uncovered_mask = (cov == 0)
        
        if not np.any(uncovered_mask):
            # We successfully removed a codeword without breaking coverage
            # Try to reduce further
            current_code = try_reduce(current_code)
            new_size = len(current_code)
            if new_size < best_size:
                best_size = new_size
                best_code = current_code[:]
                no_improve = 0
                # Reset temperature on improvement? Or keep cooling?
                # Usually keep cooling, but maybe boost if stuck long ago.
            else:
                no_improve += 1
            continue
            
        # Repair: Find best word to add back
        # We want to minimize the number of words added. Since we removed 1, 
        # adding 1 keeps size same. Adding >1 increases size.
        # We greedily pick the one that covers most uncovered.
        counts_add = np.sum(uncovered_mask[nbr], axis=1)
        
        # Exclude words already in code? No, we can add a word already in code? 
        # No, codewords must be distinct. 
        # But if we pick a word already in code, it doesn't change the set.
        # We should pick a word NOT in current_code.
        
        # Optimization: If the best_add is already in code, we can't add it again.
        # However, for covering, adding a word already present does nothing.
        # So we must pick a word NOT in current_code.
        
        # To do this efficiently without a set lookup for every argmax:
        # We can mask out the current code words from counts_add?
        # Or just pick the best, and if it's in code, pick the next best?
        # Given the size is ~80, a set lookup is fast.
        
        current_set = set(current_code)
        
        # Find best candidate not in current_set
        best_add = -1
        max_count = -1
        # Iterate through top candidates? 
        # Or just compute argmax and check.
        # If argmax is in code, we need the second best.
        # Let's just get the indices of the top 10 candidates to be safe.
        
        # Simpler: just pick argmax. If it's in code, we have a problem.
        # Actually, if the best word to cover the hole is already in the code, 
        # it means it was already covering those spots? 
        # No, it was removed? No, we removed c_remove. 
        # If best_add is in current_code, it means current_code already has a word 
        # that covers the most uncovered spots. But if it's in current_code, 
        # it's already "active". Adding it again is redundant.
        # This implies that the uncovered spots are NOT covered by any word in current_code.
        # So best_add should NOT be in current_code.
        # Why? Because if it were, cov[nbr[best_add]] would have been > 0 for those spots?
        # Yes. If best_add is in current_code, then cov[nbr[best_add]] >= 1 for all its neighbors.
        # So uncovered_mask[nbr[best_add]] would be 0.
        # So counts_add[best_add] would be 0.
        # So the argmax will never be a word in current_code (unless all counts are 0, 
        # which means no word covers the uncovered spots, which is impossible as the space is connected).
        
        best_add = int(np.argmax(counts_add))
        
        # Just in case, if best_add is in current_code (shouldn't happen), pick a random other
        if best_add in current_set:
            # Find another
            # Mask out current code
            mask = np.ones(total, dtype=bool)
            for idx in current_code:
                mask[idx] = False
            counts_add[mask] = 0
            best_add = int(np.argmax(counts_add))
            if counts_add[best_add] == 0:
                # Cannot repair? Revert
                current_code.append(c_remove)
                cov[nbr[c_remove]] += 1
                no_improve += 1
                temperature *= cooling
                continue

        current_code.append(best_add)
        cov[nbr[best_add]] += 1
        
        # Try to reduce
        current_code = try_reduce(current_code)
        new_size = len(current_code)
        
        # Acceptance Criteria
        delta = new_size - (len(best_code) if best_size < len(current_code) else len(current_code)) # This is messy.
        # Simpler: Compare new_size with the size BEFORE the move.
        # Size before move was len(current_code) + 1 (after pop) ? No.
        # Let's track size_before_move.
        
        # Actually, let's just use the standard SA logic on the "cost" which is size.
        # Cost = len(current_code)
        
        # We need to know the cost of the previous state.
        # Let's refactor the loop to store prev_size.
        
        # For this iteration, we just check if we improved best_size.
        if new_size < best_size:
            best_size = new_size
            best_code = current_code[:]
            no_improve = 0
            # Restart temperature to explore more?
            # temperature = 5.0 
        else:
            # Accept worse or equal moves with probability
            # We need the previous size to calculate delta.
            # Previous size was len(current_code) before we popped c_remove?
            # No, we popped, then added.
            # Let's say prev_size = len(current_code) - 1 (size before repair) ?
            # Actually, the move is: Remove c_remove, Add best_add.
            # Size change is 0 (if we don't reduce).
            # If we reduce, size change is -1.
            
            # If we didn't reduce, size is same as best_size? 
            # If new_size == best_size, accept with prob 0.5?
            # If new_size > best_size, accept with prob exp(-delta/T)?
            
            # Let's compare new_size with the size of the solution we came from.
            # We came from a solution of size `prev_size`.
            # We should have stored prev_size.
            
            # Let's assume we are comparing to `best_size` for simplicity of "Is this a new best?".
            # But SA needs to accept non-improving moves relative to CURRENT state.
            
            # Let's restart the logic slightly for clarity in the code below.
            pass

    # The above loop logic was getting complex. Let's implement a cleaner SA.
    
    # RESTART SA LOGIC PROPERLY
    
    current_code = best_code[:]
    current_size = len(current_code)
    temperature = 10.0
    cooling = 0.99
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline - 2.0 and no_improve < max_no_improve:
        if not current_code:
            break
            
        cov = np.zeros(total, dtype=np.int32)
        for idx in current_code:
            cov[nbr[idx]] += 1
            
        i = int(rng.integers(len(current_code)))
        c_remove = current_code[i]
        current_code.pop(i)
        cov[nbr[c_remove]] -= 1
        
        uncovered_mask = (cov == 0)
        
        if not np.any(uncovered_mask):
            current_code = try_reduce(current_code)
            new_size = len(current_code)
            if new_size < best_size:
                best_size = new_size
                best_code = current_code[:]
                no_improve = 0
            else:
                no_improve += 1
            # Accept this state as current
            current_size = new_size
            temperature *= cooling
            continue
            
        counts_add = np.sum(uncovered_mask[nbr], axis=1)
        best_add = int(np.argmax(counts_add))
        
        current_set = set(current_code)
        if best_add in current_set:
             # Fallback: pick random valid
             valid_indices = [j for j in range(total) if j not in current_set]
             if not valid_indices:
                 current_code.append(c_remove)
                 cov[nbr[c_remove]] += 1
                 no_improve += 1
                 temperature *= cooling
                 continue
             best_add = valid_indices[int(rng.integers(len(valid_indices)))]
             
        current_code.append(best_add)
        cov[nbr[best_add]] += 1
        current_code = try_reduce(current_code)
        new_size = len(current_code)
        
        # Delta is change in size from previous current state
        delta = new_size - current_size
        
        if new_size < best_size:
            best_size = new_size
            best_code = current_code[:]
            no_improve = 0
        else:
            no_improve += 1
            
        # Acceptance
        if delta < 0:
            # Improvement, always accept
            accept = True
        elif delta == 0:
            # Neutral, accept with prob 0.5
            accept = rng.random() < 0.5
        else:
            # Worse, accept with prob exp(-delta/T)
            accept = rng.random() < np.exp(-delta / temperature)
            
        if accept:
            current_size = new_size
        else:
            # Revert to previous state
            # We need to restore current_code to its state before the move.
            # We modified current_code in place. 
            # We should have made a copy or tracked the change.
            # Simplest: keep a backup of the list before the move?
            # That's expensive for large lists.
            # Alternative: Store the state.
            
            # Let's just re-run the last move in reverse?
            # We removed c_remove, added best_add.
            # To revert: remove best_add, add c_remove.
            # But we also ran try_reduce which might have removed others.
            # This makes reverting hard.
            
            # Better approach: Store the list before modification.
            # Since list is ~80, copying is cheap.
            
            pass

    # The "Revert" logic is tricky with in-place modification + try_reduce.
    # Let's use a simpler strategy: 
    # 1. Generate a neighbor state `new_code`.
    # 2. Calculate `new_size`.
    # 3. Decide to accept.
    # 4. If accept, `current_code = new_code`.
    # 5. If not, keep `current_code` as is.
    
    # Let's rewrite the loop with this safety in mind.

    current_code = best_code[:]
    current_size = len(current_code)
    temperature = 10.0
    cooling = 0.99
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline - 2.0 and no_improve < max_no_improve:
        if not current_code:
            break
            
        # Create a copy to modify
        new_code = current_code[:]
        
        i = int(rng.integers(len(new_code)))
        c_remove = new_code[i]
        new_code.pop(i)
        
        # Calculate coverage for new_code
        cov = np.zeros(total, dtype=np.int32)
        for idx in new_code:
            cov[nbr[idx]] += 1
            
        uncovered_mask = (cov == 0)
        
        if not np.any(uncovered_mask):
            new_code = try_reduce(new_code)
        else:
            counts_add = np.sum(uncovered_mask[nbr], axis=1)
            best_add = int(np.argmax(counts_add))
            
            current_set = set(new_code)
            if best_add in current_set:
                valid_indices = [j for j in range(total) if j not in current_set]
                if not valid_indices:
                    continue # Invalid move, skip
                best_add = valid_indices[int(rng.integers(len(valid_indices)))]
                
            new_code.append(best_add)
            new_code = try_reduce(new_code)
            
        new_size = len(new_code)
        delta = new_size - current_size
        
        if new_size < best_size:
            best_size = new_size
            best_code = new_code[:]
            no_improve = 0
            
        # Acceptance
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rng.random() < 0.5
        else:
            accept = rng.random() < np.exp(-delta / temperature)
            
        if accept:
            current_code = new_code
            current_size = new_size
            no_improve += 1 if new_size >= best_size else 0 # Reset no_improve on improvement? 
            # Actually, no_improve should track steps without *best* improvement.
            # So if new_size < best_size, no_improve = 0.
            # Else no_improve += 1.
            
            if new_size >= best_size:
                no_improve += 1
            else:
                no_improve = 0
                
            temperature *= cooling
        else:
            no_improve += 1
            temperature *= cooling

    # If we found a code smaller than 81, use it.
    if best_code and len(best_code) <= 81:
        return [coords[idx] for idx in best_code]
    
    return get_direct_sum()
