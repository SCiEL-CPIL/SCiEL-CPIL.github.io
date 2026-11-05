import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    L = k + 1  # length of each row

    def scope_of(rows):
        return max(max(r) for r in rows)

    def get_row_mask(row):
        """Compute the bitmask of differences for a single row."""
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def pure_greedy():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                # Calculate new differences if we add x
                new_diffs_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                if valid:
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
        return rows

    def randomized_greedy_row(used_mask, window=50):
        """
        Attempt to construct a single row given a 'used_mask' of forbidden differences.
        Uses randomized greedy within a window to find a valid row.
        Returns the row if successful, None otherwise.
        """
        row = [0]
        x = 0
        max_scope = 200  # Limit search space for speed
        while len(row) < L:
            cands = list(range(x + 1, min(x + window + 1, max_scope + 1)))
            random.shuffle(cands)
            found = False
            for cand in cands:
                new_diffs_mask = 0
                valid = True
                for a in row:
                    d = cand - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                if valid:
                    row.append(cand)
                    used_mask |= new_diffs_mask
                    x = cand
                    found = True
                    break
            if not found:
                # If window search fails, try a wider sequential scan up to max_scope
                cand = x + 1
                while cand <= max_scope:
                    new_diffs_mask = 0
                    valid = True
                    for a in row:
                        d = cand - a
                        bit = 1 << d
                        if used_mask & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                    if valid:
                        row.append(cand)
                        used_mask |= new_diffs_mask
                        x = cand
                        found = True
                        break
                    cand += 1
                if not found:
                    return None
        return row

    def row_reconstruction_search(rows, time_limit):
        """
        Local search that reconstructs entire rows to minimize scope.
        This is more powerful than single-element moves for this problem structure.
        """
        rows = [r[:] for r in rows]
        n = len(rows)
        
        # Precompute masks
        row_masks = [get_row_mask(r) for r in rows]
        global_mask = 0
        for m in row_masks:
            global_mask |= m
            
        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        current_scope = best_scope
        
        t_end = time.time() + time_limit
        last_improvement = time.time()
        
        while time.time() < t_end:
            if time.time() - last_improvement > 10.0:
                # Stagnation: break to allow other methods or return
                break

            # Select a row to reconstruct. 
            # Strategy: Pick the row with the largest last element (most likely to be the bottleneck)
            # or pick randomly to diversify.
            # Let's pick the row with the max scope, or a random one if tied.
            max_idx = -1
            max_val = -1
            for i in range(n):
                if rows[i][-1] > max_val:
                    max_val = rows[i][-1]
                    max_idx = i
            
            # 50% chance to pick the max scope row, 50% random
            if random.random() < 0.5:
                ri = max_idx
            else:
                ri = random.randrange(n)
                
            old_row = rows[ri]
            old_mask = row_masks[ri]
            
            # Remove this row's differences from the global mask
            available_mask = global_mask & ~old_mask
            
            # Try to reconstruct this row with a tighter scope
            # We want the new row's max element to be <= current_scope (ideally smaller)
            # We can bias the randomized_greedy_row to prefer smaller numbers.
            
            # We don't pass a specific scope limit to randomized_greedy_row directly in the signature above,
            # but we can rely on the window and the fact that we stop early if we find a valid row.
            # To enforce a smaller scope, we might need to modify the candidate generation.
            # For now, let's just try to reconstruct. If the new row is smaller, we accept.
            
            new_row = randomized_greedy_row(available_mask, window=30)
            
            if new_row is not None:
                new_mask = get_row_mask(new_row)
                
                # Check if the new mask is actually valid against the available mask
                # (randomized_greedy_row ensures this, but double check for safety)
                if new_mask & available_mask == 0:
                    new_scope = max(max(r) for r in rows if r is not old_row) # Approximate
                    # Actually, let's just compute full scope for simplicity as n is small
                    temp_rows = rows[:]
                    temp_rows[ri] = new_row
                    new_scope = scope_of(temp_rows)
                    
                    if new_scope < current_scope:
                        # Accept improvement
                        global_mask = available_mask | new_mask
                        row_masks[ri] = new_mask
                        rows[ri] = new_row
                        current_scope = new_scope
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                        last_improvement = time.time()
                    elif new_scope == current_scope and random.random() < 0.3:
                        # Accept sideways move to escape local minima
                        global_mask = available_mask | new_mask
                        row_masks[ri] = new_mask
                        rows[ri] = new_row
                        # current_scope remains same
                        # Do not update last_improvement for sideways moves to allow stagnation check to work
                        # But we might get stuck if we only do sideways. 
                        # Let's only accept sideways if it changes the row structure significantly?
                        # For now, accept with low probability.

        return best_rows, best_scope

    # ---------- Main Search ----------
    
    # 1. Pure Greedy for initial solution
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)
    
    # 2. Randomized Greedy Restarts to find diverse good starting points
    candidates = [(best_scope, best_rows)]
    
    t_greedy_end = start_time + (deadline - start_time) * 0.2
    
    attempts = 0
    while time.time() < t_greedy_end and attempts < 50:
        attempts += 1
        # Vary window size for diversity
        w = random.randint(20, 80)
        
        # Re-implementing a quick randomized greedy for the whole set to get diverse starts
        # We can reuse the logic but wrap it in a function that takes a window param
        def randomized_greedy_full(window):
            used_mask = 0
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < L:
                    cands = list(range(x + 1, x + window + 1))
                    random.shuffle(cands)
                    found = False
                    for cand in cands:
                        new_diffs_mask = 0
                        valid = True
                        for a in row:
                            d = cand - a
                            bit = 1 << d
                            if used_mask & bit:
                                valid = False
                                break
                            new_diffs_mask |= bit
                        if valid:
                            row.append(cand)
                            used_mask |= new_diffs_mask
                            x = cand
                            found = True
                            break
                    if not found:
                        # Fallback: sequential scan
                        cand = x + 1
                        while True:
                            new_diffs_mask = 0
                            valid = True
                            for a in row:
                                d = cand - a
                                bit = 1 << d
                                if used_mask & bit:
                                    valid = False
                                    break
                                new_diffs_mask |= bit
                            if valid:
                                row.append(cand)
                                used_mask |= new_diffs_mask
                                x = cand
                                break
                            cand += 1
                    rows.append(row)
            return rows

        rows = randomized_greedy_full(w)
        sc = scope_of(rows)
        if sc < best_scope:
            best_rows = rows
            best_scope = sc
            candidates.insert(0, (sc, rows))
            if best_scope <= 112:
                break
        elif len(candidates) < 5 and sc < best_scope + 10:
            candidates.append((sc, rows))
            
    candidates.sort(key=lambda x: x[0])
    
    # 3. Row Reconstruction Search
    # This is the main improvement over single-element moves.
    # We run it on the best candidate.
    t_recon_end = start_time + (deadline - start_time) * 0.8
    
    if time.time() < t_recon_end:
        improved, new_sc = row_reconstruction_search(candidates[0][1], time_limit=t_recon_end - time.time())
        if new_sc < best_scope:
            best_rows = improved
            best_scope = new_sc
    
    # 4. Final Polish: Try a few more reconstruction cycles if time permits
    if time.time() < deadline - 5.0:
        improved, new_sc = row_reconstruction_search(best_rows, time_limit=deadline - 5.0 - time.time())
        if new_sc < best_scope:
            best_rows = improved
            best_scope = new_sc

    return best_rows
