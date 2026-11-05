"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    L = k + 1  # Length of each row
    diffs_per_row = k * (k + 1) // 2

    def scope_of(rows):
        return max(max(r) for r in rows) if rows else 0

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def pure_greedy():
        """Fast greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def backtrack_construct(row_idx, current_row, used_diffs, bound, time_limit):
        """
        Backtracking search for a single row with a given bound.
        Returns the row if found within time, else None.
        """
        if time.time() > time_limit:
            return None
        
        if len(current_row) == L:
            return current_row[:]
        
        last_val = current_row[-1]
        # Try next values starting from last_val + 1 up to bound
        # We can optimize by skipping values that obviously create conflicts, 
        # but for small k, simple iteration is fine.
        for x in range(last_val + 1, bound + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            # Check if this choice leads to a dead end? 
            # Simple check: if we run out of space?
            # For now, just recurse.
            
            res = backtrack_construct(row_idx, current_row + [x], used_diffs | new_diffs, bound, time_limit)
            if res is not None:
                return res
        return None

    def randomized_greedy(window=80):
        """Greedy with shuffled candidate window."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                cands = list(range(x + 1, x + window + 1))
                random.shuffle(cands)
                found = False
                for cand in cands:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
                if not found:
                    cand = x + 1
                    while True:
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            break
                        cand += 1
            rows.append(row)
        return rows

    def local_search(rows, max_iters, time_limit):
        """
        Enhanced Local Search.
        """
        rows = [r[:] for r in rows]
        best = [r[:] for r in rows]
        best_scope = scope_of(rows)
        n = len(rows)
        L = len(rows[0])

        row_diff_sets = [row_diffs_set(r) for r in rows]
        all_diffs = set()
        for s in row_diff_sets:
            all_diffs |= s

        T = 50.0
        T_min = 0.1
        cooling = 0.999

        it = 0
        while it < max_iters and time.time() < time_limit:
            it += 1
            T = max(T_min, T * cooling)

            ri = random.randrange(n)
            old_row = rows[ri]
            old_diffs = row_diff_sets[ri]
            old_max = max(old_row)
            
            # Move types:
            # 1. Change a single entry (most common)
            # 2. Change the last entry specifically (to reduce scope)
            # 3. Swap two entries (maintain sort? No, must be increasing. So effectively changing values)
            
            if random.random() < 0.8:
                # Single entry change
                # Bias towards changing larger indices to potentially reduce max
                ci = random.randrange(1, L)
                # Weighted choice: higher probability for later indices
                if random.random() < 0.5:
                    ci = random.randrange(L // 2, L)
                    
                old_val = old_row[ci]
                
                # Generate candidates
                candidates = set()
                # Local search around old value
                spread = 50 if ci == L-1 else 20
                for d in range(-spread, spread + 1):
                    nv = old_val + d
                    if nv >= 1:
                        candidates.add(nv)
                
                # Global jumps
                for _ in range(10):
                    nv = random.randint(1, best_scope + 100)
                    candidates.add(nv)
                
                # Enforce ordering
                lower = old_row[ci - 1] + 1
                upper = (old_row[ci + 1] - 1) if ci + 1 < L else best_scope + 200
                candidates = [c for c in candidates if lower <= c <= upper]
                
                if not candidates:
                    continue

                random.shuffle(candidates)
                for nv in candidates:
                    if nv == old_val:
                        continue
                    new_row = old_row[:]
                    new_row[ci] = nv
                    new_diffs = row_diffs_set(new_row)

                    # Check internal consistency
                    if len(new_diffs) != diffs_per_row:
                        continue

                    # Check no collision with other rows
                    other_diffs = all_diffs - old_diffs
                    if new_diffs & other_diffs:
                        continue

                    # Valid move
                    new_scope = max(best_scope, max(new_row))
                    delta = new_scope - best_scope

                    # Acceptance criterion
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        all_diffs = (all_diffs - old_diffs) | new_diffs
                        row_diff_sets[ri] = new_diffs
                        rows[ri] = new_row

                        if new_scope < best_scope:
                            best = [r[:] for r in rows]
                            best_scope = new_scope
                        break
            else:
                # Special move: Try to strictly reduce the last element
                # This is a targeted move to improve the objective
                ci = L - 1
                old_val = old_row[ci]
                # Try values lower than old_val
                # But must be > old_row[ci-1]
                lower = old_row[ci - 1] + 1
                if old_val > lower:
                    # Try a random value in (lower, old_val)
                    nv = random.randint(lower, old_val - 1)
                    new_row = old_row[:]
                    new_row[ci] = nv
                    new_diffs = row_diffs_set(new_row)
                    if len(new_diffs) != diffs_per_row:
                        continue
                    other_diffs = all_diffs - old_diffs
                    if new_diffs & other_diffs:
                        continue
                    
                    new_scope = max(best_scope, max(new_row))
                    delta = new_scope - best_scope
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        all_diffs = (all_diffs - old_diffs) | new_diffs
                        row_diff_sets[ri] = new_diffs
                        rows[ri] = new_row
                        if new_scope < best_scope:
                            best = [r[:] for r in rows]
                            best_scope = new_scope
                        break

        return best, best_scope

    # ---------- Main Search Strategy ----------
    
    # 1. Get a baseline with pure greedy
    best_rows = pure_greedy()
    best_scope = scope_of(best_rows)
    
    # 2. Try to improve initial construction using backtracking with a tight bound
    # Start with the greedy bound, try to go lower
    current_bound = best_scope
    # We will try to construct rows one by one with decreasing bounds
    # This is expensive, so we limit it.
    
    if time.time() < deadline - 30.0:
        # Attempt a structured search for better initial rows
        # We'll try to find a configuration with max scope < current_bound
        # using backtracking for each row sequentially.
        
        # Sort rows from current best by max value descending to prioritize fixing the worst
        sorted_indices = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        
        # We don't rebuild from scratch, we just try to improve the worst row first?
        # Actually, rebuilding from scratch with backtracking is safer for validity.
        
        # Let's do a few restarts of a "Guided Backtracking"
        # where we use the greedy solution as a hint for bounds.
        
        for restart in range(5):
            if time.time() > deadline - 30.0:
                break
            
            # Try to construct with a bound slightly lower than current best
            target_bound = best_scope - random.randint(1, 10)
            if target_bound < 112:
                target_bound = 112 # Don't go below theoretical min
            
            used = set()
            new_rows = []
            success = True
            for i in range(n_rows):
                # Time limit per row
                row_time_limit = time.time() + (deadline - 30.0 - time.time()) / (n_rows - i)
                row = backtrack_construct(i, [0], used, target_bound, row_time_limit)
                if row is None:
                    success = False
                    break
                new_rows.append(row)
                used |= row_diffs_set(row)
            
            if success:
                sc = scope_of(new_rows)
                if sc < best_scope:
                    best_rows = new_rows
                    best_scope = sc
                    if best_scope <= 112:
                        break

    # 3. Local Search Phase
    # Run local search on the best solution found
    if time.time() < deadline - 10.0:
        improved, sc = local_search(best_rows, max_iters=50000, time_limit=deadline - 10.0)
        if sc < best_scope:
            best_rows = improved
            best_scope = sc

    # 4. Final Polish: Try to reduce scope by 1 at a time
    if time.time() < deadline - 5.0:
        # Simple hill climbing on the max value
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Find row with max value
            max_idx = max(range(n_rows), key=lambda i: best_rows[i][-1])
            row = best_rows[max_idx]
            last_val = row[-1]
            
            # Try to reduce last_val
            for nv in range(last_val - 1, row[-2], -1):
                new_row = row[:-1] + [nv]
                new_diffs = row_diffs_set(new_row)
                if len(new_diffs) != diffs_per_row:
                    continue
                
                # Check conflicts
                # Recompute all_diffs for simplicity in this small loop?
                # Better: maintain global set
                old_diffs = row_diffs_set(row)
                # Get all diffs from other rows
                other_diffs = set()
                for j in range(n_rows):
                    if j != max_idx:
                        other_diffs |= row_diffs_set(best_rows[j])
                
                if new_diffs & other_diffs:
                    continue
                
                # Valid
                best_rows[max_idx] = new_row
                best_scope = scope_of(best_rows)
                changed = True
                break

    return best_rows
