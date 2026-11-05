"""
Difference triangle set (n_rows, k) constructor.

Strategy:
- Fast parallel greedy construction to seed valid rows with balanced scopes.
- Iterative "Smallest-Fit" row rebuild:
  * Maintain a global difference usage set (the shared state).
  * Select a row to rebuild (prioritize rows that determine the global scope, else random).
  * Rebuild the row via DFS that places each entry at the smallest value such that
    all resulting differences are not in the global used set (excluding this row's own diffs).
  * Accept the rebuild if it yields a valid row with a smaller (or equal) max entry,
    updating the global used set and the global scope.
- Deadline guard ensures return before 150s.
"""
import random
import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    # Helper: compute all positive differences in a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - ri)
        return diffs

    # Helper: get max entry
    def get_max_entry(rows):
        m = 0
        for row in rows:
            if row and row[-1] > m:
                m = row[-1]
        return m

    # Helper: validate rows
    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    # Fast parallel greedy construction
    def parallel_greedy(seed=42):
        rng = random.Random(seed)
        rows = []
        # We will build rows one by one, but try to keep their scopes balanced.
        # Start with an empty global used set.
        used = set()
        for r in range(n_rows):
            row = [0]
            # Target scope: try to keep it around the current average or a reasonable bound.
            # For the first row, we can be greedy.
            for pos in range(k):
                # Find smallest x > row[-1] such that all diffs are unused
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row:
                    used.add(x - a)
            rows.append(row)
        return rows

    # Initialize
    best_rows = parallel_greedy(seed=42)
    if not is_valid(best_rows):
        # Fallback: simple greedy (shouldn't happen with parallel_greedy logic above, but just in case)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row:
                    used.add(x - a)
            rows.append(row)
        best_rows = rows

    best_max = get_max_entry(best_rows)
    
    # Build global used set and per-row diffs
    global_used = set()
    for row in best_rows:
        for d in row_diffs(row):
            global_used.add(d)

    # Main optimization loop: Iterative Smallest-Fit Row Rebuild
    # We will repeatedly pick a row and try to rebuild it with a smaller scope.
    # To avoid getting stuck, we allow some randomization in row selection and
    # occasionally accept a slightly larger row if it helps future reductions (though we prefer smaller).
    
    rng = random.Random(12345)
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 2.0:
        # Check time
        if time.time() > deadline - 1.0:
            break

        # Select row to rebuild
        # Strategy: 50% chance pick the row with the largest max entry (bottleneck),
        # 50% chance pick a random row.
        if rng.random() < 0.5:
            # Find row with max entry
            max_val = -1
            target_idx = 0
            for i, row in enumerate(best_rows):
                if row[-1] > max_val:
                    max_val = row[-1]
                    target_idx = i
        else:
            target_idx = rng.randint(0, n_rows - 1)

        old_row = best_rows[target_idx][:]
        old_row_max = old_row[-1]
        
        # Compute the set of differences NOT used by other rows
        # This is the "available" difference set for the new row.
        # Instead of building a huge set, we can just check against global_used minus old_row's diffs.
        other_used = global_used - row_diffs(old_row)

        # Target scope for the new row: try to be smaller than old_row_max
        # Start with a tight bound: old_row_max - 1
        # If that fails, we might try slightly larger if we are stuck? 
        # For now, let's just try to find ANY valid row with max < old_row_max.
        # If we can't find one, we try to find one with max == old_row_max (to allow permutation/shuffling)
        # or just skip.
        
        # Let's try to find a row with max entry < old_row_max.
        # If found, update.
        # If not found, try to find a row with max entry <= old_row_max that is different?
        # Actually, just trying to reduce is the main goal.
        
        success = False
        new_row = None
        
        # Try to rebuild with a max scope of old_row_max - 1
        # We use DFS with backtracking.
        # To keep it fast, we limit the depth or branching factor? 
        # The "smallest fit" greedy is usually very fast. Backtracking is rare if scope is tight.
        
        def rebuild_row(scope_limit, other_diffs):
            """
            Attempt to build a row of length k+1 starting at 0, increasing,
            with last element <= scope_limit, such that all diffs are not in other_diffs.
            Returns the row if successful, else None.
            """
            row = [0]
            
            def dfs(pos, current_row, current_max):
                if pos == k:
                    return current_row[:]
                
                # Next element must be > current_row[-1] and <= scope_limit
                start_val = current_row[-1] + 1
                # We can prune if start_val > scope_limit
                if start_val > scope_limit:
                    return None
                
                # Try values from start_val to scope_limit
                # To be fast, we iterate in increasing order.
                # If a value fails, we try the next.
                for x in range(start_val, scope_limit + 1):
                    # Check if x creates any conflict with other_diffs or within current_row
                    ok = True
                    for a in current_row:
                        d = x - a
                        if d in other_diffs:
                            ok = False
                            break
                        # Also check if d is already in current_row's diffs?
                        # Since current_row is built incrementally, we only need to check against
                        # the fixed "other_diffs". Diffs within current_row are unique by construction
                        # if we ensure x-a is not equal to any previous (x'-a')?
                        # Wait, if we place x, the new diffs are x - a_i for all a_i in current_row.
                        # These must be distinct from each other? Yes, because x is fixed and a_i are distinct.
                        # They must be distinct from other_diffs.
                        # They must be distinct from diffs already in current_row?
                        # Diffs in current_row are (a_j - a_i) for j>i.
                        # New diffs are (x - a_i).
                        # Since x > a_j > a_i, x - a_i > x - a_j.
                        # Also x - a_i > a_j - a_i (since x > a_j).
                        # So new diffs are always larger than any diff within the prefix?
                        # No. x - a_i vs a_j - a_k.
                        # Example: row [0, 2, 5]. Diffs: 2, 5, 3.
                        # Add x=7. Diffs: 7, 5, 2.
                        # 5 is in old diffs. Conflict!
                        # So we MUST check against all diffs in current_row as well.
                        pass
                    
                    # Let's optimize the check.
                    # Precompute current_row diffs? 
                    # For small k, just check all pairs.
                    # But checking against other_diffs is O(k).
                    # Checking against internal diffs is O(k^2) if done naively.
                    
                    # Let's just do the full check for simplicity, k is small (e.g., 7).
                    # Diffs in current_row:
                    # for i in range(len(current_row)):
                    #   for j in range(i+1, len(current_row)):
                    #       d = current_row[j] - current_row[i]
                    #       if d in other_diffs: return False (should not happen if current_row is valid w.r.t other_diffs)
                    #       if d == x - a_i: conflict
                    
                    # Efficient check:
                    # 1. For each a in current_row, d = x - a.
                    #    if d in other_diffs: fail.
                    # 2. Check if d is in the set of diffs of current_row.
                    
                    # Let's maintain a set of diffs for current_row.
                    # But we are in DFS, so passing it around is overhead.
                    # Given k is small, let's just recompute or check directly.
                    
                    # Direct check:
                    for i in range(len(current_row)):
                        d = x - current_row[i]
                        if d in other_diffs:
                            ok = False
                            break
                        # Check against internal diffs
                        # d = x - current_row[i]
                        # Is there j, l in current_row such that current_row[l] - current_row[j] == d?
                        # Since d > 0, and current_row is increasing.
                        # We can just check all pairs.
                        for j in range(len(current_row)):
                            for l in range(j+1, len(current_row)):
                                if current_row[l] - current_row[j] == d:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                        if not ok:
                            break
                    
                    if ok:
                        new_row_candidate = current_row + [x]
                        # Recurse
                        res = dfs(pos + 1, new_row_candidate, x)
                        if res is not None:
                            return res
                        
                return None
            
            return dfs(0, row, 0)

        # Try to find a row with max < old_row_max
        # We try scope_limit from old_row_max - 1 downwards? 
        # Or just start with old_row_max - 1. If it fails, maybe try a bit lower?
        # Actually, if old_row_max - 1 fails, it might be because the structure is tight.
        # Let's try a few lower bounds.
        found = False
        for trial in range(5):
            limit = old_row_max - 1 - trial
            if limit < k: # Minimum possible max entry is k (0, 1, ..., k)
                break
            new_row = rebuild_row(limit, other_used)
            if new_row is not None:
                # Verify it's valid w.r.t other_used
                # (rebuild_row should ensure this)
                # Update global state
                old_diffs = row_diffs(old_row)
                global_used.difference_update(old_diffs)
                new_diffs = row_diffs(new_row)
                global_used.update(new_diffs)
                
                best_rows[target_idx] = new_row
                new_max = get_max_entry(best_rows)
                
                if new_max < best_max:
                    best_max = new_max
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                found = True
                break
        
        if not found:
            no_improve_count += 1
            # If we are stuck, maybe try a different row or break
            if no_improve_count > max_no_improve:
                # Try a random restart of the optimization loop? 
                # Or just break to save time.
                # Let's try a few more random rows before giving up on this "phase".
                if no_improve_count > max_no_improve * 2:
                    break

    # Final validation
    if not is_valid(best_rows):
        # Fallback
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row:
                    used.add(x - a)
            rows.append(row)
        best_rows = rows

    return best_rows
