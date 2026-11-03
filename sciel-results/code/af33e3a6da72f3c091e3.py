"""Difference Triangle Set (4,7) solver.

Strategy:
1. Fast greedy baseline.
2. Constraint Satisfaction Problem (CSP) solver with backtracking and MRV heuristic
   to construct valid rows sequentially, avoiding global conflict graphs.
3. Randomized restarting to explore different solution spaces.
4. Local compaction to reduce the scope further.
5. Simulated annealing for final polishing.
"""
import time
import random
import sys
from collections import defaultdict

def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _solve_csp_row(n_rows, k, used_diffs, current_rows, rng, max_scope_hint):
    """
    Attempt to construct one more row given the current set of rows and used differences.
    Returns the new row if successful, None otherwise.
    Uses backtracking with MRV (Minimum Remaining Values) heuristic.
    """
    L = k + 1
    # We are constructing a new row. 
    # The state is the partial row constructed so far.
    # We want to find a full row of length L starting with 0.
    
    # To speed up, we can bound the search. 
    # The maximum value in the new row shouldn't exceed some heuristic bound.
    # A simple bound is the current max scope + a bit, or just let it run and prune.
    
    def backtrack(pos, current_row):
        if pos == L:
            return list(current_row)
        
        # Determine valid values for the next position
        # The next value must be > current_row[-1]
        # And the differences (val - current_row[i]) for all i < pos must not be in used_diffs
        # Also, differences between new val and previous elements in current_row must be unique within the row
        # But since we are building sequentially, we just need to check against global used_diffs
        # and ensure no internal duplicates (which is guaranteed if we pick strictly increasing and check global used, 
        # wait, internal duplicates like row[2]-row[0] == row[3]-row[1] are not caught by global used if they are new.
        # Actually, the requirement is "all positive differences ... are distinct over all rows".
        # So differences within the SAME row must also be distinct.
        
        # Let's collect used diffs within the current partial row
        internal_diffs = set()
        for i in range(len(current_row)):
            for j in range(i + 1, len(current_row)):
                internal_diffs.add(current_row[j] - current_row[i])
        
        # We need to pick a value v such that:
        # 1. v > current_row[-1]
        # 2. For all i in current_row, v - current_row[i] is not in used_diffs (global)
        # 3. For all i in current_row, v - current_row[i] is not in internal_diffs (local)
        
        # Optimization: Instead of checking all i, we can just check if the set of new diffs intersects with (used_diffs U internal_diffs)
        
        # Heuristic: Try values in a randomized order or increasing order?
        # Increasing order is usually better for compactness.
        
        # Determine a reasonable upper bound for v.
        # If we are at position pos, we have L - pos - 1 elements left.
        # The minimum increment is 1. So v must be at least current_row[-1] + 1.
        # We can also estimate a maximum to avoid infinite loops if no solution exists in this branch.
        # A safe upper bound for the entire row is max_scope_hint.
        # So v <= max_scope_hint - (L - pos - 1) (since we need at least 1 for each remaining step).
        
        max_v = max_scope_hint - (L - pos - 1)
        min_v = current_row[-1] + 1
        
        # Randomize the order of trying values? 
        # For compactness, we prefer smaller values. 
        # But sometimes a slightly larger value allows a much smaller overall scope later.
        # Let's stick to increasing order for determinism and compactness bias, 
        # but we can add a small random perturbation in the restart loop.
        
        for v in range(min_v, max_v + 1):
            # Check conflicts
            new_diffs = {v - a for a in current_row}
            if new_diffs & (used_diffs | internal_diffs):
                continue
            
            # If valid, recurse
            res = backtrack(pos + 1, current_row + [v])
            if res is not None:
                return res
        
        return None
    
    return backtrack(1, [0])

def _construct_with_csp(n_rows, k, rng, time_limit):
    """
    Construct n_rows using CSP backtracking with restarting.
    """
    start_time = time.time()
    best_rows = None
    best_max = float('inf')
    
    # Initial scope hint. We can start with a greedy estimate.
    # Greedy usually gives a decent upper bound.
    greedy_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    greedy_max = _max_entry(greedy_rows)
    
    # We will try to find a solution with max entry < greedy_max.
    # We can iterate on the max_scope_hint downwards.
    
    # Let's try multiple restarts with different random seeds for the CSP solver's internal choices
    # (though currently it's deterministic increasing, we can randomize the start or order).
    # To make it stochastic, we can shuffle the order of candidate values slightly or pick a random start.
    # Better: We can randomize the "max_scope_hint" or the order of rows.
    
    # Let's try a few different scope hints, starting from greedy_max - 1 downwards.
    # This is a "branch and bound" on the scope.
    
    for scope in range(greedy_max - 1, 100, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to construct the set with this scope limit
        # We need to construct n_rows sequentially.
        # For each row, we call _solve_csp_row.
        
        current_rows = []
        used_diffs = set()
        success = True
        
        for i in range(n_rows):
            # To add variety, we can shuffle the order in which we try values?
            # _solve_csp_row currently tries increasing values.
            # We can modify _solve_csp_row to accept a random seed or a flag to shuffle.
            # For now, let's rely on the scope reduction to force different structures.
            
            # Also, we can randomize the "starting point" or use a different heuristic.
            # Let's just call it. If it fails, we break.
            
            # Optimization: Pass a tighter bound for the last rows?
            # The bound is global scope.
            
            new_row = _solve_csp_row(n_rows, k, used_diffs, current_rows, rng, scope)
            
            if new_row is None:
                success = False
                break
            
            current_rows.append(new_row)
            # Update used_diffs
            L = len(new_row)
            for a in range(L):
                for b in range(a + 1, L):
                    used_diffs.add(new_row[b] - new_row[a])
        
        if success:
            # We found a valid set with max entry <= scope
            # Since we iterate scope downwards, the first one found is the best for this scope.
            # But we want the smallest max entry.
            # The actual max entry of the constructed rows might be less than scope.
            actual_max = _max_entry(current_rows)
            if actual_max < best_max:
                best_max = actual_max
                best_rows = current_rows
            
            # If we found a solution, we can break if we want to just find *a* good one,
            # but we want the *best* one. Since we are iterating scope downwards,
            # the first successful scope is the minimal scope bound that works.
            # However, the actual max entry might be significantly lower than the scope bound.
            # So we should continue to see if a smaller scope bound yields an even smaller actual max?
            # No, if a solution exists with max entry M, it also exists with scope bound M.
            # If we find a solution with scope bound S, its max entry is <= S.
            # If we try S-1 and fail, then no solution with max entry <= S-1 exists.
            # So the first successful scope S gives us the optimal max entry?
            # Not necessarily. The CSP solver might fail to find a solution within scope S
            # even if one exists, due to backtracking limits or heuristic issues.
            # But our backtracking is exhaustive within the scope.
            # So if it returns None, no solution exists with max entry <= scope.
            # Thus, the first scope S for which it returns a row is the minimal possible max entry.
            
            # Wait, _solve_csp_row is exhaustive for a single row given the previous rows.
            # So if it fails, no valid row exists with max entry <= scope given the previous rows.
            # This implies no complete set exists with max entry <= scope.
            # Therefore, the first successful scope is indeed the optimal max entry.
            
            # We can break early!
            break
    
    if best_rows is None:
        # Fallback to greedy
        return greedy_rows
    
    return best_rows

def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = set()
    L = k + 1
    for row in work:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
            for pos in range(L - 1, 0, -1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                
                old_diffs = set()
                for i in range(L):
                    if i < pos:
                        old_diffs.add(row[pos] - row[i])
                    elif i > pos:
                        old_diffs.add(row[i] - row[pos])
                
                for new_val in range(lower, old_val):
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    
    return work

def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = set()
    for row in work:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max
    
    start_temp = max(1, current_max // 8)
    end_temp = 0.05
    cooling = 0.9995
    
    t = start_temp
    stagnation_counter = 0
    max_stagnation = 2000
    
    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                t *= cooling
                continue
            
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                t *= cooling
                continue
            
            old_diffs = set()
            for i in range(L):
                if i < pos:
                    old_diffs.add(row[pos] - row[i])
                elif i > pos:
                    old_diffs.add(row[i] - row[pos])
            
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            
            other = used_set - old_diffs
            if not (new_diffs & other):
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                
                # Try to compact further
                for p in range(L - 1, 0, -1):
                    if p == pos:
                        continue
                    ov = row[p]
                    lo = row[p - 1] + 1
                    if lo >= ov:
                        continue
                    od = set()
                    for i in range(L):
                        if i < p:
                            od.add(row[p] - row[i])
                        elif i > p:
                            od.add(row[i] - row[p])
                    for nv in range(lo, ov):
                        if p < L - 1 and nv >= row[p + 1]:
                            break
                        nd = set()
                        for i in range(L):
                            if i < p:
                                nd.add(nv - row[i])
                            elif i > p:
                                nd.add(row[i] - nv)
                        ot = used_set - od
                        if nd & ot:
                            continue
                        for d in od:
                            used_set.discard(d)
                        for d in nd:
                            used_set.add(d)
                        row[p] = nv
                        break
                
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        
        elif move_type < 0.7:
            # Increase a random element, then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(2, current_max // 3))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
            
            old_diffs = set()
            for i in range(L):
                if i < pos:
                    old_diffs.add(row[pos] - row[i])
                elif i > pos:
                    old_diffs.add(row[i] - row[pos])
            
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            
            other = used_set - old_diffs
            if not (new_diffs & other):
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                
                # Recompact
                for p in range(L - 1, 0, -1):
                    ov = row[p]
                    lo = row[p - 1] + 1
                    if lo >= ov:
                        continue
                    od = set()
                    for i in range(L):
                        if i < p:
                            od.add(row[p] - row[i])
                        elif i > p:
                            od.add(row[i] - row[p])
                    for nv in range(lo, ov):
                        if p < L - 1 and nv >= row[p + 1]:
                            break
                        nd = set()
                        for i in range(L):
                            if i < p:
                                nd.add(nv - row[i])
                            elif i > p:
                                nd.add(row[i] - nv)
                        ot = used_set - od
                        if nd & ot:
                            continue
                        for d in od:
                            used_set.discard(d)
                        for d in nd:
                            used_set.add(d)
                        row[p] = nv
                        break
                
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        else:
            # Full recompact
            for r_idx in range(n_rows):
                row = work[r_idx]
                for pos in range(L - 1, 0, -1):
                    ov = row[pos]
                    lo = row[pos - 1] + 1
                    if lo >= ov:
                        continue
                    od = set()
                    for i in range(L):
                        if i < pos:
                            od.add(row[pos] - row[i])
                        elif i > pos:
                            od.add(row[i] - row[pos])
                    for nv in range(lo, ov):
                        if pos < L - 1 and nv >= row[pos + 1]:
                            break
                        nd = set()
                        for i in range(L):
                            if i < pos:
                                nd.add(nv - row[i])
                            elif i > pos:
                                nd.add(row[i] - nv)
                        ot = used_set - od
                        if nd & ot:
                            continue
                        for d in od:
                            used_set.discard(d)
                        for d in nd:
                            used_set.add(d)
                        row[pos] = nv
                        break
            
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        
        if stagnation_counter > max_stagnation:
            fresh = _greedy_construct(n_rows, k, rng=rng)
            if _check_valid(fresh, k):
                fresh_max = _max_entry(fresh)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh]
                    current_max = best_max
                    work = [list(r) for r in fresh]
                    used_set = set()
                    for row in work:
                        for i in range(L):
                            for j in range(i + 1, L):
                                used_set.add(row[j] - row[i])
                    t = start_temp
                    stagnation_counter = 0
        
        t *= cooling
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 140
    
    # 1) Greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)
    
    # 2) CSP Construction
    # We allocate 60% of the time to CSP construction
    csp_deadline = start_time + (time_limit - start_time) * 0.6
    rng_csp = random.Random(123)
    
    csp_result = _construct_with_csp(n_rows, k, rng_csp, csp_deadline - start_time)
    if csp_result and _check_valid(csp_result, k):
        cmax = _max_entry(csp_result)
        if cmax < best_max:
            best_rows = csp_result
            best_max = cmax
    
    # 3) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, time_limit - 20)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax
    
    # 4) Simulated annealing
    if time.time() < time_limit - 3:
        rng_sa = random.Random(456)
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng_sa, time_limit - 3)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax
    
    # 5) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    
    return best_rows
