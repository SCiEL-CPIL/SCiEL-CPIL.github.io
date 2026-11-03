import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Try to find better solutions with randomized backtracking
    # for target scopes below current best
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try multiple random restarts for this target scope
        found = False
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            attempt += 1
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                    break
        
        if not found:
            # If we can't find a solution at target_scope, try a slightly larger one
            # to make progress, then come back
            if target_scope > lower_bound + 2:
                target_scope = min(best_scope - 1, lower_bound + 5)
            else:
                # We've exhausted near-optimal targets; do more local search
                improved = _improve_scope(best_solution, n_rows, k, deadline)
                new_scope = max(max(row) for row in improved)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = improved
                if best_scope <= lower_bound:
                    return best_solution
                # If no progress, break to avoid wasting time
                break
    
    return best_solution

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_diffs = set()
    rows = []
    
    # Precompute the number of differences needed per row
    diffs_per_row = k * (k + 1) // 2
    
    def build_row(pos, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return list(row_so_far)
        
        remaining = k + 1 - pos  # number of positions left to fill (including current)
        
        # Find valid candidates with better ordering
        valid_candidates = []
        for x in range(row_so_far[-1] + 1, max_scope + 1):
            # Prune: need room for remaining elements
            if x + (remaining - 1) > max_scope:
                break
            
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Order candidates: prefer smaller values, but with some randomness
        # to escape local minima. Use a biased sampling strategy.
        n_candidates = len(valid_candidates)
        
        if n_candidates <= 15:
            candidates_to_try = list(valid_candidates)
            # Shuffle the tail to add randomness
            if n_candidates > 5:
                rng.shuffle(candidates_to_try[5:])
        else:
            # Take the smallest 15, plus a few random from the rest
            smallest = valid_candidates[:15]
            if rng.random() < 0.5:
                extra = rng.sample(valid_candidates[15:], min(5, n_candidates - 15))
                candidates_to_try = smallest + extra
            else:
                candidates_to_try = smallest
            rng.shuffle(candidates_to_try[10:])  # shuffle the tail
        
        for x in candidates_to_try:
            new_diffs = {x - a for a in row_so_far}
            
            # Additional feasibility check: after choosing x, we need
            # (remaining - 1) more elements, each at least 1 apart.
            # The smallest possible next element is x + 1, then x + 2, etc.
            # So we need x + (remaining - 1) <= max_scope (already checked above)
            
            # Also check: the number of new differences we're adding is `pos`
            # (differences from x to each of the pos elements in row_so_far)
            # We need to make sure we don't run out of available differences.
            # This is a weak check but helps prune.
            
            used_diffs.update(new_diffs)
            res = build_row(pos + 1, row_so_far + [x])
            if res is not None:
                return res
            used_diffs.difference_update(new_diffs)
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, [0])
        if row is None:
            return None
        rows.append(row)
        for a_idx in range(len(row)):
            for b_idx in range(a_idx + 1, len(row)):
                used_diffs.add(row[b_idx] - row[a_idx])
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce scope by shifting elements
    # Strategy: repeatedly scan rows from last to first, positions from last to first,
    # and try to move each element as far left as possible.
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2.0:
        improved = False
        passes += 1
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(len(row) - 1, 0, -1):  # from end to start
                original = row[pos]
                # Try to shift this element down to the smallest valid value
                for new_val in range(row[pos - 1] + 1, original):
                    # Compute new differences involving this position
                    new_diffs = set()
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0:
                            break
                        new_diffs.add(d)
                    else:
                        for j in range(pos + 1, len(row)):
                            d = row[j] - new_val
                            if d <= 0:
                                break
                            new_diffs.add(d)
                        else:
                            # All differences are positive, check for conflicts
                            # Old differences involving this position
                            old_diffs = set()
                            for i in range(pos):
                                old_diffs.add(original - row[i])
                            for j in range(pos + 1, len(row)):
                                old_diffs.add(row[j] - original)
                            
                            # Check if new_diffs conflict with other used differences
                            other_diffs = used_diffs - old_diffs
                            if new_diffs & other_diffs:
                                continue
                            
                            # Apply the change
                            used_diffs.difference_update(old_diffs)
                            used_diffs.update(new_diffs)
                            row[pos] = new_val
                            improved = True
                            break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
