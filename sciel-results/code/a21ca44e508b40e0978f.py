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

    # 3. Difference-set sampling approach (structural change)
    # Try to find a better solution by sampling a set of differences and 
    # solving for the row elements via constraint propagation / backtracking.
    # This is a different search space: instead of choosing elements, we 
    # choose the difference set and then find consistent rows.
    attempt = 0
    while time.time() < deadline - 5.0:
        # Try a target scope slightly below current best
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        found = False
        # Try multiple random restarts for this target scope
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            
            # Strategy: sample a random subset of differences from [1, target_scope]
            # that has the right size, then try to construct rows using only those diffs.
            # This is a structural change: we fix the "palette" of differences first.
            solution = _diffset_search(n_rows, k, target_scope, deadline, attempt)
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
            # Fallback to standard backtracking if diffset search fails
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            attempt += 1
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
    
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

def _diffset_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Structural approach: 
    1. Sample a random set D of differences of size n_rows * k*(k+1)/2 from [1, max_scope].
    2. For each row, find a sequence of k+1 integers starting at 0 such that all 
       pairwise differences are in D and distinct within the row.
    3. Ensure all differences across rows are distinct (by construction, if we use 
       disjoint subsets of D for each row, or if we carefully assign).
    
    Actually, a simpler variant: 
    We know we need exactly N = n_rows * k*(k+1)/2 distinct differences.
    We can partition [1, max_scope] into N values.
    Let's try to construct rows one by one, but restrict the allowed differences 
    to a pre-sampled set to guide the search.
    
    Better: Use the fact that if we pick the difference set first, the problem 
    becomes finding a "difference triangle" within that set.
    
    Implementation:
    - Generate a random set of allowed differences of size N.
    - Use backtracking to build rows, but only allow differences that are in this set.
    - This prunes the search space significantly if the set is "dense" enough.
    """
    rng = random.Random(seed_offset * 99999 + 12345)
    N = n_rows * k * (k + 1) // 2
    
    if N > max_scope:
        return None
    
    # Sample N distinct differences from [1, max_scope]
    # We want a "balanced" set, maybe biased towards smaller differences to keep scope small?
    # Or just uniform.
    all_diffs = list(range(1, max_scope + 1))
    sampled_diffs = set(rng.sample(all_diffs, N))
    
    used_diffs = set()
    rows = []
    
    def build_row(pos, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return list(row_so_far)
        
        remaining = k + 1 - pos
        
        # Find valid candidates
        # x must be such that for all a in row_so_far, (x-a) is in sampled_diffs and not in used_diffs
        # Also x must be > row_so_far[-1]
        valid_candidates = []
        # Upper bound for x: we need room for remaining elements.
        # Max x such that x + (remaining-1) <= max_scope is max_scope - (remaining-1)
        max_x = max_scope - (remaining - 1)
        
        for x in range(row_so_far[-1] + 1, max_x + 1):
            ok = True
            new_diffs = set()
            for a in row_so_far:
                d = x - a
                if d not in sampled_diffs:
                    ok = False
                    break
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            
            if ok:
                valid_candidates.append((x, new_diffs))
        
        if not valid_candidates:
            return None
        
        # Limit candidates to avoid explosion
        # Prefer smaller x to keep scope tight
        valid_candidates.sort(key=lambda t: t[0])
        candidates_to_try = valid_candidates[:10]
        
        # Add some randomness
        if len(valid_candidates) > 10:
            extra = rng.sample(valid_candidates[10:], min(5, len(valid_candidates)-10))
            candidates_to_try.extend(extra)
            rng.shuffle(candidates_to_try)
        
        for x, new_diffs in candidates_to_try:
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

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Standard backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_diffs = set()
    rows = []
    
    def build_row(pos, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return list(row_so_far)
        
        remaining = k + 1 - pos
        
        valid_candidates = []
        max_x = max_scope - (remaining - 1)
        for x in range(row_so_far[-1] + 1, max_x + 1):
            new_diffs = {x - a for a in row_so_far}
            if not new_diffs & used_diffs:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        n_candidates = len(valid_candidates)
        if n_candidates <= 15:
            candidates_to_try = list(valid_candidates)
            if n_candidates > 5:
                rng.shuffle(candidates_to_try[5:])
        else:
            smallest = valid_candidates[:15]
            if rng.random() < 0.5:
                extra = rng.sample(valid_candidates[15:], min(5, n_candidates - 15))
                candidates_to_try = smallest + extra
            else:
                candidates_to_try = smallest
            rng.shuffle(candidates_to_try[10:])
        
        for x in candidates_to_try:
            new_diffs = {x - a for a in row_so_far}
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
