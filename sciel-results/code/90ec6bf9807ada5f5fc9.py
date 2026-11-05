"""Difference triangle set (7,5) – improved constructive search."""
import time
import random
from itertools import combinations

# We have a hard wall-clock budget from the evaluator; keep a safety margin.
_DEADLINE_SECONDS = 150.0
_SAFETY_MARGIN = 10.0  # seconds of buffer


def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _validate(rows, k):
    """Check that all rows are valid and all positive differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & used:
            return False
        used |= rd
    return True


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng):
    """
    Greedy construction: for each row, pick the smallest next mark whose
    intra-row differences are all unused.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _solve_attempt(n_rows, k, target_scope, rng, time_limit, start_time):
    """
    Backtracking search to construct n_rows valid rows with max entry < target_scope.
    Returns (rows, scope) if successful, else (None, None).
    """
    def time_up():
        return time.time() - start_time > time_limit

    used_diffs = set()
    rows = []
    
    def _backtrack(row_idx, current_row, current_max_val):
        if time_up():
            return False
            
        if row_idx == n_rows:
            # All rows constructed
            actual_scope = max(r[-1] for r in rows)
            return actual_scope < target_scope
        
        if len(current_row) == k + 1:
            # Row complete
            rows.append(current_row)
            if _backtrack(row_idx + 1, [0], current_max_val):
                return True
            rows.pop()
            return False
        
        last_val = current_row[-1]
        
        # Upper bound for next element x
        # x must be > last_val
        # x must be < target_scope
        upper_bound = target_scope - 1
        
        # Generate candidates
        # Heuristic: prefer candidates that use "large" differences (low scarcity)
        # and keep x small.
        candidates = []
        
        # Limit search space to keep it fast
        # We can't search all x up to upper_bound if it's large, but for (7,5) it's small enough
        # if we prune well. Let's iterate and collect a few good ones.
        
        # To speed up, we can limit the number of candidates we consider at each step.
        # But we need to be careful not to miss the only solution.
        # Let's try a window of candidates.
        
        for x in range(last_val + 1, upper_bound + 1):
            new_diffs = {x - a for a in current_row}
            
            # Check conflict with used_diffs
            if new_diffs & used_diffs:
                continue
            
            # Check conflict within current_row? No, new_diffs are distinct because a are distinct.
            # But we need to ensure new_diffs don't conflict with each other?
            # Wait, "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows".
            # This means the set of differences for a single row must be a set (no duplicates).
            # Since a_j are distinct, a_j - a_i are distinct for fixed i? No.
            # Example: row [0, 2, 4]. Diffs: 2-0=2, 4-0=4, 4-2=2. Duplicate 2!
            # So a row itself must have distinct internal differences.
            # My previous _row_diffs used a set, which implicitly handled this, but I need to check if new_diffs introduces duplicates within the row.
            
            # Let's check if new_diffs has duplicates?
            # Actually, if I just add x, the new differences are {x - a for a in current_row}.
            # These are distinct from each other because a are distinct.
            # But they might conflict with EXISTING differences in current_row.
            # Example: current_row = [0, 2]. Existing diff: {2}.
            # Add x=4. New diffs: {4-0=4, 4-2=2}.
            # 2 is already in existing diffs. So this is invalid.
            
            existing_diffs = _row_diffs(current_row)
            if new_diffs & existing_diffs:
                continue
            
            # If we get here, x is a valid candidate.
            # Cost heuristic: sum(1/d) for d in new_diffs.
            cost = sum(1.0 / d for d in new_diffs)
            
            # Add small random noise to break ties
            noise = rng.random() * 0.1
            candidates.append((cost + noise, x, new_diffs))
            
            # Limit candidates to avoid explosion
            # If we have found a few good ones, stop?
            # No, we need to explore. But for speed, let's limit to top N by cost?
            # Or just break after a certain number?
            # Let's collect all valid ones in the range, but if the range is large, this is slow.
            # For (7,5) and scope ~100, the range is small.
            # However, backtracking depth is 7 rows * 5 steps = 35 steps.
            # If branching factor is high, it explodes.
            # Let's limit the number of candidates we keep per node to, say, 20 best.
            
            if len(candidates) > 50:
                # Sort and keep top 20
                candidates.sort(key=lambda c: c[0])
                candidates = candidates[:20]
                break
        
        if not candidates:
            return False
        
        # Sort candidates by cost
        candidates.sort(key=lambda c: c[0])
        
        # Randomize top few to allow exploration
        top_n = min(5, len(candidates))
        top_candidates = candidates[:top_n]
        rng.shuffle(top_candidates)
        remaining_candidates = candidates[top_n:]
        
        for cost, x, new_diffs in top_candidates + remaining_candidates:
            # Try this candidate
            used_diffs.update(new_diffs)
            current_row.append(x)
            
            if _backtrack(row_idx, current_row, max(current_max_val, x)):
                return True
            
            # Backtrack
            current_row.pop()
            used_diffs -= new_diffs
            
        return False
    
    success = _backtrack(0, [0], 0)
    if success:
        return rows, max(r[-1] for r in rows)
    return None, None


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are globally distinct.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN
    rng = random.Random(42)

    # Start with the deterministic greedy
    best_rows = _greedy_construct(n_rows, k, rng)
    best_scope = _scope(best_rows)

    # Strategy:
    # 1. Try to find a solution with scope < best_scope using backtracking.
    # 2. If found, update best_scope.
    # 3. Repeat with tighter targets or different random seeds.
    
    # Initial target: best_scope (we want to find something < best_scope)
    # But _solve_attempt checks for < target_scope.
    # So if best_scope is 112, we try target_scope = 112.
    # If we find a solution with max < 112, we update best_scope.
    
    # Let's try a range of target scopes.
    # We can start from best_scope and go down.
    # But backtracking might be slow if target is too tight.
    
    # Let's use a loop that tries to improve.
    # We'll keep trying with different random seeds and potentially tightening the target.
    
    # To avoid getting stuck in a local minimum of the search (i.e. failing to find a solution
    # for a specific target due to bad luck), we should try multiple seeds for the same target.
    
    # Also, we can start with a slightly looser target to find a valid solution quickly,
    # then tighten. But greedy already gave us a valid one.
    
    # Let's try to find a solution with scope < best_scope.
    # We'll iterate target_scope from best_scope down to some minimum (e.g. 100).
    # For each target_scope, we try a few random seeds.
    
    # However, if we fail for target_scope = best_scope, it doesn't mean no solution exists < best_scope.
    # It just means our search didn't find one.
    
    # A better approach:
    # Use simulated annealing or random restarts of the backtracker.
    
    # Let's implement a simple loop:
    # while time < deadline:
    #   target = best_scope
    #   for seed in range(10):
    #     rows, scope = _solve_attempt(n_rows, k, target, rng, time_limit, start_time)
    #     if rows:
    #       best_scope = scope
    #       break
    #   if best_scope == 105: # counting bound
    #     break
    
    # But _solve_attempt has its own time limit check via time_up().
    # We need to pass the global deadline.
    
    # Let's refine _solve_attempt to accept a deadline.
    # Actually, it already uses start_time and time_limit.
    # We can just call it repeatedly.
    
    # To make it more robust, let's try decreasing target_scope.
    # Start with best_scope. If we find a solution, best_scope drops.
    # Then we try the new best_scope.
    
    # We also need to handle the case where _solve_attempt fails due to time limit.
    # It returns (None, None).
    
    # Let's add a counter to avoid infinite loops if no improvement is made.
    attempts = 0
    last_improvement = start_time
    
    while time.time() < deadline:
        attempts += 1
        
        # Try to find a solution with scope < best_scope
        # Use a random seed for each attempt
        seed = attempts * 12345 + 67890
        rng2 = random.Random(seed)
        
        # Give each attempt a short time limit, e.g. 5 seconds
        # But we need to check global deadline too.
        # _solve_attempt checks time.time() - start_time > time_limit.
        # If we pass time_limit = deadline - start_time, it will respect the global deadline.
        # But if it takes too long, it will return None.
        
        # Let's pass a slightly shorter time limit to allow multiple attempts.
        # e.g. 2 seconds per attempt?
        # If best_scope is large, search space is large.
        
        # Let's try with a time limit of 3 seconds per attempt.
        attempt_time_limit = 3.0
        
        rows, scope = _solve_attempt(n_rows, k, best_scope, rng2, attempt_time_limit, start_time)
        
        if rows is not None:
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                last_improvement = time.time()
                # If we hit the counting bound, we can stop
                if best_scope <= 105:
                    break
        
        # If we haven't improved in a while, we might try a different strategy
        # or just keep trying with different seeds.
        
        # Also, occasionally try to construct with a specific target smaller than best_scope
        # to see if we can jump.
        # e.g. try target = best_scope - 5
        if attempts % 10 == 0:
            target = best_scope - 5
            if target > 105:
                rows, scope = _solve_attempt(n_rows, k, target, rng2, attempt_time_limit, start_time)
                if rows is not None and scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                    last_improvement = time.time()

    # Final validation
    if not _validate(best_rows, k):
        # Fallback to simple greedy
        best_rows = _greedy_construct(n_rows, k, rng)

    return best_rows
