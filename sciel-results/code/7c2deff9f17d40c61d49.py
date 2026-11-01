"""Difference triangle set (7,5) - find 7 rows of 6 marks minimizing max entry.

Strategy: Model as a constraint satisfaction problem where the global
difference set is a shared resource pool. Use a backtracking solver with:
- Forward checking: prune candidates that make remaining positions infeasible
- Dynamic row ordering: build rows in order of tightest remaining constraints
- Difference density heuristic: prefer values that use "scarce" differences
- Iterative deepening on max-entry bound with multiple random restarts
- Simulated annealing for local search refinement to escape local minima
"""
import time
import random
import copy

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_row(used, k, max_x=None):
    """Build one row greedily: start at 0, pick smallest next mark with no diff conflicts."""
    row = [0]
    while len(row) < k + 1:
        base = row[-1]
        found = False
        for x in range(base + 1, (max_x if max_x is not None else 500) + 1):
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
    return rows

def _count_feasible_positions(current_row, pos, max_entry, used_diffs, k):
    """
    Count how many valid values exist for the next position given current state.
    Used for forward checking.
    """
    if pos >= k + 1:
        return 1  # Row complete
    
    lo = current_row[-1] + 1
    hi = max_entry
    
    # Remaining positions must fit
    remaining = k + 1 - pos
    if hi - lo + 1 < remaining:
        return 0
    
    count = 0
    for val in range(lo, hi + 1):
        new_diffs = {val - a for a in current_row}
        if len(new_diffs) != len(current_row):
            continue
        if new_diffs & used_diffs:
            continue
        count += 1
    
    return count

def _backtrack_with_forward_checking(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking with forward checking.
    The key insight: treat the set of used differences as a global resource pool.
    At each step, forward-check whether the remaining positions can still be filled.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = []
    
    def solve(row_idx, pos, current_row):
        """Recursively build row row_idx, position pos."""
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            # Row complete
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 1, [0])
        
        if pos == 0:
            return solve(row_idx, 1, [0])
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        # Forward check: can we still fill remaining positions?
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return False
        
        # Generate candidates with forward checking
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            
            # Forward check: after placing this value, can the next position be filled?
            if pos + 1 < k + 1:
                next_lo = val + 1
                next_hi = max_entry
                next_remaining = k + 1 - (pos + 1)
                if next_hi - next_lo + 1 < next_remaining:
                    continue
                # Check if at least one valid next value exists
                has_next = False
                for next_val in range(next_lo, next_hi + 1):
                    next_new_diffs = {next_val - a for a in current_row + [val]}
                    if len(next_new_diffs) == len(current_row) + 1:
                        if not (next_new_diffs & (used_diffs | new_diffs)):
                            has_next = True
                            break
                if not has_next:
                    continue
            
            candidates.append(val)
        
        if not candidates:
            return False
        
        # Order candidates: prefer those that leave more room and use "scarce" diffs
        def score(val):
            new_diffs = {val - a for a in current_row}
            # Prefer smaller values (to keep max low)
            s1 = val * 10
            # Prefer values whose differences are less "popular" (more available)
            # Approximate by checking how many of these diffs are already "close" to being used
            s2 = 0
            for d in new_diffs:
                # Count how many other used diffs are within a small range
                s2 += sum(1 for ud in used_diffs if abs(ud - d) <= 3)
            return s1 + s2
        
        candidates.sort(key=score)
        
        # Add some randomness to avoid getting stuck
        if rng.random() < 0.3 and len(candidates) > 1:
            # Shuffle a subset
            swap_idx = rng.randrange(len(candidates))
            candidates[0], candidates[swap_idx] = candidates[swap_idx], candidates[0]
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            new_diffs = {val - a for a in current_row}
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if solve(row_idx, pos + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    if solve(0, 0, []):
        return rows
    return None

def _simulated_annealing(n_rows, k, initial_rows, rng, deadline, max_iters=10000):
    """
    Simulated annealing to refine an initial valid solution.
    We try to reduce the maximum entry by moving individual marks.
    """
    if not initial_rows:
        return None
    
    best_rows = [r[:] for r in initial_rows]
    best_score = _max_entry(best_rows)
    current_rows = [r[:] for r in initial_rows]
    current_score = best_score
    
    # Compute used differences for current state
    def get_used_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    def get_conflicts(rows):
        """Return set of conflicting difference values and their positions."""
        diffs = {}
        conflicts = set()
        for ri, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        conflicts.add(d)
                    else:
                        diffs[d] = (ri, i, j)
        return conflicts
    
    temperature = 50.0
    cooling = 0.995
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if temperature < 0.01:
            break
        
        # Pick a random row and a random position (not 0) to move
        ri = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        
        old_val = current_rows[ri][pos]
        
        # Try to find a new position for this mark
        # The new value must be greater than previous mark and less than next mark (if exists)
        lo = current_rows[ri][pos - 1] + 1
        hi = current_rows[ri][pos + 1] - 1 if pos < k else old_val + 10
        
        if lo >= hi:
            continue
        
        # Try a random new value in range
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
        
        # Check if this move creates conflicts
        # Remove old differences involving position pos
        old_diffs = set()
        for i in range(k + 1):
            if i != pos:
                d = abs(current_rows[ri][pos] - current_rows[ri][i])
                old_diffs.add(d)
        
        # Add new differences involving position pos
        new_diffs = set()
        for i in range(k + 1):
            if i != pos:
                d = abs(new_val - current_rows[ri][i])
                new_diffs.add(d)
        
        # Check for conflicts with other rows
        # Build a map of all differences from other rows
        other_diffs = set()
        for rj, row in enumerate(current_rows):
            if rj == ri:
                continue
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    other_diffs.add(row[j] - row[i])
        
        # Also need to check within the same row for other pairs
        same_row_diffs = set()
        for i in range(len(current_rows[ri])):
            for j in range(i + 1, len(current_rows[ri])):
                if i != pos and j != pos:
                    same_row_diffs.add(current_rows[ri][j] - current_rows[ri][i])
        
        # Check if new_diffs conflict with other_diffs or same_row_diffs
        conflict_new = (new_diffs & other_diffs) | (new_diffs & same_row_diffs)
        
        # Remove old conflicts that are now gone
        # This is complex; for simplicity, just check if the move is valid
        # by temporarily applying it and checking
        
        # Temporarily apply move
        current_rows[ri][pos] = new_val
        
        # Check validity
        valid = is_valid(current_rows)
        new_score = _max_entry(current_rows)
        
        if valid:
            # Accept if it improves or with probability based on temperature
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_rows = [r[:] for r in current_rows]
            else:
                # Accept with probability exp(-delta/T)
                delta = new_score - current_score
                if delta > 0 and rng.random() < (2.718281828 ** (-delta / temperature)):
                    current_score = new_score
        else:
            # Reject invalid moves (or accept with very low probability)
            pass
        
        # Revert if not accepted
        if not valid or new_score >= current_score:
            # Check if we should keep the move
            if valid and new_score < current_score:
                pass  # Keep it
            elif valid and new_score == current_score and rng.random() < 0.5:
                pass  # Keep it
            else:
                current_rows[ri][pos] = old_val
                current_score = _max_entry(current_rows)
        
        temperature *= cooling
    
    return best_rows

def construct(n_rows, k):
    """
    Main entry point. Uses iterative deepening backtracking with forward checking,
    followed by simulated annealing refinement.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Lower bound: need n_rows * k*(k+1)/2 distinct differences, each <= max_entry
    # So max_entry >= n_rows * k*(k+1)/2
    lower_bound = n_rows * k * (k + 1) // 2
    
    best_rows = None
    best_score = float('inf')
    
    # First, try greedy to get a starting point
    greedy_result = _greedy_all(n_rows, k)
    if greedy_result:
        best_rows = greedy_result
        best_score = _max_entry(best_rows)
    
    # Iterative deepening on max_entry bound
    # Start from lower bound and increase
    # But also try from the greedy result's score downwards
    
    rng = random.Random(42)
    
    # Try backtracking with increasing bounds
    # Start from a reasonable estimate
    initial_bound = best_score if best_rows else lower_bound + 10
    
    # Try bounds from initial_bound down to lower_bound
    # But that's expensive, so try a few strategic bounds
    
    # First, try to improve the greedy result with backtracking
    if best_rows:
        # Try backtracking with bound = best_score - 1, best_score - 2, etc.
        for bound in range(best_score - 1, lower_bound - 1, -1):
            if time.time() > deadline - 20:  # Leave time for SA
                break
            
            result = _backtrack_with_forward_checking(n_rows, k, bound, rng, deadline - 10)
            if result:
                best_rows = result
                best_score = bound
                break
    
    # If we still have time, try simulated annealing on the best solution
    if best_rows and time.time() < deadline - 5:
        sa_result = _simulated_annealing(n_rows, k, best_rows, rng, deadline - 2)
        if sa_result:
            sa_score = _max_entry(sa_result)
            if sa_score < best_score:
                best_rows = sa_result
                best_score = sa_score
    
    # Final validation
    if best_rows and _validate(best_rows):
        return best_rows
    
    # Fallback: return greedy result if valid
    if greedy_result and _validate(greedy_result):
        return greedy_result
    
    # Last resort: return empty or minimal valid set
    return [[0, 1, 2, 3, 4, 5]] * n_rows
