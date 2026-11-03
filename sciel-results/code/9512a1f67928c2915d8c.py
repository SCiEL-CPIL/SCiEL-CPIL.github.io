import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    k_plus_1 = k + 1

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

    # 3. Polish pass for further compaction
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. CSP Backtracking Search with Symmetry Breaking
    # We search for a solution with scope < best_scope
    # We try scopes from lower_bound up to best_scope - 1
    # But we start from best_scope - 1 and go down? 
    # Actually, finding a solution for a specific M is hard.
    # Let's use the backtracker to find ANY solution for M < best_scope.
    
    # We will iterate over possible scopes M from lower_bound to best_scope - 1
    # But this is too slow. 
    # Instead, we use the backtracker to minimize M directly.
    
    # The backtracker below will try to find a solution with max element <= target_scope.
    # We will try to find a solution with target_scope = best_scope - 1.
    # If found, we update best_scope and repeat.
    
    attempt = 0
    while time.time() < deadline - 2.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        # Try multiple random restarts for this target scope
        found = False
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break

            solution = _backtracking_search_csp(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further with local search
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    polished = _polish_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in polished)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found:
            # If we can't find a solution for best_scope - 1, 
            # it's likely that best_scope is optimal or close to it.
            # We can stop or try a few more restarts.
            # Given the time limit, we stop if we've tried enough.
            if target_scope <= lower_bound + 10:
                break

    # Final polish
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

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


def _backtracking_search_csp(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    CSP Backtracking Search with Symmetry Breaking.
    Variables: a[r][j] for r in 0..n_rows-1, j in 1..k.
    Constraints: 
    1. a[r][0] < a[r][1] < ... < a[r][k] <= max_scope
    2. All differences a[r][j] - a[r][i] are distinct globally.
    3. Symmetry: a[0][1] <= a[1][1] <= ... <= a[n_rows-1][1]
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    used_mask = 0  # Bitmask for used differences
    
    # We build rows sequentially: row 0, then row 1, etc.
    # Within a row, we build elements sequentially: a[0], a[1], ..., a[k].
    
    # To enforce symmetry, we restrict the first non-zero element of each row.
    # a[0][1] can be anything valid.
    # a[1][1] must be >= a[0][1].
    # a[2][1] must be >= a[1][1].
    # a[3][1] must be >= a[2][1].
    
    def is_valid(row_idx, pos, val, mask):
        # Check if val creates any conflict with existing elements in this row
        # and global used_mask
        row = rows[row_idx]
        for i in range(pos):
            d = val - row[i]
            if d > max_scope:
                return False
            if (mask >> d) & 1:
                return False
        return True

    def get_candidates(row_idx, pos, mask, min_val):
        candidates = []
        # Max possible value is max_scope - (k - pos)
        # because we need k-pos more elements, each at least 1 greater.
        max_val = max_scope - (k - pos)
        if max_val < min_val:
            return []
        
        # We want to try smaller values first to keep scope small, 
        # but also randomize to escape local minima.
        # We'll generate candidates in range [min_val, max_val]
        # and filter them.
        
        # Optimization: Instead of iterating all, we can sample.
        # But for k=7, the range is small enough to iterate if max_scope is tight.
        # max_scope is around 130.
        
        for x in range(min_val, max_val + 1):
            if is_valid(row_idx, pos, x, mask):
                candidates.append(x)
        
        if not candidates:
            return []
        
        # Heuristic: Bias towards smaller values
        # Sort by value, then shuffle the first N to add randomness
        candidates.sort()
        
        # If there are too many candidates, limit the search breadth
        # to prevent explosion.
        limit = 20
        if len(candidates) > limit:
            # Take the smallest 'limit' candidates
            # But also pick a few random ones from the rest?
            # For now, just take the smallest.
            candidates = candidates[:limit]
        
        rng.shuffle(candidates)
        return candidates

    def backtrack(row_idx, pos, mask):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if pos == k + 1:
            return backtrack(row_idx + 1, 0, mask)
        
        # Determine min_val for symmetry breaking
        min_val = 1
        if row_idx > 0 and pos == 1:
            min_val = rows[row_idx - 1][1]
        
        # Also, min_val must be > rows[row_idx][pos-1]
        if pos > 0:
            min_val = max(min_val, rows[row_idx][pos - 1] + 1)
        
        candidates = get_candidates(row_idx, pos, mask, min_val)
        
        for x in candidates:
            # Apply change
            rows[row_idx][pos] = x
            
            # Calculate new mask
            new_mask = mask
            for i in range(pos):
                d = x - rows[row_idx][i]
                new_mask |= (1 << d)
            
            if backtrack(row_idx, pos + 1, new_mask):
                return True
            
            # Backtrack
            # No need to explicitly unmask if we pass mask by value
            # which we do.
        
        return False

    # Start search
    if backtrack(0, 1, 0):
        return [row[:] for row in rows]
    
    return None


def _improve_scope(solution, n_rows, k, deadline):
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                if original < best_scope and pos != k_plus_1 - 1:
                    continue

                for new_val in range(row[pos - 1] + 1, original):
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    if any(d <= 0 for d in new_diffs):
                        continue

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


def _polish_scope(solution, n_rows, k, deadline):
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    for iteration in range(20):
        if time.time() > deadline - 2.0:
            break

        any_improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                for new_val in range(original - 1, row[pos - 1], -1):
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    if any(d <= 0 for d in new_diffs):
                        continue

                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_improved = True
                    break

        if not any_improved:
            break

    return best
