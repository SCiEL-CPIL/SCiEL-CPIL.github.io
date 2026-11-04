import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    # Get an initial upper bound using a fast greedy method
    # This provides a tight UB for the binary search
    init_rows = _simple_greedy(n_rows, k)
    if not init_rows:
        return []
    current_best_scope = max(max(r) for r in init_rows)
    current_best_rows = init_rows

    # Binary search for the minimal scope M
    # We search in [lower_bound, current_best_scope]
    lo = lower_bound
    hi = current_best_scope
    
    # If the range is small, just check
    # We will try to find the smallest M such that a valid solution exists.
    # We use a randomized CSP solver for the check.
    
    # To avoid getting stuck in a bad binary search path, we can do a few
    # randomized restarts for the "check" function.
    
    best_found_scope = current_best_scope
    best_found_rows = current_best_rows

    # We iterate from lo to hi? No, binary search.
    # But CSP check is expensive.
    # Let's try a linear scan from lower_bound upwards if the gap is small,
    # otherwise binary search.
    
    # Actually, the gap might be large.
    # Let's use binary search.
    
    found = False
    while lo <= hi and time.time() < deadline - 10.0:
        mid = (lo + hi) // 2
        rows, success = _csp_solve(mid, n_rows, k, deadline - 10.0)
        
        if success:
            # We found a solution with max <= mid
            # Update best if this scope is smaller
            scope = max(max(r) for r in rows)
            if scope < best_found_scope:
                best_found_scope = scope
                best_found_rows = rows
            hi = mid - 1
            found = True
        else:
            lo = mid + 1

    # If binary search didn't find anything better, or if we want to polish the best found
    if best_found_rows:
        # Polish the best found rows to see if we can reduce scope further
        # The CSP solver might have found a solution with max < M, 
        # but maybe we can squeeze it more?
        # Actually, the CSP solver finds *a* solution. 
        # The scope is the max element.
        # If we found a solution with max=100, we know 100 is achievable.
        # The binary search will try to find if 99 is achievable.
        
        # Let's run a local search on the best found rows to try to reduce scope
        # This can sometimes find a solution with max < current_best_scope 
        # even if the binary search didn't explicitly check that exact number 
        # or if the CSP solver failed due to randomness/timeout.
        
        polished_rows, polished_scope = _polish(best_found_rows, k, deadline - 5.0)
        if polished_scope < best_found_scope:
            best_found_scope = polished_scope
            best_found_rows = polished_rows

    if best_found_rows:
        return best_found_rows
    else:
        return current_best_rows


def _csp_solve(M, n_rows, k, deadline):
    """
    Attempts to construct n_rows of k+1 integers starting at 0, increasing,
    such that all differences are distinct and all elements <= M.
    Returns (rows, success).
    """
    rng = random.Random()
    
    # Try a few random restarts if the first one fails quickly
    # But CSP with backtracking is deterministic if order is fixed.
    # We randomize the candidate order.
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None, False
            
        rows, success = _csp_backtrack(M, n_rows, k, deadline, rng)
        if success:
            return rows, True
            
    return None, False


def _csp_backtrack(M, n_rows, k, deadline, rng):
    used_diffs = bytearray(M + 1)
    rows = []
    
    # Heuristic: We build rows one by one.
    # For each row, we build it element by element.
    
    def dfs(row_idx, current_row):
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            # Row is complete
            rows.append(current_row)
            
            # Mark differences for this row
            new_diffs = []
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    d = current_row[j] - current_row[i]
                    if used_diffs[d]:
                        # Should not happen if we checked before adding
                        # But if we are backtracking, we need to be careful
                        # Actually, we check before recursing, so this is safe
                        pass
                    used_diffs[d] = 1
                    new_diffs.append(d)
            
            if dfs(row_idx + 1, [0]):
                return True
            
            # Backtrack: unmark differences
            for d in new_diffs:
                used_diffs[d] = 0
            rows.pop()
            return False
        
        # Determine next element
        last_val = current_row[-1]
        start = last_val + 1
        end = M
        
        candidates = []
        for x in range(start, end + 1):
            # Check if adding x creates any conflict
            conflict = False
            new_diffs = []
            for a in current_row:
                d = x - a
                if d > M:
                    conflict = True
                    break
                if used_diffs[d]:
                    conflict = True
                    break
                new_diffs.append(d)
                
            if not conflict:
                # Heuristic score: 
                # We prefer x that leaves more "small" differences unused?
                # Or x that is smaller?
                # Let's use a score based on how many of these new_diffs are "small"
                # Small differences are more constrained.
                # So we want to avoid using small differences if possible?
                # No, we MUST use them.
                # Actually, small differences are more likely to conflict with future rows
                # because future rows will also have small differences.
                # So we should prefer x that uses LARGE differences?
                # Wait, if we use large differences now, we leave small ones for later.
                # But later rows also need small differences.
                # It's a trade-off.
                
                # Let's try: Score = sum(1/d for d in new_diffs)
                # Higher score = uses more small diffs = worse?
                # So we want to minimize score?
                # Or maximize?
                # If we use small diffs now, we block them for later.
                # Later rows are "harder" because they have fewer options?
                # Actually, all rows have the same structure.
                
                # Let's just randomize with a bias towards smaller x (to keep M small)
                # But we are already bounded by M.
                
                score = rng.random()
                # Add a penalty for using small differences
                # This encourages using larger differences first, leaving small ones for "later"
                # which might be a mistake, but let's try.
                # Alternatively, prefer x that is "central"?
                
                # Let's try sorting by x ascending, but with random jitter
                candidates.append((x, score))
        
        if not candidates:
            return False
            
        # Sort candidates
        # We want to try "promising" candidates first.
        # Promising: Smaller x (keeps scope tight) but avoids "bad" diffs.
        # Let's sort by x, but randomize the order slightly to allow backtracking to explore different branches.
        # Actually, for CSP, a good heuristic is MRV (Minimum Remaining Values).
        # Here, the "variable" is the next element.
        # The "domain" is the candidates.
        # We can estimate the impact of each candidate on the *next* step.
        
        # For simplicity, let's sort by x ascending, but pick a random candidate from the top K?
        # Or just sort by a heuristic score.
        
        # Heuristic: Prefer x that minimizes the number of *blocked* future candidates?
        # Too expensive.
        
        # Let's use: Score = x + rng.random() * 10
        # This biases towards smaller x but allows exploration.
        
        candidates.sort(key=lambda t: t[0] + t[1] * 5)
        
        for x, _ in candidates:
            # Recurse
            # We need to simulate the effect of x on used_diffs
            # We already computed new_diffs in the candidate generation loop, 
            # but we discarded them. Let's recompute or store them.
            
            # To save time, we stored candidates as (x, score).
            # We need new_diffs.
            new_diffs = []
            for a in current_row:
                new_diffs.append(x - a)
                
            # Check again (safety)
            valid = True
            for d in new_diffs:
                if used_diffs[d]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Apply
            for d in new_diffs:
                used_diffs[d] = 1
                
            if dfs(row_idx, current_row + [x]):
                return True
                
            # Undo
            for d in new_diffs:
                used_diffs[d] = 0
                
        return False

    # Start DFS
    if dfs(0, [0]):
        return rows, True
    return None, False


def _polish(rows, k, deadline):
    """Local search: repeatedly try to move a single element to a smaller
    value that keeps the whole set valid, accepting only moves that reduce
    the global max."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)

    def _valid_diffs(rows):
        all_d = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in all_d:
                        return None
                    all_d.add(d)
        return all_d

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find the row holding the current max.
        max_row_idx = max(range(len(best)), key=lambda i: max(best[i]))
        row = best[max_row_idx]
        # Try to move elements from the back (largest positions) to smaller values.
        for pos in range(len(row) - 1, 0, -1):
            if time.time() > deadline:
                return best, best_scope
            old_val = row[pos]
            for new_val in range(row[pos - 1] + 1, old_val):
                row[pos] = new_val
                # Quick validity check.
                all_d = set()
                ok = True
                for r in best:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d in all_d:
                                ok = False
                                break
                            all_d.add(d)
                    if not ok:
                        break
                if ok:
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
                row[pos] = old_val
            if improved:
                break

    return best, best_scope


def _simple_greedy(n_rows, k):
    """Deterministic smallest-first greedy (fallback)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows
