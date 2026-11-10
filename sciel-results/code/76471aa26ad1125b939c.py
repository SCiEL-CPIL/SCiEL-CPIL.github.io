"""Difference triangle set (4,7) - backtracking CSP with forward checking."""
import time
import random
import sys
import math

_DEADLINE = 150.0
_SAFETY = 10.0
_TIME_LIMIT = _DEADLINE - _SAFETY


def _row_diffs(row):
    """All positive differences within a single row."""
    d = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ri)
    return d


def _is_valid(rows):
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _score(rows):
    return max(max(r) for r in rows)


def _solve_csp(n_rows, k, deadline, best_max, rng):
    """
    Backtracking CSP solver.
    Variables: (row_idx, pos_idx) for pos_idx in 1..k
    Values: integers > previous in row
    Constraints: all differences globally distinct, and max entry <= best_max
    """
    rows = [[0]] * n_rows
    used_diffs = set()
    # Track current max per row
    row_max = [0] * n_rows
    
    # Order of variables: row by row, position by position
    # We'll use recursion with explicit stack for speed
    
    # Precompute: for each row, the positions to fill
    # Variable ordering: row 0 pos 1, row 0 pos 2, ..., row 0 pos k, row 1 pos 1, ...
    
    def backtrack(var_idx):
        """var_idx encodes (row_idx, pos_idx)"""
        if time.time() > deadline:
            return False
        
        ri = var_idx // k
        pi = var_idx % k + 1  # positions 1..k
        
        if ri >= n_rows:
            # All rows complete
            return True
        
        row = rows[ri]
        # Current row has elements at positions 0..pi-1
        # We need to choose row[pi] > row[pi-1]
        
        lo = row[pi - 1] + 1
        # Upper bound: we want to keep max <= best_max
        # If this is the last position in this row, it's the max
        # Actually, later positions will be larger, so row[pi] <= best_max - (k - pi) * 1
        # More precisely: row[pi] + (k - pi) <= best_max (minimum increment of 1 each)
        # So row[pi] <= best_max - (k - pi)
        hi = best_max - (k - pi)
        
        if hi < lo:
            return False
        
        # Try values in increasing order (greedy: smallest first)
        # But also try some larger values to explore
        # For efficiency, try in order: lo, lo+1, ..., hi
        # With some randomization for diversity
        
        # Generate candidate values
        candidates = list(range(lo, hi + 1))
        # Shuffle a bit for diversity but keep bias toward small
        if len(candidates) > 50:
            # Keep first 20 in order, shuffle rest
            first = candidates[:20]
            rest = candidates[20:]
            rng.shuffle(rest)
            candidates = first + rest
        
        for nv in candidates:
            # Check if adding nv creates valid differences
            # New differences: nv - row[i] for i in 0..pi-1
            new_diffs = set()
            valid = True
            for i in range(pi):
                d = nv - row[i]
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Also need to ensure that future positions can be filled
            # Forward check: the next position (pi+1) must have some value > nv
            # that doesn't conflict. This is expensive, so we skip for now
            # and rely on backtracking.
            
            # Apply
            row.append(nv)
            used_diffs.update(new_diffs)
            old_max = row_max[ri]
            row_max[ri] = max(row_max[ri], nv)
            
            if backtrack(var_idx + 1):
                return True
            
            # Undo
            row.pop()
            used_diffs.difference_update(new_diffs)
            row_max[ri] = old_max
        
        return False
    
    # Initialize
    for i in range(n_rows):
        rows[i] = [0]
    
    return backtrack(0)


def _solve_with_restarts(n_rows, k, deadline, initial_best):
    """Try multiple restarts with different random seeds and strategies."""
    best_rows = None
    best_max = initial_best
    rng = random.Random()
    
    # Strategy 1: Pure backtracking with decreasing best_max
    # Start from initial_best and try to find a solution with smaller max
    
    # First, try to find any valid solution close to the bound
    # The counting bound is 112, best known is 134
    
    # Try to find solution with max <= 134, then 133, etc.
    # But that's too slow. Instead, use the initial solution and try to improve.
    
    # Alternative: use backtracking with a tighter upper bound
    # Start with best_max = initial_best - 1 and try to find a solution
    
    # For efficiency, we'll do a few full searches with different upper bounds
    # and different random orderings
    
    search_deadline = deadline - 5  # leave 5 sec for final processing
    
    # Try several upper bounds
    # Start from initial_best and go down
    # But we don't want to spend too long on each
    
    # Strategy: binary search on the upper bound
    # Low: 112 (counting bound), High: initial_best
    
    # Actually, let's just do a few restarts with the current best_max
    # and hope to find something better through randomization
    
    # Better strategy: use the backtracking solver with a time limit
    # and try to find a solution with max < initial_best
    
    target = initial_best - 1
    while target >= 112 and time.time() < search_deadline:
        # Try to find a solution with max <= target
        # Use backtracking with randomization
        found = False
        attempts = 0
        attempt_deadline = min(time.time() + 2.0, search_deadline)  # 2 sec per attempt
        
        while time.time() < attempt_deadline and not found:
            rng = random.Random(attempts * 7919 + 42)
            # Reset
            # We need to re-run the backtracking
            # This is expensive, so let's be smart
            
            # Actually, let's use a different approach:
            # Greedy construction with backtracking on conflicts
            
            found = _greedy_with_backtrack(n_rows, k, target, rng, attempt_deadline)
            if found:
                # Extract the solution
                # _greedy_with_backtrack modifies global state? No, let's make it return
                pass
            attempts += 1
        
        if found:
            best_max = target
            # Store the solution
            break
        else:
            target -= 1
    
    return best_rows, best_max


def _greedy_with_backtrack(n_rows, k, max_bound, rng, deadline):
    """
    Greedy construction with limited backtracking.
    Build rows one at a time, positions left to right.
    At each step, try values in increasing order.
    If stuck, backtrack a few positions.
    """
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    # Stack of (row_idx, pos_idx, value)
    # We'll use a simpler approach: build row by row
    
    def build_row(row_idx, used):
        """Build one row with backtracking."""
        row = [0]
        
        def bt(pos):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return list(row)
            
            lo = row[pos - 1] + 1
            hi = max_bound  # last element can be at most max_bound
            # Actually, for intermediate positions, they can be up to max_bound
            # but we want to keep room for future elements
            
            # For position pos (0-indexed), we need (k+1 - pos) more elements after
            # Each at least 1 more, so row[pos] <= max_bound - (k + 1 - pos)
            # Wait, pos is 1-indexed here. Let me reclarify.
            # row has k+1 elements: row[0]=0, row[1], ..., row[k]
            # We're choosing row[pos] for pos in 1..k
            # After choosing row[pos], we need k-pos more elements, each >= row[pos]+1
            # So row[pos] + (k-pos) <= max_bound
            # row[pos] <= max_bound - (k - pos)
            
            hi = max_bound - (k - pos)
            
            if hi < lo:
                return None
            
            # Try values in increasing order
            for nv in range(lo, hi + 1):
                new_diffs = set()
                valid = True
                for i in range(pos):
                    d = nv - row[i]
                    if d in used or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                row.append(nv)
                used.update(new_diffs)
                
                result = bt(pos + 1)
                if result is not None:
                    return result
                
                # Undo
                row.pop()
                used.difference_update(new_diffs)
            
            return None
        
        return bt(1)
    
    # Build all rows
    for ri in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = build_row(ri, used_diffs)
        if row is None:
            return None
        rows[ri] = row
    
    # Verify
    if not _is_valid(rows):
        return None
    
    if _score(rows) > max_bound:
        return None
    
    return rows


def _construct_greedy(n_rows, k):
    """Fast greedy initialization."""
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


def _local_search_refine(rows, k, n_rows, deadline):
    """Quick local search to improve the solution."""
    used = set()
    for r in rows:
        used.update(_row_diffs(r))
    
    best = _score(rows)
    best_rows = [list(r) for r in rows]
    
    rng = random.Random(999)
    T = 5.0
    T_min = 0.1
    cool = 0.9999
    
    while time.time() < deadline:
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, len(rows[ri]))
        row = rows[ri]
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi + 1 < len(row) else max(row) + 10
        if hi < lo:
            hi = lo
        nv = rng.randrange(lo, hi + 1)
        if nv == row[pi]:
            continue
        
        old = row[pi]
        affected_old = set()
        for i in range(len(row)):
            if i != pi:
                affected_old.add(abs(old - row[i]))
        affected_new = set()
        for i in range(len(row)):
            if i != pi:
                affected_new.add(abs(nv - row[i]))
        removed = affected_old - affected_new
        added = affected_new - affected_old
        
        ok = True
        for d in added:
            if d in used and d not in removed:
                ok = False
                break
        if not ok:
            continue
        
        old_score = max(max(r) for r in rows)
        if old == max(row):
            new_row_max = max(nv, row[-1])
            others_max = max(max(r) for r in rows if r is not row) if n_rows > 1 else 0
            new_score = max(new_row_max, others_max)
        else:
            new_score = old_score
        
        delta = new_score - old_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            row[pi] = nv
            for d in removed:
                used.discard(d)
            for d in added:
                used.add(d)
            
            if new_score < best:
                best = new_score
                best_rows = [list(r) for r in rows]
        
        T *= cool
        if T < T_min:
            T = T_min
    
    return best_rows, best


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT
    
    # Step 1: Greedy initialization
    rows = _construct_greedy(n_rows, k)
    best = _score(rows)
    best_rows = [list(r) for r in rows]
    
    # Step 2: Try backtracking CSP with decreasing upper bound
    # Start from best-1 and go down
    # Time budget: 60 seconds for backtracking
    
    bt_deadline = t0 + 60
    target = best - 1
    
    while target >= 112 and time.time() < bt_deadline:
        # Try a few random restarts for this target
        found = False
        for attempt in range(5):
            if time.time() >= bt_deadline:
                break
            rng = random.Random(attempt * 104729 + 17)
            attempt_deadline = min(time.time() + 3.0, bt_deadline)
            result = _greedy_with_backtrack(n_rows, k, target, rng, attempt_deadline)
            if result is not None:
                found = True
                best = target
                best_rows = result
                break
        
        if found:
            # Try to go lower
            target -= 1
        else:
            # Can't find with this target, try lower
            target -= 1
    
    # Step 3: Local search refinement with remaining time
    ls_deadline = t0 + _TIME_LIMIT - 5
    if time.time() < ls_deadline:
        ls_rows, ls_best = _local_search_refine([list(r) for r in best_rows], k, n_rows, ls_deadline)
        if ls_best < best:
            best = ls_best
            best_rows = ls_rows
    
    # Final validity check
    if not _is_valid(best_rows):
        best_rows = _construct_greedy(n_rows, k)
    
    return best_rows
