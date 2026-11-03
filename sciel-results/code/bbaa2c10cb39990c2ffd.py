"""Difference triangle set (7,5) - CSP backtracking on difference sets."""
import time
import random
import math

_DEADLINE = 143.0  # safety margin under 150s wall-clock


def _scope(rows):
    m = 0
    for r in rows:
        if r:
            v = r[-1]
            if v > m:
                m = v
    return m


def _validate(rows):
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy(n_rows, k, rng=None):
    """Greedy construction: smallest next mark with no new difference collisions."""
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


def _build_counts(rows):
    counts = {}
    for r in rows:
        n = len(r)
        for i in range(n):
            ai = r[i]
            for j in range(i + 1, n):
                d = r[j] - ai
                counts[d] = counts.get(d, 0) + 1
    return counts


def _row_diffs_list(row):
    out = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ai)
    return out


def _cost(scope, counts):
    # collisions = number of diffs with count >= 2 (each such diff contributes count-1)
    coll = 0
    for c in counts.values():
        if c > 1:
            coll += c - 1
    return scope + 500 * coll


def _sa_optimize(rows, n_rows, k, deadline, rng):
    rows = [list(r) for r in rows]
    counts = _build_counts(rows)
    scope = _scope(rows)
    cost = _cost(scope, counts)

    best_rows = [list(r) for r in rows]
    best_scope = scope
    best_cost = cost

    T = 6.0
    cooling = 0.9997
    min_T = 0.05

    while time.time() < deadline and T > min_T:
        # choose move type
        if rng.random() < 0.55:
            # ---- single-element decrement move (cheap) ----
            r = rng.randrange(n_rows)
            j = rng.randrange(1, k + 1)
            row = rows[r]
            if j >= len(row):
                continue
            old = row[j]
            if old <= row[j - 1] + 1:
                # cannot decrease
                T *= cooling
                continue
            new_val = old - 1
            # compute old and new diffs involving j
            old_diffs = []
            for i in range(j):
                old_diffs.append(old - row[i])
            for i in range(j + 1, len(row)):
                old_diffs.append(row[i] - old)
            new_diffs = []
            for i in range(j):
                new_diffs.append(new_val - row[i])
            for i in range(j + 1, len(row)):
                new_diffs.append(row[i] - new_val)

            # temporarily apply to counts
            for d in old_diffs:
                counts[d] -= 1
                if counts[d] == 0:
                    del counts[d]
            for d in new_diffs:
                counts[d] = counts.get(d, 0) + 1

            new_scope = _scope(rows) if (row[j - 1] < new_val < old and r == 0) else scope
            # recompute scope only if this row holds the max
            if old == scope:
                new_scope = _scope(rows)
            else:
                new_scope = scope
            new_cost = _cost(new_scope, counts)

            if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / T):
                row[j] = new_val
                scope = new_scope
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_scope = scope
                    best_rows = [list(x) for x in rows]
            else:
                # revert
                for d in new_diffs:
                    counts[d] -= 1
                    if counts[d] == 0:
                        del counts[d]
                for d in old_diffs:
                    counts[d] = counts.get(d, 0) + 1
                row[j] = old
        else:
            # ---- row-rebuild move (escape mechanism) ----
            r = rng.randrange(n_rows)
            old_row = list(rows[r])
            old_diffs = _row_diffs_list(old_row)
            # remove old row's diffs from counts
            for d in old_diffs:
                counts[d] -= 1
                if counts[d] == 0:
                    del counts[d]

            # rebuild row greedily, avoiding other rows' diffs (counts>0)
            max_scope = scope + 6
            new_row = [0]
            x = 0
            ok = True
            local = set()
            for _ in range(1, k + 1):
                found = False
                t = x + 1
                while t <= max_scope:
                    newd = [t - a for a in new_row]
                    conflict = False
                    for d in newd:
                        if d in counts:  # already used by another row
                            conflict = True
                            break
                    if not conflict:
                        new_row.append(t)
                        local.update(newd)
                        x = t
                        found = True
                        break
                    t += 1
                if not found:
                    ok = False
                    break
            if not ok:
                # revert: put old row back
                for d in old_diffs:
                    counts[d] = counts.get(d, 0) + 1
                rows[r] = old_row
                T *= cooling
                continue

            # add new row's diffs
            new_diffs = _row_diffs_list(new_row)
            for d in new_diffs:
                counts[d] = counts.get(d, 0) + 1

            new_scope = _scope(rows)
            new_cost = _cost(new_scope, counts)

            if new_cost < cost or rng.random() < math.exp(-(new_cost - cost) / T):
                rows[r] = new_row
                scope = new_scope
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_scope = scope
                    best_rows = [list(x) for x in rows]
            else:
                # revert
                for d in new_diffs:
                    counts[d] -= 1
                    if counts[d] == 0:
                        del counts[d]
                for d in old_diffs:
                    counts[d] = counts.get(d, 0) + 1
                rows[r] = old_row

        T *= cooling

    return best_rows, best_scope


def _final_compress(rows, n_rows, k, deadline, rng):
    """Cheap deterministic pass: repeatedly try to decrement any element by 1 if valid."""
    rows = [list(r) for r in rows]
    counts = _build_counts(rows)
    scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = scope

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(n_rows):
            row = rows[r]
            for j in range(k, 0, -1):
                old = row[j]
                if old <= row[j - 1] + 1:
                    continue
                new_val = old - 1
                old_diffs = []
                for i in range(j):
                    old_diffs.append(old - row[i])
                for i in range(j + 1, len(row)):
                    old_diffs.append(row[i] - old)
                new_diffs = []
                for i in range(j):
                    new_diffs.append(new_val - row[i])
                for i in range(j + 1, len(row)):
                    new_diffs.append(row[i] - new_val)
                # check new diffs don't collide with counts (other rows)
                conflict = False
                for d in new_diffs:
                    # count of d excluding this row's own old occurrence
                    c = counts.get(d, 0)
                    # subtract old occurrences of d in this row
                    own = old_diffs.count(d)
                    if c - own > 0:
                        conflict = True
                        break
                if conflict:
                    continue
                # apply
                for d in old_diffs:
                    counts[d] -= 1
                    if counts[d] == 0:
                        del counts[d]
                for d in new_diffs:
                    counts[d] = counts.get(d, 0) + 1
                row[j] = new_val
                changed = True
                if old == scope:
                    scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(x) for x in rows]
    return best_rows, best_scope


def _csp_backtrack(n_rows, k, deadline, rng):
    """
    CSP backtracking on difference sets.
    Variables: rows.
    Values: valid Sidon sets (rows) with max entry <= current_scope.
    Constraint: all differences across all rows must be distinct.
    Heuristic: Most Constrained Variable (MCV) - pick the row whose current
    difference set has the highest "density" (number of differences used, or
    potential for collision). Actually, since we build rows one by one, MCV
    suggests picking the row that has the fewest available valid difference sets
    given the already used differences.
    """
    # We will build rows one by one.
    # State: used_diffs (set), rows (list)
    # To speed up, we pre-generate a pool of candidate rows?
    # Or generate them on the fly.
    # Given the time limit, on-the-fly with pruning is safer.
    
    used_diffs = set()
    rows = []
    
    # Precompute all possible "shapes" or just generate candidates?
    # Generating candidates for a specific scope is expensive.
    # Let's try to generate candidates for the current scope limit.
    
    # Initial scope limit
    # Greedy scope is ~150. Best known is 112.
    # Let's start with a tight limit, e.g., 115, and increase if stuck.
    # But we need to be fast.
    
    # Alternative: Use the current SA solution's scope as the upper bound.
    # The SA solution is usually close to optimal.
    
    # Let's assume we have a good initial solution from SA.
    # The CSP search tries to find a solution with scope <= current_best_scope.
    
    # Since we don't have the SA solution here in this function signature,
    # we should probably call SA first, then CSP with SA's scope as limit.
    # Or, just run CSP with a fixed limit like 115?
    # If we fail, we fall back to SA.
    
    # Let's make CSP take a max_scope parameter.
    
    pass # Placeholder, logic below

def _csp_solve(n_rows, k, max_scope, deadline, rng):
    """
    Attempt to find a valid set of n_rows rows with max entry <= max_scope.
    Returns list of rows or None.
    """
    used_diffs = set()
    rows = []
    
    # Candidates for each row index?
    # MCV heuristic: We pick the row index that has the fewest valid candidates
    # given the current used_diffs.
    # Since rows are symmetric (order doesn't matter for validity), we can just
    # pick the next row to construct.
    # To apply MCV, we would need to know the "tightness" of each row.
    # In a sequential construction, the "most constrained" row is the one
    # that has the least freedom.
    # A simple heuristic: Construct rows in an order that maximizes constraint
    # propagation. E.g., construct the row with the largest scope first?
    # Or just sequential.
    
    # Let's try a randomized restart approach with MCV-inspired ordering.
    # Actually, for this specific problem (7 rows), sequential is likely fine
    # if we have good pruning.
    
    # Pruning: When constructing a row, if we hit a dead end, backtrack.
    
    def backtrack(row_idx):
        if time.time() > deadline:
            return None
        
        if row_idx == n_rows:
            return rows[:]
        
        # Generate candidates for this row
        # Candidate: a list of k+1 integers starting at 0, strictly increasing,
        # max <= max_scope, and differences disjoint from used_diffs.
        
        # To make this efficient, we can generate candidates greedily
        # and check for conflicts.
        
        # However, generating ALL candidates is expensive.
        # We can use a randomized search within the candidate space.
        
        # Let's try to find a valid row using a local search / greedy with backtracking.
        # Or just a simple DFS with pruning.
        
        # Simple DFS:
        # a[0] = 0
        # a[1] ... a[k]
        
        # To speed up, we can pre-filter possible values for a[1] etc.
        
        # Let's use a recursive helper to build the row.
        
        def build_row(depth, current_row, local_diffs):
            if time.time() > deadline:
                return None
            
            if depth == k + 1:
                # Check if this row is valid (differences disjoint from used_diffs)
                # local_diffs already contains all differences in this row.
                # We need to ensure no overlap with global used_diffs.
                if local_diffs & used_diffs:
                    return None
                # Also ensure local_diffs has no duplicates (Sidon property)
                # This is guaranteed by the construction if we check against local_diffs.
                # Wait, we must check against local_diffs during construction.
                
                # If valid, commit
                new_used = used_diffs | local_diffs
                rows.append(current_row)
                
                # Recursive call for next row
                res = backtrack(row_idx + 1)
                if res is not None:
                    return res
                
                # Backtrack
                rows.pop()
                return None
            
            # Determine range for next value
            start = current_row[-1] + 1 if current_row else 1
            # Upper bound for a[depth]
            # a[depth] <= max_scope
            # Also, we need enough space to fit k-depth more elements?
            # a[depth] + (k-depth) <= max_scope? No, they must be increasing by at least 1.
            # So a[depth] <= max_scope - (k - depth)
            end = max_scope - (k - depth)
            
            # Randomize order to help escape local minima?
            # Or try smallest first to keep scope small?
            # Since we are checking against a fixed max_scope, smallest first
            # might be better for finding *a* solution, but MCV suggests
            # trying values that are most constrained.
            
            # Let's try values in a randomized order, biased towards smaller?
            # Or just random.
            
            candidates = list(range(start, end + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                # Calculate new differences
                new_diffs = [val - a for a in current_row]
                
                # Check for conflicts with local_diffs
                conflict_local = False
                for d in new_diffs:
                    if d in local_diffs:
                        conflict_local = True
                        break
                if conflict_local:
                    continue
                
                # Check for conflicts with global used_diffs
                # We can prune early if any new_diff is in used_diffs
                conflict_global = False
                for d in new_diffs:
                    if d in used_diffs:
                        conflict_global = True
                        break
                if conflict_global:
                    continue
                
                # Recurse
                new_local = local_diffs | set(new_diffs)
                res = build_row(depth + 1, current_row + [val], new_local)
                if res is not None:
                    return res
            
            return None
        
        # Start building row
        # Optimization: If we have many rows, the "most constrained" row is the one
        # that has the least available space.
        # For now, let's just build row_idx.
        
        # To apply MCV, we could reorder the rows to build.
        # But since rows are symmetric, we can just build them in order.
        # However, the "density" of used_diffs increases, so later rows are more constrained.
        # This is naturally handled by the backtracking.
        
        return build_row(0, [0], set())
    
    return backtrack(0)


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    # Initial valid solution via greedy
    rows = _greedy(n_rows, k, rng)
    best = [list(r) for r in rows]
    best_scope = _scope(best)

    # Main SA with row-rebuild + decrement
    if time.time() < deadline - 10:
        improved, sc = _sa_optimize(rows, n_rows, k, deadline - 10, rng)
        if sc < best_scope and _validate(improved):
            best = improved
            best_scope = sc

    # Try CSP Backtracking with a tight scope limit
    # If SA found a good scope, try to beat it or match it with CSP.
    # CSP is exact, so if it finds a solution with scope < best_scope, it's great.
    # Let's try scope = best_scope - 1, then best_scope - 2, etc.
    # But CSP can be slow. Limit time.
    
    csp_deadline = deadline - 2
    current_limit = best_scope - 1
    # Try a few limits
    for limit in range(current_limit, max(100, current_limit - 10), -1):
        if time.time() > csp_deadline:
            break
        # CSP search
        csp_rows = _csp_solve(n_rows, k, limit, csp_deadline, rng)
        if csp_rows is not None and _validate(csp_rows):
            csp_scope = _scope(csp_rows)
            if csp_scope < best_scope:
                best = csp_rows
                best_scope = csp_scope
            # If we found a solution at this limit, we can stop or try lower?
            # If limit == best_scope - 1, we improved by 1.
            # We can try lower limits if time permits.
            # But finding a solution at limit L doesn't guarantee one at L-1.
            # So we should continue to lower limits.
            pass

    # Final cheap deterministic compression
    if time.time() < deadline - 1:
        comp, sc = _final_compress(best, n_rows, k, deadline - 1, rng)
        if sc < best_scope and _validate(comp):
            best = comp
            best_scope = sc

    # Final validation; fall back to greedy if anything went wrong
    if not _validate(best):
        best = _greedy(n_rows, k, rng)

    return best
