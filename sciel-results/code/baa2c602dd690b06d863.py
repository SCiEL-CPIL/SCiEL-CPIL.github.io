import time
import random
import math
from itertools import combinations

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _row_diffs(row):
    """Return the set of 15 pairwise differences for a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _enumerate_rows(m, k, rng=None):
    """
    Enumerate all valid rows of length k+1 starting at 0 with max <= m.
    A row is valid if all 15 pairwise differences are distinct and <= m.
    For k=5, we need 6 elements: 0 = a0 < a1 < a2 < a3 < a4 < a5 <= m.
    """
    rows = []
    # For k=5: choose 5 values from 1..m, sort them
    # But that's C(m,5) which is huge. Instead, use a smarter enumeration.
    # For small m (<= ~120), C(m,5) is manageable with pruning.
    
    # Actually, let's use a different approach: generate candidate rows via
    # a structured method. For k=5, a row is (0, a, b, c, d, e) with 0 < a < b < c < d < e <= m.
    # The 15 differences are: a, b, c, d, e, b-a, c-a, d-a, e-a, c-b, d-b, e-b, d-c, e-c, e-d
    # All must be distinct and in [1, m].
    
    # For efficiency, we'll use a partial enumeration with pruning.
    # For k=5 and m <= 120, the total number of valid rows is manageable.
    
    # Use itertools to generate combinations of 5 values from range(1, m+1)
    # But C(120, 5) = 190,578,024 which is too many.
    # We need a smarter approach.
    
    # Alternative: use the fact that the differences must all be distinct.
    # For k=5, the 15 differences are determined by the 5 chosen values.
    # Let's use a recursive backtracking approach.
    
    def backtrack(remaining_slots, current_row, used_diffs):
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            return
        
        if len(current_row) == 0:
            current_row.append(0)
        
        last = current_row[-1]
        slots_left = (k + 1) - len(current_row)
        
        # The next value must be > last and <= m
        # Also, the remaining slots must fit: next value + (slots_left - 1) <= m is not required,
        # but we need enough room for increasing values.
        
        start = last + 1
        # Upper bound: we need slots_left values from start to m, so start + slots_left - 1 <= m
        # Actually no, we just need each subsequent value to be larger.
        # But for efficiency, the max possible value for the next element is m - (slots_left - 1)
        end = m - (slots_left - 1) + 1  # inclusive
        if end > m:
            end = m
        
        for v in range(start, end + 1):
            # Check if adding v creates any duplicate differences
            new_diffs = set()
            ok = True
            for a in current_row:
                d = v - a
                if d <= 0:
                    continue
                if d in used_diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            
            current_row.append(v)
            backtrack(remaining_slots - 1, current_row, used_diffs | new_diffs)
            current_row.pop()
    
    backtrack(k, [], set())
    return rows

def _backtrack_solve(n_rows, k, m, rng):
    """
    Try to find n_rows valid rows with max element <= m.
    Uses backtracking with MRV (minimum remaining values) heuristic.
    """
    # First, enumerate all valid rows within [0, m]
    # For efficiency, we do this on-the-fly with caching
    
    all_rows = _enumerate_rows(m, k)
    if not all_rows:
        return None
    
    # Convert to difference sets for quick conflict checking
    row_diffs = [_row_diffs(r) for r in all_rows]
    
    # For each row index, we need to pick one row such that all difference sets are disjoint
    # This is a constraint satisfaction problem.
    
    # Precompute for each pair of rows whether they conflict
    # n_rows is small (7), all_rows might be large (~10k-50k for m=112)
    # We'll use a greedy + backtracking approach.
    
    best = None
    
    # Try multiple random orderings
    for trial in range(50):
        rng.shuffle(all_rows)
        rng.shuffle(row_diffs)
        
        used_diffs = set()
        placed = []
        
        def solve(idx):
            nonlocal best
            if idx == n_rows:
                best = [list(r) for r in placed]
                return True
            
            # MRV: try rows in order of fewest conflicts with already-placed rows
            # For simplicity, just iterate in shuffled order
            for i in range(len(all_rows)):
                if i in placed:
                    continue
                rd = row_diffs[i]
                if rd & used_diffs:
                    continue
                # Place this row
                placed.append(i)
                used_diffs |= rd
                if solve(idx + 1):
                    return True
                used_diffs -= rd
                placed.pop()
            
            return False
        
        if solve(0):
            return best
    
    return None

def _greedy_initial(n_rows, k, m, rng):
    """Greedy construction with randomization, targeting scope m."""
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        for _j in range(k):
            found = False
            for t in range(x + 1, m + 1):
                ok = True
                for a in row:
                    d = t - a
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                if ok:
                    row.append(t)
                    x = t
                    for a in row[:-1]:
                        used_diffs.add(t - a)
                    found = True
                    break
            if not found:
                # Try with some randomness
                for _attempt in range(20):
                    row = [0]
                    x = 0
                    ok_row = True
                    for _j2 in range(k):
                        found2 = False
                        for t in range(x + 1, m + 1):
                            ok2 = True
                            for a in row:
                                d = t - a
                                if d <= 0 or d in used_diffs:
                                    ok2 = False
                                    break
                            if ok2:
                                row.append(t)
                                x = t
                                found2 = True
                                break
                        if not found2:
                            ok_row = False
                            break
                    if ok_row:
                        for a in row[:-1]:
                            used_diffs.add(row[-1] - a)
                        # Actually need all pairwise diffs
                        used_diffs = used_diffs | _row_diffs(row)
                        rows.append(row)
                        found = True
                        break
                if not found:
                    return None
                continue
            else:
                rows.append(row)
                used_diffs |= _row_diffs(row)
    
    return rows

def _sa_optimize(rows, k, deadline, rng):
    """Simulated annealing with efficient cost tracking."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # Build difference count
    diff_counts = {}
    for r in current_rows:
        for d in _row_diffs(r):
            diff_counts[d] = diff_counts.get(d, 0) + 1
    
    def get_cost():
        collisions = sum(max(0, c - 1) for c in diff_counts.values())
        return current_scope + 1000 * collisions
    
    current_cost = get_cost()
    best_cost = current_cost
    
    T = 5.0
    cooling = 0.9998
    min_T = 0.02
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current_rows[i][j]
        
        lo = current_rows[i][j-1] + 1
        hi = current_rows[i][j+1] - 1 if j < k else current_scope
        
        if hi <= lo:
            continue
        
        # Bias toward decreasing the value
        if old_val > lo and rng.random() < 0.7:
            new_val = rng.randint(lo, old_val - 1)
        else:
            new_val = rng.randint(old_val + 1, hi) if old_val < hi else old_val
        
        if new_val == old_val:
            continue
        
        # Remove old row's diffs
        for d in _row_diffs(current_rows[i]):
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        
        current_rows[i][j] = new_val
        
        # Add new row's diffs
        for d in _row_diffs(current_rows[i]):
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        new_scope = _scope(current_rows)
        new_cost = new_scope + 1000 * sum(max(0, c - 1) for c in diff_counts.values())
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
            current_cost = new_cost
            current_scope = new_scope
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            for d in _row_diffs(current_rows[i]):
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            current_rows[i][j] = old_val
            for d in _row_diffs(current_rows[i]):
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    rng = random.Random(12345)
    
    # Lower bound: n_rows * k*(k+1)/2 = 7 * 15 = 105
    lower_bound = n_rows * k * (k + 1) // 2
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try to solve exactly using backtracking for increasing m
    # Start from lower bound and go up
    for m in range(lower_bound, lower_bound + 20):
        if time.time() > deadline - 60:
            break
        solution = _backtrack_solve(n_rows, k, m, rng)
        if solution is not None and _valid(solution, k):
            best_scope = m
            best_rows = solution
            break
    
    # Phase 2: If backtracking didn't find a solution, use greedy + SA
    if best_rows is None or time.time() < deadline - 30:
        # Generate diverse initial solutions
        candidates = []
        for trial in range(20):
            if time.time() > deadline - 40:
                break
            m_try = lower_bound + trial
            rows = _greedy_initial(n_rows, k, m_try, rng)
            if rows is not None and _valid(rows, k):
                s = _scope(rows)
                candidates.append((s, rows))
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
        
        if not candidates:
            # Last resort: simple sequential
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                for j in range(1, k+1):
                    x += 1
                    while any((x - row[i]) in used for i in range(len(row))):
                        x += 1
                    for i in range(len(row)):
                        used.add(x - row[i])
                    row.append(x)
                rows.append(row)
            return rows
        
        candidates.sort(key=lambda x: x[0])
        
        # Phase 3: SA optimization
        remaining = deadline - time.time()
        top_n = min(3, len(candidates))
        time_per = remaining / max(1, top_n)
        
        for idx in range(top_n):
            if time.time() > deadline - 10:
                break
            s, rows = candidates[idx]
            sa_deadline = time.time() + min(time_per, deadline - time.time() - 5)
            sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
            
            if sa_rows is not None and _valid(sa_rows, k):
                actual = _scope(sa_rows)
                if actual < best_scope:
                    best_scope = actual
                    best_rows = sa_rows
    
    return best_rows

