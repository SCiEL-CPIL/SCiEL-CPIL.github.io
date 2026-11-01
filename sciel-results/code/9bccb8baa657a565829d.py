"""Structural rewrite: systematic M-search via backtracking with random restarts,
targeting the counting lower bound. Uses a different search paradigm than SA:
explicitly enumerate candidate M values from the lower bound upward and run
constraint-satisfaction backtracking with randomized variable/value ordering."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _counting_lower_bound(n_rows, k):
    """The simple counting bound: need n_rows * C(k+1,2) distinct positive differences,
    all <= M, so M >= n_rows * (k+1)*k/2."""
    return n_rows * (k + 1) * k // 2

def _backtrack_solve(n_rows, k, M, rng, time_budget):
    """Try to find n_rows rows of k+1 increasing integers starting at 0, max entry <= M,
    with all differences distinct. Uses backtracking with randomized ordering.
    Returns list of rows or None."""
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if M < total_diffs_needed:
        return None

    t0 = time.time()
    
    # We'll build rows one at a time. For each row, we pick k+1 values from {0..M}
    # such that the row is increasing, starts at 0, and all pairwise differences
    # are unique within the row and don't conflict with previously used differences.
    
    used_diffs = set()
    rows = []
    
    def _build_row(remaining_slots, start_val, current_row):
        """Recursively build one row. current_row starts with [0]."""
        if len(current_row) == k + 1:
            return current_row[:]
        
        elapsed = time.time() - t0
        if elapsed > time_budget:
            return None
        
        # Try candidate values for the next position
        prev = current_row[-1]
        candidates = list(range(prev + 1, M + 1))
        rng.shuffle(candidates)
        
        # Limit candidates to keep search tractable
        if len(candidates) > 50:
            # Keep a mix: some near prev, some spread out
            near = candidates[:20]
            spread = candidates[::max(1, len(candidates) // 30)][:30]
            candidates = list(set(near + spread))
            rng.shuffle(candidates)
        
        for val in candidates:
            # Check that adding val to current_row doesn't create duplicate diffs
            # within this row
            new_diffs = [val - x for x in current_row]
            if len(new_diffs) != len(set(new_diffs)):
                continue
            # Check against globally used diffs
            if any(d in used_diffs for d in new_diffs):
                continue
            
            current_row.append(val)
            for d in new_diffs:
                used_diffs.add(d)
            
            result = _build_row(remaining_slots - 1, val, current_row)
            
            current_row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
            
            if result is not None:
                return result
            if time.time() - t0 > time_budget:
                return None
        
        return None
    
    for i in range(n_rows):
        if time.time() - t0 > time_budget:
            return None
        row = _build_row(k, 0, [0])
        if row is None:
            return None
        rows.append(row)
    
    return rows

def _backtrack_solve_v2(n_rows, k, M, rng, time_budget):
    """Alternative: build all rows simultaneously using a global assignment approach.
    Assign positions row by row, but with more aggressive pruning."""
    t0 = time.time()
    
    used_diffs = set()
    rows = [[0] for _ in range(n_rows)]
    
    def _assign_all(row_idx, pos_idx):
        """Assign row_idx at position pos_idx (0-indexed, 0 is always 0)."""
        if time.time() - t0 > time_budget:
            return False
        if row_idx == n_rows:
            return True
        if pos_idx == 0:
            # Position 0 is always 0
            return _assign_all(row_idx + 1, 0)
        
        row = rows[row_idx]
        prev = row[pos_idx - 1]
        
        # Generate candidates
        candidates = list(range(prev + 1, M + 1))
        rng.shuffle(candidates)
        
        # Prune: limit candidates
        if len(candidates) > 40:
            near = candidates[:15]
            far = candidates[::max(1, len(candidates) // 25)][:25]
            candidates = list(set(near + far))
            rng.shuffle(candidates)
        
        for val in candidates:
            new_diffs = [val - x for x in row]
            if len(new_diffs) != len(set(new_diffs)):
                continue
            if any(d in used_diffs for d in new_diffs):
                continue
            
            row.append(val)
            for d in new_diffs:
                used_diffs.add(d)
            
            if pos_idx == k:
                if _assign_all(row_idx + 1, 0):
                    return True
            else:
                if _assign_all(row_idx, pos_idx + 1):
                    return True
            
            row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
        
        return False
    
    if _assign_all(0, 0):
        return [r[:] for r in rows]
    return None

def _greedy_construct(n_rows, k, rng=None):
    """Fast greedy construction for initial solution or fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = [x - a for a in row]
            if any(d in used for d in new_diffs):
                continue
            if len(new_diffs) != len(set(new_diffs)):
                continue
            row.append(x)
            for d in new_diffs:
                used.add(d)
        rows.append(row)
    return rows

def _local_improve(rows, n_rows, k, rng, time_budget):
    """Simple local search to reduce max entry after initial construction."""
    t0 = time.time()
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    
    while time.time() - t0 < time_budget:
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        row = cur[ri]
        old_val = row[pi]
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else cur_val
        
        if hi <= lo:
            continue
        
        # Try to decrease the value
        candidates = list(range(lo, old_val))
        rng.shuffle(candidates)
        
        found = False
        for nv in candidates:
            cand_row = row[:]
            cand_row[pi] = nv
            n = len(cand_row)
            # Check Sidon within row
            seen = set()
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in seen:
                        ok = False
                        break
                    seen.add(d)
                if not ok:
                    break
            if not ok:
                continue
            # Check against other rows
            conflict = False
            for i2, r2 in enumerate(cur):
                if i2 == ri:
                    continue
                for a in range(len(r2)):
                    for b in range(a + 1, len(r2)):
                        if r2[b] - r2[a] in seen:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if conflict:
                continue
            
            # Accept
            cur[ri] = cand_row
            new_val = _max_entry(cur)
            if new_val < best_val:
                best_val = new_val
                best = [r[:] for r in cur]
            found = True
            break
        
        if not found and rng.random() < 0.1:
            # Occasionally try increasing a value to escape local minima
            ri2 = rng.randrange(n_rows)
            pi2 = rng.randrange(1, k + 1)
            row2 = cur[ri2]
            old2 = row2[pi2]
            lo2 = row2[pi2 - 1] + 1
            hi2 = row2[pi2 + 1] - 1 if pi2 < k else min(old2 + 5, cur_val + 10)
            if hi2 > lo2:
                nv2 = rng.randint(lo2, hi2)
                if nv2 != old2:
                    cand2 = row2[:]
                    cand2[pi2] = nv2
                    n2 = len(cand2)
                    seen2 = set()
                    ok2 = True
                    for i in range(n2):
                        for j in range(i + 1, n2):
                            d = cand2[j] - cand2[i]
                            if d in seen2:
                                ok2 = False
                                break
                            seen2.add(d)
                        if not ok2:
                            break
                    if ok2:
                        conflict2 = False
                        for i2, r2 in enumerate(cur):
                            if i2 == ri2:
                                continue
                            for a in range(len(r2)):
                                for b in range(a + 1, len(r2)):
                                    if r2[b] - r2[a] in seen2:
                                        conflict2 = True
                                        break
                                if conflict2:
                                    break
                            if conflict2:
                                break
                        if not conflict2:
                            cur[ri2] = cand2
    
    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    lb = _counting_lower_bound(n_rows, k)
    # Best known upper bound for (7,5) is 112
    # Start searching from lb upward
    best_solution = None
    best_val = float('inf')
    
    # First, get a quick greedy solution as fallback
    greedy_sol = _greedy_construct(n_rows, k)
    greedy_val = _max_entry(greedy_sol)
    best_solution = greedy_sol
    best_val = greedy_val
    
    # Try systematic backtracking from lower bound upward
    rng = random.Random(42)
    
    for M in range(lb, min(lb + 30, greedy_val + 1)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple restarts for this M
        max_restarts = 5
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            budget = min(15.0, deadline - time.time() - 3)
            if budget < 2.0:
                break
            
            sol = _backtrack_solve_v2(n_rows, k, M, rng, budget)
            if sol is not None:
                if _valid(sol, n_rows, k):
                    if M < best_val:
                        best_val = M
                        best_solution = sol
                    found = True
                    break
        
        if found:
            break
    
    # If we found a solution, try to improve with local search
    if best_solution is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            improved, imp_val = _local_improve(best_solution, n_rows, k, rng, remaining)
            if imp_val < best_val:
                best_val = imp_val
                best_solution = improved
    
    # Final validation
    if best_solution is None or not _valid(best_solution, n_rows, k):
        best_solution = _greedy_construct(n_rows, k)
    
    return best_solution

# EVOLVE-BLOCK-END
