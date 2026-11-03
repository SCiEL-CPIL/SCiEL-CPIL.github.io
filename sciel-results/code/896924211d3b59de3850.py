"""Constraint-satisfaction search for the difference-triangle problem."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


def _build_diff_counts(rows):
    """Build a dict mapping diff value -> count of occurrences across all rows."""
    counts = {}
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] = counts.get(d, 0) + 1
    return counts


def _get_collisions(counts):
    """Count total collisions: sum over all d of (count[d] - 1) for count[d] > 1."""
    coll = 0
    for d, c in counts.items():
        if c > 1:
            coll += c - 1
    return coll


def _simulate_annealing(rows, n_rows, k, time_limit, rng, initial_best_max):
    """
    Simulated annealing with incremental diff-counting for O(k) cost updates.
    """
    row_len = k + 1
    rows = [list(r) for r in rows]
    current_max = max(r[-1] for r in rows)
    
    # Build diff counts
    counts = _build_diff_counts(rows)
    current_collisions = _get_collisions(counts)
    current_cost = current_collisions * 10000 + current_max
    
    best_local_cost = current_cost
    best_local_rows = [list(r) for r in rows]
    best_local_max = current_max
    
    T = 50.0
    cooling = 0.9998
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick random row and position to modify
        r_idx = rng.randint(0, n_rows - 1)
        i_idx = rng.randint(1, row_len - 1)
        
        lower = rows[r_idx][i_idx - 1] + 1
        upper_limit = best_local_max + 30
        if i_idx < row_len - 1:
            upper = rows[r_idx][i_idx + 1] - 1
            upper = min(upper, upper_limit)
        else:
            upper = upper_limit
        
        if lower > upper:
            continue
        
        new_val = rng.randint(lower, upper)
        if new_val == rows[r_idx][i_idx]:
            continue
        
        old_val = rows[r_idx][i_idx]
        
        # Compute delta in collisions by removing old diffs and adding new diffs
        delta_coll = 0
        for j in range(row_len):
            if j == i_idx:
                continue
            old_d = abs(old_val - rows[r_idx][j])
            new_d = abs(new_val - rows[r_idx][j])
            if old_d != new_d:
                # Remove old_d
                oc = counts.get(old_d, 0)
                if oc > 0:
                    if oc == 1:
                        del counts[old_d]
                        delta_coll -= 0
                    else:
                        counts[old_d] = oc - 1
                        delta_coll -= 1
                # Add new_d
                nc = counts.get(new_d, 0)
                if nc == 0:
                    counts[new_d] = 1
                    delta_coll += 0
                else:
                    counts[new_d] = nc + 1
                    delta_coll += 1
        
        new_collisions = current_collisions + delta_coll
        new_max = max(max(r) for r in rows)
        new_cost = new_collisions * 10000 + new_max
        
        # Accept if better or with probability exp(-delta/T)
        if new_cost < current_cost:
            accept = True
        else:
            delta = new_cost - current_cost
            if T > 0:
                import math
                accept = rng.random() < math.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            rows[r_idx][i_idx] = new_val
            current_collisions = new_collisions
            current_cost = new_cost
            current_max = max(max(r) for r in rows)
            
            if current_cost < best_local_cost:
                best_local_cost = current_cost
                best_local_rows = [list(r) for r in rows]
                best_local_max = current_max
        else:
            # Revert counts
            for j in range(row_len):
                if j == i_idx:
                    continue
                old_d = abs(old_val - rows[r_idx][j])
                new_d = abs(new_val - rows[r_idx][j])
                if old_d != new_d:
                    # Undo: remove new_d, add old_d back
                    nc = counts.get(new_d, 0)
                    if nc == 1:
                        del counts[new_d]
                    else:
                        counts[new_d] = nc - 1
                    oc = counts.get(old_d, 0)
                    if oc == 0:
                        counts[old_d] = 1
                    else:
                        counts[old_d] = oc + 1
        
        T *= cooling
    
    return best_local_rows, best_local_max


def _csp_build_rows(target_scope, time_limit, n_rows, k, rng):
    """
    CSP backtracking with forward checking to build rows within a target scope.
    """
    start_time = time.time()
    used = set()
    rows = []
    row_len = k + 1
    
    def build_one_row():
        """Build one row with forward checking. Returns list or None."""
        row = [0]
        
        def dfs_row(current_row, remaining):
            if time.time() - start_time > time_limit:
                return None
            if remaining == 0:
                return list(current_row)
            
            last = current_row[-1]
            # Try candidate next values
            candidates = []
            for v in range(last + 1, target_scope + 1):
                new_diffs = [v - a for a in current_row]
                # Check internal distinctness
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                # Check against used
                if any(d in used for d in new_diffs):
                    continue
                candidates.append((v, new_diffs))
            
            # Shuffle for diversity
            rng.shuffle(candidates)
            
            for v, new_diffs in candidates:
                # Tentatively add
                old_used = set(used)
                used.update(new_diffs)
                result = dfs_row(current_row + [v], remaining - 1)
                if result is not None:
                    return result
                used.clear()
                used.update(old_used)
            
            return None
        
        return dfs_row(row, k)
    
    for _ in range(n_rows):
        row = build_one_row()
        if row is None:
            return None
        # Update used
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
        rows.append(row)
    
    return rows


def _local_search(rows, k, deadline, rng):
    """
    Local search: try to reduce scope by shifting entries down.
    """
    best_rows = [list(r) for r in rows]
    best_sc = _scope(best_rows)
    row_len = k + 1
    
    def compute_used(rws):
        used = set()
        for r in rws:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    improved = True
    t0 = time.time()
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            # Try to reduce each entry (from last to first to preserve increase)
            for pos in range(len(row) - 1, 0, -1):
                lower = row[pos - 1] + 1
                upper = row[pos] - 1
                if upper < lower:
                    continue
                # Try candidate values from lower upward, pick the smallest valid
                for cand in range(lower, upper + 1):
                    if cand == row[pos]:
                        break
                    # Check: does setting row[pos]=cand cause any duplicate difference?
                    old = row[pos]
                    row[pos] = cand
                    # Check all pairs in this row
                    valid = True
                    diffs_in_row = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in diffs_in_row:
                                valid = False
                                break
                            diffs_in_row.add(d)
                        if not valid:
                            break
                    if valid:
                        # Check against other rows
                        for rj in range(len(best_rows)):
                            if rj == ri:
                                continue
                            r = best_rows[rj]
                            for i in range(len(r)):
                                for j in range(i + 1, len(r)):
                                    d = r[j] - r[i]
                                    if d in diffs_in_row:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                    if valid:
                        new_sc = _scope(best_rows)
                        if new_sc <= best_sc:
                            best_sc = new_sc
                            improved = True
                        else:
                            row[pos] = old
                    else:
                        row[pos] = old
                        break
        # Also try to reduce the overall max by checking if the global max entry
        # can be moved down
    return best_rows, best_sc


def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(12345)
    
    # Phase 1: Get a baseline with greedy
    best_rows = _greedy_fallback(n_rows, k)
    best_sc = _scope(best_rows)
    
    # Phase 2: Try CSP to get a tighter initial solution
    csp_start = time.time()
    csp_time = 20.0
    if time.time() + csp_time < deadline:
        # Try a few target scopes
        for target in range(best_sc, max(best_sc - 30, 100) - 1, -1):
            if time.time() - csp_start > csp_time:
                break
            result = _csp_build_rows(target, csp_time - (time.time() - csp_start), n_rows, k, rng)
            if result is not None:
                rmax = max(r[-1] for r in result)
                if rmax < best_sc:
                    best_sc = rmax
                    best_rows = result
                break
    
    # Phase 3: Simulated annealing with incremental diff-counting
    sa_time = min(80.0, deadline - 20.0)
    if time.time() < deadline - 20.0:
        rows, sc = _simulate_annealing(best_rows, n_rows, k, sa_time, rng, best_sc)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows
    
    # Phase 4: Local search to polish
    if time.time() < deadline - 5.0:
        rows, sc = _local_search(best_rows, k, deadline - 5.0, rng)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows
    
    return best_rows

# EVOLVE-BLOCK-END
