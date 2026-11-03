"""Difference triangle set (4,7) — DSATUR graph coloring + SA refinement."""
import time
import random
import sys
from collections import Counter, defaultdict

# EVOLVE-BLOCK-START


def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _max_entry(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _count_conflicts(rows):
    """Count how many duplicate differences exist across all rows."""
    from collections import Counter
    seen = Counter()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                seen[d] += 1
    conflicts = 0
    for d, cnt in seen.items():
        if cnt > 1:
            conflicts += cnt - 1
    return conflicts


def _greedy_build(n_rows, k, rng, bias=1.0):
    """Greedy build with optional randomness."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if bias >= 1.0 or rng.random() < bias:
                    row.append(x)
                    used |= new
                else:
                    if rng.random() > 0.3:
                        row.append(x)
                        used |= new
        rows.append(row)
    return rows


def _enumerate_rows(k, max_scope, rng, sample_limit=5000):
    """
    Enumerate candidate rows of k+1 increasing integers starting at 0
    with max entry <= max_scope. Returns list of (row_tuple, diff_set).
    Uses randomized DFS to sample a diverse subset.
    """
    rows_found = []
    seen_diffs = set()  # to deduplicate by diff-set
    
    def dfs(current_row, start, depth):
        if len(rows_found) >= sample_limit:
            return True  # stop
        if depth == k + 1:
            # Compute diff set
            ds = frozenset()
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    ds = ds | frozenset([current_row[j] - current_row[i]])
            if ds not in seen_diffs:
                seen_diffs.add(ds)
                rows_found.append((tuple(current_row), ds))
            return False
        # Try values from start to max_scope
        # Bias toward smaller values for tighter packing
        max_val = max_scope
        # Randomize order slightly
        vals = list(range(start, max_val + 1))
        if rng.random() < 0.5:
            rng.shuffle(vals)
        else:
            # Partial shuffle: keep small values more likely
            for i in range(len(vals) - 1, 0, -1):
                j = rng.randint(0, i)
                vals[i], vals[j] = vals[j], vals[i]
        for v in vals:
            if len(rows_found) >= sample_limit:
                return True
            current_row.append(v)
            if dfs(current_row, v + 1, depth + 1):
                current_row.pop()
                return True
            current_row.pop()
        return False
    
    dfs([0], 1, 1)
    return rows_found


def _dsatur_select(rows_candidates, n_rows, rng, deadline):
    """
    Use DSATUR-inspired greedy to select n_rows rows from candidates
    such that no two share a difference. Minimizes max scope.
    rows_candidates: list of (row_tuple, diff_set)
    Returns: list of rows (as lists) or None
    """
    # Sort candidates by max entry (scope) ascending
    candidates = sorted(rows_candidates, key=lambda x: max(x[0]))
    
    # Build conflict structure: for each candidate, which others conflict
    # To save memory, we'll check on-the-fly
    
    selected = []
    selected_diffs = set()
    
    # DSATUR: at each step, pick the candidate with the most "saturation"
    # (most conflicts with already-selected) but that's still feasible.
    # Actually for independent set selection, we want candidates with NO conflicts.
    # So greedy: pick the smallest-scope candidate that doesn't conflict with selected.
    
    for _ in range(n_rows):
        best_candidate = None
        best_scope = float('inf')
        # Randomize tie-breaking
        order = list(range(len(candidates)))
        rng.shuffle(order)
        for idx in order:
            row, ds = candidates[idx]
            scope = max(row)
            if scope >= best_scope:
                continue
            # Check no conflict with selected
            if not (ds & selected_diffs):
                best_candidate = (row, ds)
                best_scope = scope
        if best_candidate is None:
            return None
        selected.append(list(best_candidate[0]))
        selected_diffs |= best_candidate[1]
        # Remove this candidate from list (mark as used)
        candidates.remove(best_candidate)
    
    return selected


def _sa_optimize(rows, deadline, rng):
    """
    Simulated annealing on the full set of rows.
    Moves: pick a random row and a random position, then try to compact
    that element or the whole row by shifting elements down.
    Accepts invalid states with probability based on conflict count and temperature.
    """
    n_rows = len(rows)
    k = len(rows[0]) - 1
    
    # Current state
    current = [r[:] for r in rows]
    current_conflicts = _count_conflicts(current)
    current_scope = _max_entry(current)
    
    # Best state
    best = [r[:] for r in current]
    best_scope = current_scope
    best_conflicts = current_conflicts
    
    # Temperature schedule
    t_start = time.time()
    total_time = deadline - t_start
    t0 = 100.0  # initial temperature (in terms of conflict units)
    t_end = 0.01
    
    def temperature(elapsed_frac):
        # Exponential cooling
        return t0 * (t_end / t0) ** elapsed_frac
    
    steps = 0
    while time.time() < deadline:
        elapsed_frac = (time.time() - t_start) / max(total_time, 1e-9)
        if elapsed_frac >= 1.0:
            break
        T = temperature(elapsed_frac)
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Compact a single element: try to reduce row[r][p] to a smaller value
            r_idx = rng.randrange(n_rows)
            if len(current[r_idx]) <= 1:
                continue
            p = rng.randrange(1, len(current[r_idx]))
            old_val = current[r_idx][p]
            
            # Try values from old_val-1 down to current[r_idx][p-1]+1
            found = False
            for new_val in range(old_val - 1, current[r_idx][p - 1], -1):
                row = current[r_idx]
                row[p] = new_val
                # Check intra-row distinctness
                valid_intra = True
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if row[j] == row[i]:
                            valid_intra = False
                            break
                    if not valid_intra:
                        break
                if not valid_intra:
                    row[p] = old_val
                    continue
                # Check if this creates inter-row conflicts
                new_diffs = set()
                for i in range(len(row)):
                    if i != p:
                        d = abs(row[p] - row[i])
                        new_diffs.add(d)
                
                conflict_count = 0
                for other_idx in range(n_rows):
                    if other_idx == r_idx:
                        continue
                    other_row = current[other_idx]
                    for i in range(len(other_row)):
                        for j in range(i + 1, len(other_row)):
                            d = other_row[j] - other_row[i]
                            if d in new_diffs:
                                conflict_count += 1
                
                if conflict_count == 0:
                    found = True
                    break
                else:
                    row[p] = old_val
                    if rng.random() < T / max(conflict_count, 1):
                        found = True
                        break
            
            if found:
                new_scope = _max_entry(current)
                new_conflicts = _count_conflicts(current)
                delta_scope = new_scope - current_scope
                delta_conf = new_conflicts - current_conflicts
                if delta_scope < 0 or delta_conf < 0:
                    current = [r[:] for r in current]
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                    if new_scope < best_scope and new_conflicts == 0:
                        best = [r[:] for r in current]
                        best_scope = new_scope
                        best_conflicts = 0
                elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                    current = [r[:] for r in current]
                    current_scope = new_scope
                    current_conflicts = new_conflicts
        
        elif move_type < 0.75:
            # Rebuild a single row greedily with randomness
            r_idx = rng.randrange(n_rows)
            old_row = current[r_idx][:]
            
            used_others = set()
            for i in range(n_rows):
                if i != r_idx:
                    used_others |= _all_diffs(current[i])
            
            new_row = [0]
            x = 0
            max_x = old_row[-1] + 5
            while len(new_row) < len(old_row) and x < max_x:
                x += 1
                new = {x - a for a in new_row}
                if not new & used_others:
                    if rng.random() < 0.8:
                        new_row.append(x)
                    else:
                        if rng.random() < 0.4:
                            new_row.append(x)
            
            if len(new_row) == len(old_row):
                trial = current[:]
                trial[r_idx] = new_row
                new_conflicts = _count_conflicts(trial)
                new_scope = _max_entry(trial)
                
                delta_scope = new_scope - current_scope
                delta_conf = new_conflicts - current_conflicts
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best = trial
                    best_scope = new_scope
                    best_conflicts = 0
                    current = trial
                    current_scope = new_scope
                    current_conflicts = 0
                elif delta_scope < 0 or delta_conf < 0:
                    current = trial
                    current_scope = new_scope
                    current_conflicts = new_conflicts
                elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                    current = trial
                    current_scope = new_scope
                    current_conflicts = new_conflicts
        
        else:
            # Swap elements between two rows
            r1 = rng.randrange(n_rows)
            r2 = rng.randrange(n_rows)
            if r1 == r2:
                continue
            if len(current[r1]) != len(current[r2]):
                continue
            old1 = current[r1][-1]
            old2 = current[r2][-1]
            if old1 == old2:
                continue
            current[r1][-1] = old2
            current[r2][-1] = old1
            
            new_conflicts = _count_conflicts(current)
            new_scope = _max_entry(current)
            delta_scope = new_scope - current_scope
            delta_conf = new_conflicts - current_conflicts
            
            if new_conflicts == 0 and new_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = new_scope
                best_conflicts = 0
                current_scope = new_scope
                current_conflicts = new_conflicts
            elif delta_scope < 0 or delta_conf < 0:
                current_scope = new_scope
                current_conflicts = new_conflicts
            elif rng.random() < T / max(abs(delta_scope) + abs(delta_conf), 1):
                current_scope = new_scope
                current_conflicts = new_conflicts
            else:
                current[r1][-1] = old1
                current[r2][-1] = old2
        
        steps += 1
        if steps % 1000 == 0 and current_conflicts == 0:
            if current_scope < best_scope:
                best = [r[:] for r in current]
                best_scope = current_scope
                best_conflicts = 0
    
    if best_conflicts == 0:
        return best
    if current_conflicts == 0:
        return current
    return best


def _local_compact(rows, deadline, rng):
    """
    Local search: try to reduce each row's max entry by backtracking
    from the last element inward.
    """
    n_rows = len(rows)
    current = [r[:] for r in rows]
    current_scope = _max_entry(current)
    best = [r[:] for r in current]
    best_scope = current_scope
    
    t_start = time.time()
    
    while time.time() < deadline:
        improved = False
        for r_idx in range(n_rows):
            row = current[r_idx]
            k = len(row) - 1
            # Try to reduce from the last element
            for p in range(k, 0, -1):
                old_val = row[p]
                # Try smaller values
                for new_val in range(row[p-1] + 1, old_val):
                    row[p] = new_val
                    # Check intra-row validity
                    valid = True
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            if row[j] <= row[i]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        row[p] = old_val
                        continue
                    # Check inter-row conflicts
                    new_diffs = set()
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            new_diffs.add(row[j] - row[i])
                    
                    conflict = False
                    for other_idx in range(n_rows):
                        if other_idx == r_idx:
                            continue
                        other = current[other_idx]
                        for i in range(len(other)):
                            for j in range(i+1, len(other)):
                                if other[j] - other[i] in new_diffs:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    
                    if not conflict:
                        new_scope = _max_entry(current)
                        if new_scope < best_scope:
                            best = [r[:] for r in current]
                            best_scope = new_scope
                        improved = True
                        break
                    else:
                        row[p] = old_val
                if improved:
                    break
            if improved:
                current_scope = best_scope
                # Reset current to best
                current = [r[:] for r in best]
                break
        
        if not improved:
            # Try randomized restarts of local search
            if rng.random() < 0.3:
                # Perturb: randomly increase an element
                r_idx = rng.randrange(n_rows)
                p = rng.randrange(1, len(current[r_idx]))
                current[r_idx][p] += rng.randint(1, 5)
                current_scope = _max_entry(current)
            else:
                break
    
    return best, best_scope


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    rng = random.Random(42)
    
    best = None
    best_score = float('inf')
    
    # Phase 1: Greedy baselines
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=rng.choice([1.0, 0.95, 0.9, 0.85]))
        if _is_valid(trial):
            sc = _max_entry(trial)
            if sc < best_score:
                best_score = sc
                best = trial
    
    if best is None:
        best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
        best_score = _max_entry(best)
    
    # Phase 2: DSATUR-style graph coloring approach
    # Enumerate candidate rows and select independent set
    t_phase2_start = time.time()
    dsatur_deadline = min(deadline - 40.0, t_phase2_start + 30.0)
    
    # Try multiple scope targets
    for scope_target in [134, 130, 126, 122, 118, 114, 112]:
        if time.time() > dsatur_deadline:
            break
        if scope_target < best_score - 5:
            continue  # Not worth trying if already much better
        
        # Enumerate rows within this scope
        candidates = _enumerate_rows(k, scope_target, rng, sample_limit=3000)
        if len(candidates) < n_rows:
            continue
        
        # Try DSATUR selection multiple times with different random orders
        for attempt in range(5):
            if time.time() > dsatur_deadline:
                break
            selected = _dsatur_select(
                [c for c in candidates], n_rows, 
                random.Random(attempt * 1000 + seed if 'seed' in dir() else attempt),
                dsatur_deadline
            )
            if selected is not None and _is_valid(selected):
                sc = _max_entry(selected)
                if sc < best_score:
                    best_score = sc
                    best = selected
    
    # Phase 3: Local compaction
    if time.time() < deadline - 10.0:
        best, best_score = _local_compact(best, deadline - 5.0, rng)
    
    # Phase 4: SA refinement on remaining time
    if time.time() < deadline - 5.0:
        best = _sa_optimize(best, deadline - 2.0, rng)
        if _is_valid(best):
            final_score = _max_entry(best)
            if final_score < best_score:
                best_score = final_score
    
    return best

# EVOLVE-BLOCK-END
