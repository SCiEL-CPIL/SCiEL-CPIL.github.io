"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

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
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Return the set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    """Return the list of all pairwise differences in a row."""
    n = len(row)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _is_sidon(row):
    """Check if a row is a Sidon set (all pairwise differences distinct)."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return len(diffs) == len(set(diffs))

def _check_row_valid(row, k):
    """Check if a single row is strictly increasing and starts at 0."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    random.seed(42)

    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)
    
    # Track used differences
    best_used = set()
    for r in best:
        best_used |= _row_diffs_set(r)
    
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set(best_used)
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    
    # Simulated annealing parameters
    T = 30.0
    T_min = 0.1
    cool = 0.99995
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 30000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        step += 1
        
        # Pick a random row and a random position to perturb
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        old_val = cur[ri][pi]
        
        # Generate candidate moves
        candidates = []
        
        # Try decreasing values (to reduce scope)
        for d in range(1, 25):
            new_val = old_val - d
            if new_val <= cur[ri][pi - 1]:
                break
            candidates.append(new_val)
        
        # Try small increases for diversification
        for d in range(1, 15):
            new_val = old_val + d
            if pi < k and new_val >= cur[ri][pi + 1]:
                break
            candidates.append(new_val)
        
        # Random perturbation
        for d in range(-20, 21):
            if d == 0:
                continue
            new_val = old_val + d
            if new_val <= cur[ri][pi - 1]:
                continue
            if pi < k and new_val >= cur[ri][pi + 1]:
                continue
            if new_val not in candidates:
                candidates.append(new_val)
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            # Check strict increase within row
            if new_val <= cur[ri][pi - 1]:
                continue
            if pi < k and new_val >= cur[ri][pi + 1]:
                continue
            
            # Build candidate row
            cand_row = cur[ri][:]
            cand_row[pi] = new_val
            
            # Check row validity
            if not _check_row_valid(cand_row, k):
                continue
            
            # Compute new diffs for this row
            new_diffs = _row_diffs_set(cand_row)
            
            # Check for conflicts with other rows' diffs
            other_diffs = cur_used - cur_row_diffs[ri]
            if new_diffs & other_diffs:
                continue
            
            # Check Sidon property within the row
            if not _is_sidon(cand_row):
                continue
            
            # Compute new max
            max_other = 0
            for i, r in enumerate(cur):
                if i != ri:
                    m = max(r)
                    if m > max_other:
                        max_other = m
            cand_max = max(max_other, max(cand_row))
            
            # Acceptance
            delta_e = cand_max - cur_val
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if random.random() < 0.3:
                    accept = True
            else:
                if T > T_min:
                    prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                cur_used -= cur_row_diffs[ri]
                cur_used |= new_diffs
                cur[ri] = cand_row
                cur_row_diffs[ri] = new_diffs
                cur_val = cand_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                moved = True
                break
        
        if not moved:
            no_improve_steps += 1
        
        # Cool down
        T = max(T_min, T * cool)
        
        # Periodic restart
        if no_improve_steps > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 30.0
            no_improve_steps = 0
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best
# EVOLVE-BLOCK-END
