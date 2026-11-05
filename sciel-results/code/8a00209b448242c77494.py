"""Difference triangle set (7,5): minimize the largest entry.
Rewritten with optimized backtracking + simulated annealing + local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 3))
                found = False
                for _attempt in range(80):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _backtrack_row(k, used, max_val, rng, deadline):
    """Try to build a single row of k+1 elements starting at 0, with all differences
    not in 'used', and all elements <= max_val. Returns row or None."""
    if time.time() > deadline:
        return None
    
    row = [0]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        if low > high:
            return False
        
        # Generate candidates in a smart order: prefer smaller values first for packing
        # but use randomization to escape local optima
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            row.append(val)
            used.update(new_diffs)
            if backtrack(idx + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_with_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using backtracking, each with elements <= max_val."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = _backtrack_row(k, used, max_val, rng, deadline)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope:
                    continue
                
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                candidates = list(range(high, low - 1, -1))
                rng.shuffle(candidates)
                candidates.sort(key=lambda v: abs(v - old_val))
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, len(row)):
                        old_diffs.add(row[i] - old_val)
                    
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, len(row)):
                        new_diffs.add(row[i] - new_val)
                    
                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue
                    
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
    
    return rows


def _simulated_annealing(rows, k, deadline, rng):
    """Simulated annealing to reduce the maximum entry."""
    rows = [list(r) for r in rows]
    used = _all_diffs(rows)
    current_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    n_rows = len(rows)
    n_elements = k + 1
    
    # Temperature schedule
    t_start = 50.0
    t_end = 0.1
    n_iter = 20000
    cooling = (t_end / t_start) ** (1.0 / n_iter)
    temp = t_start
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        temp *= cooling
        
        # Pick a random row and position
        ri = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elements - 1)
        row = rows[ri]
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < n_elements else old_val + 15
        
        if low >= high:
            continue
        
        # Try a random new value
        new_val = rng.randint(low, high)
        if new_val == old_val:
            continue
        
        # Compute old and new diffs
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for i in range(pos + 1, n_elements):
            old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for i in range(pos + 1, n_elements):
            new_diffs.add(row[i] - new_val)
        
        # Check for conflicts
        conflict = new_diffs & (used - old_diffs)
        if conflict:
            # Penalty for conflicts
            delta_cost = len(conflict) * 10
            if rng.random() > (1.0 / (1.0 + 2.71828 ** (delta_cost / temp))):
                continue
        else:
            delta_cost = 0
        
        # Compute scope change
        new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
        old_row_max = max(row)
        new_scope = max(new_row_max, max(max(r) for r in rows if r is not row))
        scope_delta = new_scope - current_scope
        
        # Acceptance probability
        if scope_delta < 0 or delta_cost == 0:
            accept = True
        else:
            total_delta = scope_delta + delta_cost
            accept = rng.random() < (1.0 / (1.0 + 2.71828 ** (total_delta / temp)))
        
        if accept:
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            current_scope = max(max(r) for r in rows)
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy baseline (multiple seeds)
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(10):
        if time.time() > deadline - 60.0:
            break
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    # Phase 2: Randomized backtracking with decreasing bounds
    if time.time() < deadline - 40.0:
        for max_val in range(best_scope, 104, -1):
            if time.time() > deadline - 40.0:
                break
            for attempt in range(3):
                if time.time() > deadline - 40.0:
                    break
                bt_rows = _construct_with_backtracking(n_rows, k, max_val, rng, deadline - 40.0)
                if bt_rows is not None:
                    scope = _scope(bt_rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in bt_rows]
                        break
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 10.0:
        improved = _local_search_improve(best_rows, k, deadline - 10.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5.0:
        sa_rows, sa_scope = _simulated_annealing(best_rows, k, deadline - 5.0, rng)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 5: Final local search polish
    if time.time() < deadline - 2.0:
        improved = _local_search_improve(best_rows, k, deadline - 2.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
