"""
Difference triangle set (7,5) solver.

Strategy:
1. Fast greedy construction to get a valid baseline.
2. Targeted local search: shrink, swap, and row replacement.
3. Bounded backtracking search to find solutions at progressively lower scopes.
4. Simulated annealing with row-replacement moves.
5. Multi-start with diverse initializations.
6. Final exhaustive polish.
"""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    all_d = set()
    for r in rows:
        d = _diffs_of_row(r)
        if all_d & d:
            return False
        all_d |= d
    return True


def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
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


def _generate_row_greedy(used, k, rng, pool_size=3):
    """
    Greedily construct a row of k+1 integers starting at 0.
    At each step, pick from the top 'pool_size' smallest valid candidates.
    Returns (row, diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.5:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, frozenset(new_diffs)


def _generate_row_greedy_pure(used, k):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, frozenset(new_diffs)


def _try_shrink_position(row, pos, used, k):
    """
    Try to shrink row[pos] to the smallest valid value.
    Returns (new_row, new_used, scope_changed) or None.
    """
    old_val = row[pos]
    new_diffs_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            new_diffs_old.add(old_val - row[i])
        else:
            new_diffs_old.add(row[i] - old_val)
    
    remaining = used - new_diffs_old
    
    for new_val in range(row[pos-1] + 1, old_val):
        new_diffs_new = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs_new.add(new_val - row[i])
            else:
                new_diffs_new.add(row[i] - new_val)
        
        if not (new_diffs_new & remaining):
            new_row = row[:]
            new_row[pos] = new_val
            new_used = (used - new_diffs_old) | new_diffs_new
            return new_row, new_used, True
    
    return None


def _local_search(rows, k, rng, deadline):
    """
    Aggressive local search to reduce scope.
    Uses multiple strategies: shrink, swap, and row replacement.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    best_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        
        # Shuffle row order each iteration
        row_order = list(range(len(rows)))
        rng.shuffle(row_order)
        
        for idx in row_order:
            if time.time() > deadline:
                return best_rows
            
            row = rows[idx]
            
            # Try shrinking each position from right to left
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    return best_rows
                
                result = _try_shrink_position(row, pos, used, k)
                if result is not None:
                    new_row, new_used, _ = result
                    rows[idx] = new_row
                    used = new_used
                    current_scope = _total_scope(rows)
                    improved = True
                    no_improve_count = 0
                    
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                    break  # Restart position loop for this row
        
        if not improved:
            # Try swap-based moves
            improved = _try_swap_moves(rows, used, k, rng, deadline)
            if improved:
                no_improve_count = 0
                current_scope = _total_scope(rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
    
    return best_rows


def _try_swap_moves(rows, used, k, rng, deadline):
    """
    Try swapping elements between rows to free up differences.
    Returns True if any improvement was made.
    """
    improved = False
    n_rows = len(rows)
    
    for _ in range(10):
        if time.time() > deadline:
            break
        
        # Pick two random rows
        i, j = rng.sample(range(n_rows), 2)
        
        # Try swapping a pair of elements
        pos_i = rng.randint(1, len(rows[i]) - 1)
        pos_j = rng.randint(1, len(rows[j]) - 1)
        
        val_i = rows[i][pos_i]
        val_j = rows[j][pos_j]
        
        if val_i == val_j:
            continue
        
        # Check if swap maintains increasing order
        if pos_i > 1 and val_j <= rows[i][pos_i - 1]:
            continue
        if pos_i < len(rows[i]) - 1 and val_j >= rows[i][pos_i + 1]:
            continue
        if pos_j > 1 and val_i <= rows[j][pos_j - 1]:
            continue
        if pos_j < len(rows[j]) - 1 and val_i >= rows[j][pos_j + 1]:
            continue
        
        # Compute diff changes
        old_diffs_i = set()
        for p in range(len(rows[i])):
            if p == pos_i:
                continue
            if p < pos_i:
                old_diffs_i.add(val_i - rows[i][p])
            else:
                old_diffs_i.add(rows[i][p] - val_i)
        
        old_diffs_j = set()
        for p in range(len(rows[j])):
            if p == pos_j:
                continue
            if p < pos_j:
                old_diffs_j.add(val_j - rows[j][p])
            else:
                old_diffs_j.add(rows[j][p] - val_j)
        
        new_diffs_i = set()
        for p in range(len(rows[i])):
            if p == pos_i:
                continue
            if p < pos_i:
                new_diffs_i.add(val_j - rows[i][p])
            else:
                new_diffs_i.add(rows[i][p] - val_j)
        
        new_diffs_j = set()
        for p in range(len(rows[j])):
            if p == pos_j:
                continue
            if p < pos_j:
                new_diffs_j.add(val_i - rows[j][p])
            else:
                new_diffs_j.add(rows[j][p] - val_i)
        
        remaining = used - old_diffs_i - old_diffs_j
        if new_diffs_i & remaining or new_diffs_j & remaining:
            continue
        
        # Perform swap
        rows[i][pos_i] = val_j
        rows[j][pos_j] = val_i
        used = (used - old_diffs_i - old_diffs_j) | new_diffs_i | new_diffs_j
        improved = True
        
        # Try to shrink after swap
        for row_idx in [i, j]:
            for pos in range(len(rows[row_idx]) - 1, 0, -1):
                result = _try_shrink_position(rows[row_idx], pos, used, k)
                if result is not None:
                    new_row, new_used, _ = result
                    rows[row_idx] = new_row
                    used = new_used
    
    return improved


def _backtrack_row(used, max_val, row, depth, rng, deadline):
    """
    Build one row via backtracking within max_val.
    Returns row list or None if not possible.
    """
    if time.time() > deadline:
        return None
    
    if depth == k + 1:
        return row[:]
    
    last = row[-1] if row else 0
    start = last + 1
    
    # Try values in random order for diversity
    candidates = list(range(start, max_val + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            return None
        
        new_diffs = {x - a for a in row}
        if new_diffs & used:
            continue
        
        row.append(x)
        used_copy = used | new_diffs
        
        result = _backtrack_row(used_copy, max_val, row, depth + 1, rng, deadline)
        if result is not None:
            return result
        
        row.pop()
    
    return None


def _backtrack_construct(max_val, n_rows, k, rng, deadline):
    """
    Try to construct all rows within max_val using backtracking.
    Returns list of rows or None if not possible.
    """
    used = set()
    rows = []
    
    for i in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = [0]
        result = _backtrack_row(used, max_val, row, 1, rng, deadline)
        if result is None:
            return None
        
        rows.append(result)
        for a in result:
            for b in result:
                if a != b:
                    used.add(abs(a - b))
    
    return rows


def _simulated_annealing(rows, k, rng, deadline):
    """
    Simulated annealing with row-replacement moves.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    temperature = 20.0
    cooling_rate = 0.99
    
    while time.time() < deadline and temperature > 0.5:
        # Pick a random row to replace
        idx = rng.randint(0, len(rows) - 1)
        old_row = rows[idx]
        old_diffs = set(_diffs_of_row(old_row))
        
        # Generate a new candidate row
        remaining_used = used - old_diffs
        new_row, new_diffs = _generate_row_greedy(remaining_used, k, rng, pool_size=rng.choice([2, 3, 5]))
        
        if new_row is None:
            temperature *= cooling_rate
            continue
        
        # Check if new row is valid with other rows
        other_diffs = used - old_diffs
        if new_diffs & other_diffs:
            temperature *= cooling_rate
            continue
        
        # Compute new scope
        new_used = (used - old_diffs) | new_diffs
        new_scope = _total_scope([rows[i] if i != idx else new_row for i in range(len(rows))])
        
        delta_scope = new_scope - current_scope
        
        # Accept if better or with probability exp(-delta/temperature)
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / max(temperature, 0.1)):
            rows[idx] = new_row
            used = new_used
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        
        temperature *= cooling_rate
    
    return best_rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(row[pos-1] + 1, old_val):
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break
    
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Step 1: Get a valid baseline
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        return []
    
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # Step 2: Local search
    if time.time() < deadline - 40:
        ls_rows = _local_search(rows, k, rng, deadline - 40)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [r[:] for r in ls_rows]
            rows = ls_rows
    
    # Step 3: Bounded backtracking search for lower scopes
    if time.time() < deadline - 30:
        total_diffs = n_rows * k * (k + 1) // 2
        lower_bound = max(total_diffs, 50)
        
        current_max = best_scope
        
        while time.time() < deadline - 30:
            target_scope = current_max - 1
            
            if target_scope < lower_bound:
                break
            
            # Try backtracking with this max_val
            timeout_check = lambda: time.time() > deadline - 5
            
            found = False
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                result = _backtrack_construct(target_scope, n_rows, k, rng, deadline - 5)
                if result is not None:
                    best_rows = result
                    best_scope = target_scope
                    current_max = target_scope
                    found = True
                    break
            
            if not found:
                # Give up on this target, try one lower
                current_max = target_scope
    
    # Step 4: Simulated annealing
    if time.time() < deadline - 20:
        sa_rows = _simulated_annealing(rows, k, rng, deadline - 20)
        sa_scope = _total_scope(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = [r[:] for r in sa_rows]
            rows = sa_rows
    
    # Step 5: Multi-start
    if time.time() < deadline - 10:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Generate diverse initial solution
            used = set()
            new_rows = []
            for _ in range(n_rows):
                if rng.random() < 0.3:
                    row, diffs = _generate_row_greedy(used, k, rng, pool_size=rng.choice([2, 3, 5]))
                else:
                    row, diffs = _generate_row_greedy_pure(used, k)
                if row is not None:
                    new_rows.append(row)
                    used |= diffs
            
            if len(new_rows) < n_rows:
                continue
            
            # Local search on new solution
            ms_rows = _local_search(new_rows, k, rng, deadline - 15)
            ms_scope = _total_scope(ms_rows)
            
            if ms_scope < best_scope:
                best_scope = ms_scope
                best_rows = [r[:] for r in ms_rows]
                rows = ms_rows
    
    # Step 6: Final polish
    if time.time() < deadline - 2:
        polished = _polish_pass(best_rows, k, deadline - 2)
        polished_scope = _total_scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
    
    # Final validation
    if not _is_valid(best_rows):
        # Fallback to greedy
        return _greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
