"""Difference triangle set (7,5): minimize the largest entry.
Structural redesign: bitmask-based difference tracking with row-swap local search
and iterative deepening backtracking."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _row_diff_mask(row):
    """Return a bitmask where bit d is set if d is a positive difference in the row."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask


def _all_diff_mask(rows):
    """Return the complete bitmask of all differences across all rows."""
    mask = 0
    for row in rows:
        mask |= _row_diff_mask(row)
    return mask


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    mask = 0
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rm = _row_diff_mask(row)
        if rm & mask:
            return False
        mask |= rm
    return True


def _greedy_construct(n_rows, k, seed_offset=0):
    """Greedy construction using bitmask for O(1) conflict checking."""
    rng = random.Random(42 + seed_offset)
    used_mask = 0
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            best_val = None
            best_score = float('inf')
            
            x_start = row[-1] + 1
            x_end = x_start + 250
            
            for x in range(x_start, x_end + 1):
                # Build mask of new differences
                new_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if new_mask & bit:
                        continue  # internal duplicate (shouldn't happen for increasing)
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if conflict:
                    continue
                
                # Score: prefer smaller x, reward packing (fewer isolated bits)
                score = x * 10 - bin(new_mask).count('1')
                if score < best_score:
                    best_score = score
                    best_val = x
            
            if best_val is None:
                x = row[-1] + 1
                while True:
                    new_mask = 0
                    conflict = False
                    for a in row:
                        d = x - a
                        bit = 1 << d
                        if used_mask & bit:
                            conflict = True
                            break
                        new_mask |= bit
                    if not conflict:
                        best_val = x
                        break
                    x += 1
            
            row.append(best_val)
            for a in row[:-1]:
                used_mask |= (1 << (best_val - a))
        rows.append(row)
    
    return rows


def _backtrack_row(k, used_mask, max_val, rng, deadline):
    """Build a single row using bitmask conflict checking. Returns row or None."""
    row = [0]
    
    def backtrack(idx):
        nonlocal used_mask
        if time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            new_mask = 0
            conflict = False
            for a in row:
                d = val - a
                bit = 1 << d
                if new_mask & bit:
                    conflict = True
                    break
                if used_mask & bit:
                    conflict = True
                    break
                new_mask |= bit
            if conflict:
                continue
            
            row.append(val)
            old_mask = used_mask
            used_mask |= new_mask
            if backtrack(idx + 1):
                return True
            used_mask = old_mask
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_with_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using bitmask backtracking."""
    used_mask = 0
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = _backtrack_row(k, used_mask, max_val, rng, deadline)
        if row is None:
            return None
        rows.append(row)
        used_mask |= _row_diff_mask(row)
    
    return rows


def _row_swap_local_search(rows, k, deadline, rng):
    """Local search: try shifting elements of the max-element row downward,
    and try swapping entire rows to improve packing. Uses bitmask for speed."""
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    T = 30.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        current_scope = _scope(rows)
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Element shift: pick a row with large max, try to shift an element
            ri = rng.randint(0, len(rows) - 1)
            row = rows[ri]
            pos = rng.randint(1, len(row) - 1)
            
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 5
            
            if low >= high:
                T *= cooling
                continue
            
            # Try values that reduce scope if this is near the max
            candidates = []
            if old_val >= current_scope - 5:
                for v in range(low, min(high + 1, old_val)):
                    candidates.append(v)
            for v in range(max(low, old_val - 20), min(high + 1, old_val + 5)):
                if v != old_val and v not in candidates:
                    candidates.append(v)
            rng.shuffle(candidates)
            
            for new_val in candidates[:15]:
                if new_val == old_val:
                    continue
                
                # Check conflict using bitmask
                old_mask = 0
                for i in range(pos):
                    old_mask |= (1 << (old_val - row[i]))
                for i in range(pos + 1, len(row)):
                    old_mask |= (1 << (row[i] - old_val))
                
                new_mask = 0
                conflict = False
                for i in range(pos):
                    d = new_val - row[i]
                    bit = 1 << d
                    if new_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                for i in range(pos + 1, len(row)):
                    d = row[i] - new_val
                    bit = 1 << d
                    if new_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                # Check against global used (excluding this row's old diffs)
                global_mask = _all_diff_mask(rows)
                other_mask = global_mask & ~old_mask
                if new_mask & other_mask:
                    continue
                
                # Compute new scope
                temp_row = row[:pos] + [new_val] + row[pos + 1:]
                new_scope = max(max(r) if r is not row else max(temp_row) for r in rows)
                
                delta = new_scope - current_scope
                if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    row[pos] = new_val
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    break
        
        elif move_type < 0.75:
            # Row swap: try swapping two rows to see if it helps (symmetry breaking)
            if len(rows) >= 2:
                r1, r2 = rng.sample(range(len(rows)), 2)
                rows[r1], rows[r2] = rows[r2], rows[r1]
                # Scope doesn't change with row swap, but this helps escape plateaus
                # in subsequent element shifts
        
        else:
            # Full row rebuild: pick the row with the largest max, try to rebuild it
            ri = max(range(len(rows)), key=lambda i: max(rows[i]))
            row = rows[ri]
            old_mask = _row_diff_mask(row)
            global_mask = _all_diff_mask(rows)
            other_mask = global_mask & ~old_mask
            
            # Try to rebuild this row with a tighter bound
            target_max = max(rows[ri]) - rng.randint(1, 5)
            new_row = _backtrack_row(k, other_mask, target_max, rng, time.time())
            if new_row is not None:
                new_scope = max(max(r) if i != ri else max(new_row) for i, r in enumerate(rows))
                if new_scope <= best_scope:
                    rows[ri] = new_row
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    return best_rows


def _iterative_deepening(n_rows, k, deadline, rng, initial_max=115):
    """Iterative deepening: try to find a solution with max_val = initial_max,
    then initial_max-1, etc. Uses bitmask backtracking."""
    best_rows = None
    best_scope = float('inf')
    
    for max_val in range(initial_max, 104, -1):
        if time.time() > deadline:
            break
        
        # Multiple randomized attempts
        attempts = max(1, int((deadline - time.time()) / 8.0))
        attempts = min(attempts, 5)
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            rows = _construct_with_backtracking(n_rows, k, max_val, rng, deadline)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    break  # Try next (lower) max_val
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 144.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy baseline with bitmask
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Iterative deepening backtracking with bitmask
    if time.time() < deadline - 50.0:
        bt_rows, bt_scope = _iterative_deepening(n_rows, k, deadline - 50.0, rng, initial_max=best_scope)
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Row-swap + element-shift local search with bitmask
    if time.time() < deadline - 5.0:
        improved = _row_swap_local_search(best_rows, k, deadline - 5.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Final polish - one more local search pass
    if time.time() < deadline - 1.0:
        improved2 = _row_swap_local_search(best_rows, k, deadline - 1.0, rng)
        new_scope2 = _scope(improved2)
        if new_scope2 < best_scope:
            best_rows = improved2
            best_scope = new_scope2
    
    return best_rows

# EVOLVE-BLOCK-END
