"""Difference triangle set construction with time-bounded search."""
import time
import random
import sys

def _greedy_construct(n_rows, k, used=None):
    """Build n_rows rows greedily, reusing 'used' if provided (else fresh set)."""
    if used is None:
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
    return rows, used

def _row_differences(row):
    """Return the set of positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check that all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        rd = _row_differences(row)
        if rd & all_diffs:
            return False
        all_diffs |= rd
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)

def _sum_entries(rows):
    return sum(sum(row) for row in rows)

def _local_search(rows, k, max_iters=50000, time_limit=10.0):
    """
    Local search: try to reduce the maximum entry by shifting rows.
    For each row, try to replace a larger element with a smaller unused one.
    """
    start = time.time()
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_sum = _sum_entries(best_rows)
    
    # Precompute all used differences
    all_diffs = set()
    for row in best_rows:
        all_diffs |= _row_differences(row)
    
    # Build list of (row_idx, pos_idx) for non-zero positions
    positions = []
    for ri, row in enumerate(best_rows):
        for pi in range(1, len(row)):
            positions.append((ri, pi))
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Pick a random position
        ri, pi = random.choice(positions)
        row = best_rows[ri]
        old_val = row[pi]
        
        # Try to find a smaller value that works
        # The new value must be > row[pi-1] and < row[pi+1] (if exists)
        low = row[pi - 1] + 1
        high = row[pi + 1] - 1 if pi + 1 < len(row) else old_val - 1
        
        if high < low:
            continue
        
        # Try values in random order from low to high (prefer smaller)
        candidates = list(range(low, high + 1))
        random.shuffle(candidates)
        # Sort by value to prefer smaller, but with some randomness
        candidates.sort()
        
        found = False
        for new_val in candidates:
            if new_val >= old_val:
                break  # We only want to reduce
            
            # Check if new_val - row[i] for all i < pi are not in all_diffs
            # and row[j] - new_val for all j > pi are not in all_diffs
            # Also need to remove old differences and add new ones
            new_diffs = set()
            conflict = False
            for i in range(len(row)):
                if i == pi:
                    continue
                d = abs(row[i] - new_val)
                if d == 0:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            # Check against all_diffs minus the old differences of this row
            old_diffs = _row_differences(row)
            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                continue
            
            # Also check internal consistency: new_diffs should be all distinct
            # (they are, since row is strictly increasing and new_val is in valid range)
            # But we need to ensure no duplicate within new_diffs itself
            if len(new_diffs) != len([d for i in range(len(row)) if i != pi]):
                continue
            
            # Apply the change
            row[pi] = new_val
            all_diffs = (all_diffs - old_diffs) | new_diffs
            
            new_scope = _max_entry(best_rows)
            new_sum = _sum_entries(best_rows)
            
            # Accept if scope improved, or same scope but lower sum
            if new_scope < best_scope or (new_scope == best_scope and new_sum < best_sum):
                best_scope = new_scope
                best_sum = new_sum
                found = True
                break
        
        if not found:
            # Occasionally try a random perturbation to escape local minima
            if it % 100 == 0:
                ri2, pi2 = random.choice(positions)
                row2 = best_rows[ri2]
                old_val2 = row2[pi2]
                low2 = row2[pi2 - 1] + 1
                high2 = min(old_val2 + 5, row2[pi2 + 1] - 1 if pi2 + 1 < len(row2) else old_val2 + 5)
                if high2 >= low2:
                    new_val2 = random.randint(low2, high2)
                    if new_val2 != old_val2:
                        old_diffs2 = _row_differences(row2)
                        row2[pi2] = new_val2
                        new_diffs2 = _row_differences(row2)
                        other_diffs2 = all_diffs - old_diffs2
                        if not (new_diffs2 & other_diffs2):
                            all_diffs = (all_diffs - old_diffs2) | new_diffs2
                        else:
                            # Revert
                            row2[pi2] = old_val2
    
    return best_rows

def _simulated_annealing(rows, k, time_limit=20.0):
    """
    Simulated annealing: perturb rows and accept based on energy.
    Energy = scope + 0.1 * sum_entries (to break ties)
    """
    start = time.time()
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_sum = _sum_entries(best_rows)
    best_energy = best_scope + 0.1 * best_sum
    
    current_rows = [list(r) for r in rows]
    current_scope = best_scope
    current_sum = best_sum
    current_energy = best_energy
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    all_diffs = set()
    for row in current_rows:
        all_diffs |= _row_differences(row)
    
    positions = []
    for ri, row in enumerate(current_rows):
        for pi in range(1, len(row)):
            positions.append((ri, pi))
    
    while time.time() - start < time_limit:
        T = max(T * cooling, T_min)
        
        # Pick a random position and try a random change
        ri, pi = random.choice(positions)
        row = current_rows[ri]
        old_val = row[pi]
        
        low = row[pi - 1] + 1
        high = row[pi + 1] - 1 if pi + 1 < len(row) else old_val + 20
        
        if high < low:
            continue
        
        # Random new value, biased toward smaller
        if random.random() < 0.7:
            new_val = random.randint(low, min(high, old_val))
        else:
            new_val = random.randint(max(low, old_val - 20), high)
        
        if new_val == old_val:
            continue
        
        old_diffs = _row_differences(row)
        row[pi] = new_val
        new_diffs = _row_differences(row)
        other_diffs = all_diffs - old_diffs
        
        if new_diffs & other_diffs:
            # Revert
            row[pi] = old_val
            continue
        
        all_diffs = (all_diffs - old_diffs) | new_diffs
        
        new_scope = _max_entry(current_rows)
        new_sum = _sum_entries(current_rows)
        new_energy = new_scope + 0.1 * new_sum
        
        delta = new_energy - current_energy
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            current_scope = new_scope
            current_sum = new_sum
            current_energy = new_energy
            if new_energy < best_energy:
                best_rows = [list(r) for r in current_rows]
                best_scope = new_scope
                best_sum = new_sum
                best_energy = new_energy
        else:
            # Revert
            row[pi] = old_val
            all_diffs = (all_diffs - new_diffs) | old_diffs
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Phase 1: Greedy construction
    rows, used = _greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    
    # Phase 2: Local search
    if time.time() < deadline - 30:
        ls_rows = _local_search(best_rows, k, max_iters=100000, time_limit=15.0)
        ls_scope = _max_entry(ls_rows)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        sa_rows = _simulated_annealing(best_rows, k, time_limit=min(20.0, deadline - time.time() - 5))
        sa_scope = _max_entry(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Another round of local search on the best result
    if time.time() < deadline - 5:
        ls2_rows = _local_search(best_rows, k, max_iters=50000, time_limit=min(10.0, deadline - time.time() - 3))
        ls2_scope = _max_entry(ls2_rows)
        if ls2_scope < best_scope:
            best_rows = ls2_rows
            best_scope = ls2_scope
    
    # Final validation
    if not _is_valid(best_rows):
        # Fallback to greedy
        best_rows, _ = _greedy_construct(n_rows, k)
    
    return best_rows
