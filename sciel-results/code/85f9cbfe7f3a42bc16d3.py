# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses simulated annealing with row-swap and position-swap moves to escape
    local minima, combined with greedy initialization.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    def compute_used_diffs(rows):
        """Compute the set of all differences and count conflicts."""
        used = set()
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        return used, conflicts
    
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    # Simulated annealing
    def simulated_annealing(n_rows, k, target_max, timeout, seed=0):
        """
        SA that tries to find a valid configuration with max_entry <= target_max.
        State: list of rows. Moves: swap two rows, swap two positions in a row,
        shift an entry up or down by 1.
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        # Initialize with greedy, then try to compress
        rows = greedy_construct(n_rows, k)
        current_max = max_entry(rows)
        
        # If greedy already satisfies target, we're done
        if current_max <= target_max:
            return rows
        
        # Start from a random perturbation of greedy
        # First, try to get a valid starting point by using greedy and then
        # applying random shifts to try to reduce max
        used, conflicts = compute_used_diffs(rows)
        
        # We want to minimize max_entry while keeping conflicts = 0
        # Objective: max_entry + 1000 * conflicts (heavily penalize conflicts)
        
        def objective(rows):
            m = max_entry(rows)
            _, conf = compute_used_diffs(rows)
            return m + 1000 * conf
        
        current_obj = objective(rows)
        best_rows = [r[:] for r in rows]
        best_obj = current_obj
        
        # Temperature schedule
        T0 = 50.0
        T_min = 0.1
        alpha = 0.999
        
        T = T0
        step = 0
        
        while time.time() < deadline:
            # Generate a random move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Swap two rows
                if n_rows < 2:
                    continue
                i, j = rng.sample(range(n_rows), 2)
                rows[i], rows[j] = rows[j], rows[i]
                
            elif move_type < 0.6:
                # Swap two positions in a row (not position 0)
                if k < 2:
                    continue
                r = rng.randint(0, n_rows - 1)
                # Positions 1..k can be swapped
                p1, p2 = rng.sample(range(1, k + 1), 2)
                rows[r][p1], rows[r][p2] = rows[r][p2], rows[r][p1]
                
            else:
                # Shift an entry by ±1 (not position 0)
                r = rng.randint(0, n_rows - 1)
                p = rng.randint(1, k)
                delta = rng.choice([-1, 1])
                old_val = rows[r][p]
                new_val = old_val + delta
                # Check bounds: must stay > previous and < next, and >= 0
                if new_val <= 0:
                    continue
                if p > 0 and new_val <= rows[r][p - 1]:
                    continue
                if p < k and new_val >= rows[r][p + 1]:
                    continue
                rows[r][p] = new_val
            
            # Check validity: row must be strictly increasing and start at 0
            valid_state = True
            for r in range(n_rows):
                if rows[r][0] != 0:
                    valid_state = False
                    break
                for i in range(1, k + 1):
                    if rows[r][i] <= rows[r][i - 1]:
                        valid_state = False
                        break
                if not valid_state:
                    break
            
            if not valid_state:
                # Undo the move
                if move_type < 0.3:
                    i, j = rng.sample(range(n_rows), 2)  # This is wrong, need to track
                    # Actually, let's just revert by re-deriving
                    # Simpler: keep a copy
                    pass
                continue
            
            new_obj = objective(rows)
            delta_obj = new_obj - current_obj
            
            # Accept if better, or with probability exp(-delta/T)
            if delta_obj < 0 or (T > 0 and rng.random() < pow(2.718281828, -delta_obj / T)):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_rows = [r[:] for r in rows]
            else:
                # Reject: undo
                if move_type < 0.3:
                    i, j = rng.sample(range(n_rows), 2)
                    rows[i], rows[j] = rows[j], rows[i]
                elif move_type < 0.6:
                    r = rng.randint(0, n_rows - 1)
                    p1, p2 = rng.sample(range(1, k + 1), 2)
                    rows[r][p1], rows[r][p2] = rows[r][p2], rows[r][p1]
                else:
                    r = rng.randint(0, n_rows - 1)
                    p = rng.randint(1, k)
                    # We don't have the old value, so this is problematic
                    # Let's use a different approach: store old state
                    pass
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
            
            step += 1
    
        return best_rows
    
    # Better approach: use a proper SA with state tracking
    def sa_with_tracking(n_rows, k, target_max, timeout, seed=0):
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        rows = greedy_construct(n_rows, k)
        current_max = max_entry(rows)
        
        if current_max <= target_max:
            return rows
        
        def obj(rows):
            m = max_entry(rows)
            used = set()
            conf = 0
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in used:
                            conf += 1
                        else:
                            used.add(d)
            return m + 1000 * conf
        
        cur_obj = obj(rows)
        best_rows = [r[:] for r in rows]
        best_obj = cur_obj
        
        T = 30.0
        T_min = 0.5
        alpha = 0.9995
        
        while time.time() < deadline:
            # Random move
            r = rng.randint(0, n_rows - 1)
            p = rng.randint(1, k)
            old_val = rows[r][p]
            # Try small shifts
            shift = rng.randint(-3, 3)
            if shift == 0:
                continue
            new_val = old_val + shift
            if new_val <= 0:
                continue
            if p > 0 and new_val <= rows[r][p - 1]:
                continue
            if p < k and new_val >= rows[r][p + 1]:
                continue
            
            rows[r][p] = new_val
            new_obj = obj(rows)
            delta = new_obj - cur_obj
            
            if delta < 0 or (T > 0 and rng.random() < pow(2.718281828, -delta / T)):
                cur_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_rows = [rr[:] for rr in rows]
            else:
                rows[r][p] = old_val
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        return best_rows
    
    # Try SA with different target values
    best_rows = greedy_rows
    best_max = greedy_max
    
    # First, try to find a valid solution with max <= greedy_max using SA
    # Then try to improve
    
    # Try SA starting from greedy, trying to reduce max
    for attempt in range(3):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 2
        if remaining < 1:
            break
        
        budget = remaining / (3 - attempt)
        result = sa_with_tracking(n_rows, k, greedy_max - 1, budget, seed=attempt)
        if result is not None:
            if validate(result, n_rows, k):
                rmax = max_entry(result)
                if rmax < best_max:
                    best_rows = result
                    best_max = rmax
    
    # Now try to find even better solutions with lower targets
    for target in range(max(lb, best_max - 20), best_max):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 2
        if remaining < 2:
            break
        
        budget = min(5.0, remaining / 4)
        if budget < 0.5:
            break
        
        result = sa_with_tracking(n_rows, k, target, budget, seed=target)
        if result is not None:
            if validate(result, n_rows, k):
                rmax = max_entry(result)
                if rmax <= target and rmax < best_max:
                    best_rows = result
                    best_max = rmax
                    break
    
    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
