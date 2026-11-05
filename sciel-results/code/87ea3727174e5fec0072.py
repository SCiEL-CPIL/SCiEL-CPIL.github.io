import time
import random
import math
import bisect
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Phase 1: Get an initial feasible solution via randomized greedy
    best_rows = None
    best_scope = float('inf')
    
    for attempt in range(5):
        if time.time() > deadline - 80:
            break
        rows = _greedy_construct(n_rows, k, seed=attempt * 137 + 42)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    if best_rows is None:
        # Fallback: very loose greedy
        best_rows = _greedy_construct(n_rows, k, seed=999)
        if best_rows is None:
            # Last resort: just return something valid-ish
            best_rows = [[0, 1, 3, 6, 10, 15, 21] for _ in range(n_rows)]
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated annealing to reduce scope
    if time.time() < deadline - 10:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _validate(rows, n_rows, k):
    """Check if rows form a valid difference triangle set."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def _greedy_construct(n_rows, k, seed):
    """Randomized greedy for all rows."""
    rng = random.Random(seed)
    used = set()
    rows = []

    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        if row is None:
            return None
        rows.append(row)
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                used.add(d)

    return rows

def _greedy_row(used_diffs, k, rng):
    """Best-fit greedy for a single row."""
    row = [0]

    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 300

        x = last_x + 1
        while len(valid_xs) < 30 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, sorted(used_diffs), x)
                scored_xs.append((score, x))
            x += 1

        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 30:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, sorted(used_diffs), x)
                    scored_xs.append((score, x))
                x += 1

        if not valid_xs:
            return None

        scored_xs.sort(key=lambda t: t[0])
        top_n = min(5, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)

    return row

def _packing_score(new_diffs, used_sorted, x):
    """Score a candidate x by how well its differences pack into used space."""
    if not used_sorted:
        return x * 1.0

    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist

    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """
    Simulated annealing to minimize the maximum element (scope).
    
    State: list of n_rows rows, each with k+1 increasing integers starting at 0.
    Moves:
      1. Replace one element in a row (maintaining order)
      2. Swap two rows
      3. Perturb an entire row
    Cost: scope (max element) + penalty for invalid differences
    """
    rng = random.Random(2025)
    
    # Initial state
    state = [list(r) for r in initial_rows]
    best_state = [list(r) for r in state]
    best_cost = _cost(state, n_rows, k)
    
    # Temperature schedule
    T_start = max(10.0, best_cost * 0.1)
    T_end = 0.1
    total_time = deadline - start_time - 5
    steps_per_second = 5000  # target steps per second
    
    # Precompute used diffs for current state
    def compute_diffs(rows):
        diffs = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i+1, n):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def cost_func(rows):
        """Cost = scope + penalty for duplicate diffs."""
        scope = _scope(rows)
        diffs = compute_diffs(rows)
        # Count duplicates
        total_diffs = 0
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i+1, n):
                    total_diffs += 1
        num_unique = len(diffs)
        penalty = max(0, total_diffs - num_unique) * 10
        return scope + penalty
    
    current_cost = cost_func(state)
    best_cost = min(best_cost, current_cost)
    
    # Annealing loop
    step = 0
    t = start_time
    T = T_start
    
    while t < deadline - 5:
        # Update temperature (exponential cooling)
        progress = (t - start_time) / total_time if total_time > 0 else 1
        T = T_start * (T_end / T_start) ** progress
        
        # Choose a random move
        move_type = rng.random()
        
        new_state = [list(r) for r in state]
        
        if move_type < 0.5:
            # Move 1: Replace one element in a random row
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # positions 1..k (position 0 is always 0)
            
            # Determine valid range for new value
            min_val = new_state[row_idx][pos-1] + 1 if pos > 0 else 1
            max_val = new_state[row_idx][pos+1] - 1 if pos < k else 10000
            
            if min_val > max_val:
                # Can't change this position, try another
                continue
            
            # Sample new value
            new_val = rng.randint(min_val, min(max_val, min_val + 200))
            new_state[row_idx][pos] = new_val
            
            # Ensure strict increasing
            valid = True
            for i in range(1, len(new_state[row_idx])):
                if new_state[row_idx][i] <= new_state[row_idx][i-1]:
                    valid = False
                    break
            if not valid:
                continue
                
        elif move_type < 0.8:
            # Move 2: Perturb an entire row
            row_idx = rng.randint(0, n_rows - 1)
            used_diffs = compute_diffs([r for i, r in enumerate(state) if i != row_idx])
            
            # Try to rebuild this row with some randomness
            new_row = [0]
            success = True
            for _ in range(k):
                last_x = new_row[-1]
                valid_xs = []
                for x in range(last_x + 1, last_x + 500):
                    new_diffs = {x - a for a in new_row}
                    if not (new_diffs & used_diffs):
                        valid_xs.append(x)
                    if len(valid_xs) > 20:
                        break
                
                if not valid_xs:
                    success = False
                    break
                
                # Random choice with bias toward smaller values
                weight = 1.0 / (1 + valid_xs.index(x)) if x in valid_xs else 1
                chosen = rng.choice(valid_xs[:min(10, len(valid_xs))])
                new_row.append(chosen)
                used_diffs.update({chosen - a for a in new_row[:-1]})
            
            if success and len(new_row) == k + 1:
                new_state[row_idx] = new_row
            else:
                continue
                
        else:
            # Move 3: Swap two rows
            i, j = rng.sample(range(n_rows), 2)
            new_state[i], new_state[j] = new_state[j], new_state[i]
        
        # Evaluate new state
        new_cost = cost_func(new_state)
        delta = new_cost - current_cost
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            state = new_state
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = [list(r) for r in state]
        
        step += 1
        t = time.time()
        
        # Periodic check
        if step % 10000 == 0 and t > deadline - 10:
            break
    
    # Final validation
    if _validate(best_state, n_rows, k):
        return best_state
    else:
        # Return initial if best is invalid
        if _validate([list(r) for r in initial_rows], n_rows, k):
            return [list(r) for r in initial_rows]
        return None
