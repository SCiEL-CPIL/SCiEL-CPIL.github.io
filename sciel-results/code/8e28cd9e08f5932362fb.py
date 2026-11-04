<<<<<<< SEARCH
import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s kill
    
    # Theoretical lower bound on the max element
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Try backtracking with constraint propagation
    # This is more principled than pure greedy
    max_scope_to_try = min(200, lower_bound + 30)
    
    # Try different scope bounds with backtracking
    for scope in range(lower_bound, max_scope_to_try + 1):
        if time.time() > deadline:
            break
            
        result = _backtracking_search(n_rows, k, scope, start_time, deadline)
        if result is not None:
            scope_val = max(max(row) for row in result) if result else 0
            if scope_val < best_scope:
                best_scope = scope_val
                best_rows = result
            # If we found a solution at the lower bound, stop
            if scope_val == lower_bound:
                break
        elif time.time() > deadline:
            break
    
    # Phase 2: If backtracking didn't find good solutions, use randomized greedy as fallback
    if best_rows is None or best_scope > max_scope_to_try:
        seed = 0
        while time.time() < deadline:
            rows = _randomized_greedy(n_rows, k, seed)
            scope = max(max(row) for row in rows) if rows else float('inf')
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope == lower_bound:
                    break
            seed += 1
    
    # If still no solution, use simple greedy
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)
    
    return best_rows

def _backtracking_search(n_rows, k, max_scope, start_time, deadline):
    """
    Backtracking search with constraint propagation.
    Uses a degree heuristic to minimize future constraint violations.
    """
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, col_idx, current_row):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if col_idx == k + 1:
            # Row complete, move to next row
            rows.append(list(current_row))
            success = backtrack(row_idx + 1, 1, [0])
            if success:
                return True
            rows.pop()
            return False
        
        # Try to place an element at position col_idx in current_row
        # The element must be > current_row[col_idx-1] and <= max_scope
        start_val = current_row[col_idx - 1] + 1
        end_val = min(max_scope, current_row[col_idx - 1] + (max_scope - current_row[col_idx - 1]))
        
        # Collect valid candidates
        valid_candidates = []
        for val in range(start_val, end_val + 1):
            # Check if adding val creates any duplicate differences
            new_diffs = set()
            for i in range(col_idx):
                d = val - current_row[i]
                if d <= 0:
                    continue
                if d in used_diffs:
                    break
                new_diffs.add(d)
            else:
                # All differences are valid
                # Compute "degree": how many future difference slots remain
                # A lower degree means more constraints are satisfied
                # We want to minimize the number of newly used differences that
                # block future possibilities
                valid_candidates.append((val, new_diffs))
        
        if not valid_candidates:
            return False
        
        # Sort by degree heuristic: prefer values that use "less constrained" differences
        # A difference d is "less constrained" if it's larger (more room for future diffs)
        # Or we can prefer values that create differences that are less likely to be needed
        # Simple heuristic: prefer smaller values first (greedy), but with some randomness
        # Better: sort by the number of future rows that would be affected
        
        # For now, use a simple heuristic: try smaller values first
        # But add some randomization to escape local optima
        random.shuffle(valid_candidates)
        
        for val, new_diffs in valid_candidates:
            # Check if this leads to a solution
            used_diffs.update(new_diffs)
            current_row.append(val)
            
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            if time.time() > deadline:
                return False
        
        return False
    
    # Start backtracking
    if backtrack(0, 1, [0]):
        return list(rows)
    return None

def _randomized_greedy(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            # Find valid x's
            valid_xs = []
            last_x = row[-1]
            limit = 500
            
            x = last_x + 1
            while len(valid_xs) < 20 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 20:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            if not valid_xs:
                # Fallback: just pick the next integer
                chosen_x = last_x + 1
            else:
                chosen_x = rng.choice(valid_xs)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            
        rows.append(row)
        
    return rows

def _simple_greedy(n_rows, k):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        rows.append(row)
    
    return rows
=======
import time
import random
import math

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows) if rows else 0

def _all_diffs(rows):
    """Return the set of all positive differences within rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, rng=None):
    """Fast greedy construction: place smallest valid value at each position."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        rows.append(row)
    return rows

def _generate_candidate_row(k, other_rows, rng):
    """Generate a random valid row given constraints from other rows."""
    other_diffs = set()
    for row in other_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                other_diffs.add(row[j] - row[i])
    
    row = [0]
    for _ in range(k):
        last = row[-1]
        # Try random values with increasing offset
        attempts = 0
        while attempts < 500:
            x = last + 1 + rng.randint(0, max(1, last // 2))
            new_diffs = {x - a for a in row}
            if not (new_diffs & other_diffs) and not (new_diffs & set()):
                row.append(x)
                break
            attempts += 1
        else:
            # Fallback: greedy
            x = last + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & other_diffs):
                    row.append(x)
                    break
                x += 1
    return row

def _simulated_annealing(n_rows, k, deadline, rng):
    """
    Simulated annealing that rebuilds individual rows to escape local optima.
    State: list of n_rows rows, each of length k+1 starting at 0.
    Move: rebuild one row randomly, accepting if it doesn't increase scope
          or with probability exp(-delta/T).
    """
    # Initial solution from greedy
    rows = _greedy_construct(n_rows, k, rng)
    current_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    # Temperature schedule
    t0 = max(1.0, current_scope * 0.3)
    t_end = 0.1
    num_steps = 50000
    cooling = (t_end / t0) ** (1.0 / max(1, num_steps - 1))
    
    step = 0
    while time.time() < deadline:
        T = t0 * (cooling ** step)
        step += 1
        
        # Pick a random row to rebuild
        row_idx = rng.randint(0, n_rows - 1)
        other_rows = [rows[i] for i in range(n_rows) if i != row_idx]
        
        # Generate a new candidate row
        new_row = _generate_candidate_row(k, other_rows, rng)
        
        # Compute new scope
        other_max = max(max(r) for r in other_rows) if other_rows else 0
        new_row_max = max(new_row)
        new_scope = max(other_max, new_row_max)
        
        delta = new_scope - current_scope
        
        # Acceptance criterion
        if delta <= 0:
            # Accept improvement
            rows[row_idx] = new_row
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Accept worse with probability exp(-delta/T)
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    rows[row_idx] = new_row
                    current_scope = new_scope
        
        # Periodically try a more aggressive move: rebuild two rows
        if step % 100 == 0 and time.time() < deadline:
            if n_rows >= 2:
                r1 = rng.randint(0, n_rows - 1)
                r2 = rng.randint(0, n_rows - 1)
                if r2 == r1:
                    r2 = (r1 + 1) % n_rows
                other_rows = [rows[i] for i in range(n_rows) if i not in (r1, r2)]
                
                new_row1 = _generate_candidate_row(k, other_rows, rng)
                new_row2 = _generate_candidate_row(k, other_rows, rng)
                
                other_max = max(max(r) for r in other_rows) if other_rows else 0
                new_scope = max(other_max, max(new_row1), max(new_row2))
                delta = new_scope - current_scope
                
                if delta <= 0:
                    rows[r1] = new_row1
                    rows[r2] = new_row2
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                else:
                    if T > 0:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            rows[r1] = new_row1
                            rows[r2] = new_row2
                            current_scope = new_scope
        
        # Check for early termination
        if best_scope <= n_rows * (k + 1) * k // 2:
            break
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    rng = random.Random(42)
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k, rng)
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Phase 2: Simulated annealing for improvement
    if time.time() < deadline - 5:
        sa_rows = _simulated_annealing(n_rows, k, deadline - 5, rng)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
    
    # Phase 3: If time remains, try additional SA restarts with different seeds
    remaining = deadline - time.time()
    if remaining > 10:
        for seed in range(1, 10):
            if time.time() > deadline - 2:
                break
            rng2 = random.Random(seed * 1000 + 7)
            sa_rows2 = _simulated_annealing(n_rows, k, deadline - 2, rng2)
            sa_scope2 = _scope(sa_rows2)
            if sa_scope2 < best_scope:
                best_scope = sa_scope2
                best_rows = sa_rows2
    
    return best_rows
>>>>>>> REPLACE
