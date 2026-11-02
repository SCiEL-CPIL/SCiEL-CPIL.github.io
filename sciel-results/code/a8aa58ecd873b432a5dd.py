import time
import random
import math

def construct(n_rows, k):
    deadline = time.time() + 115
    best_rows = None
    best_scope = float('inf')
    
    def _scope(rows):
        return max(max(row) for row in rows)
    
    def _compute_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    def _greedy_construct(n_rows, k, seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        
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
    
    def _biased_sa(rows, n_rows, k, deadline, seed=42):
        """
        Simulated annealing with a strong bias towards reducing the scope.
        Moves that decrease the scope are always accepted.
        Moves that increase the scope are accepted with probability exp(-delta/T),
        but the temperature is kept higher for longer to allow exploration.
        Also, we bias the selection of the position to move towards the end of the row
        (larger values) as they contribute more to the scope.
        """
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)
        current_rows = [list(r) for r in rows]
        current_scope = _scope(current_rows)
        
        rng = random.Random(seed)
        start_temp = 100.0
        end_temp = 0.5
        cooling_rate = 0.999
        
        # Precompute diffs for current
        row_diffs = [_compute_diffs(r) for r in current_rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        temp = start_temp
        iterations = 0
        max_iterations = 100000
        
        while time.time() < deadline - 5 and temp > end_temp and iterations < max_iterations:
            iterations += 1
            
            # Pick a random row
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            
            # Bias towards positions with larger values (later in the list)
            # We pick a position with probability proportional to (index + 1)
            # Or simply pick from the latter half more often?
            # Let's just pick uniformly first, but maybe weight by value?
            # Simpler: pick a random position, but if it's in the first half, 
            # with 50% chance pick a position in the second half.
            pos = rng.randrange(1, len(row))
            if pos < len(row) // 2 and rng.random() < 0.5:
                pos = rng.randrange(len(row) // 2, len(row))
            
            old_val = row[pos]
            
            # Bias the delta towards negative (decreasing values)
            # 70% chance to try a negative delta, 30% positive
            if rng.random() < 0.7:
                delta = -rng.randint(1, 10)
            else:
                delta = rng.randint(1, 10)
            
            new_val = old_val + delta
            
            # Check validity
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Compute old diffs for this row
            old_diffs = _compute_diffs(row)
            
            # Apply change
            row[pos] = new_val
            new_diffs = _compute_diffs(row)
            
            # Check for internal duplicates
            expected_diff_count = len(row) * (len(row) - 1) // 2
            if len(new_diffs) != expected_diff_count:
                row[pos] = old_val
                continue
            
            # Check conflicts with other rows
            other_diffs = all_diffs - old_diffs
            if new_diffs & other_diffs:
                row[pos] = old_val
                continue
            
            # Valid move - compute new scope
            new_scope = _scope(current_rows)
            delta_scope = new_scope - current_scope
            
            # Accept if better, or with probability exp(-delta_scope/temp) if worse
            if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temp):
                # Accept the move
                all_diffs = (all_diffs - old_diffs) | new_diffs
                row_diffs[ri] = new_diffs
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    # Reheat slightly on improvement?
                    # temp = max(temp, start_temp * 0.5) 
            else:
                # Reject the move
                row[pos] = old_val
            
            temp *= cooling_rate
        
        return best_rows, best_scope

    def _local_polish(rows, n_rows, k, deadline):
        """Final local polish with small moves to find immediate improvements."""
        best_rows = [list(r) for r in rows]
        best_scope = _scope(best_rows)
        rng = random.Random(123)
        
        row_diffs = [_compute_diffs(r) for r in best_rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        # Try many small moves
        for _ in range(50000):
            if time.time() >= deadline - 2:
                break
            
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            if len(row) <= 1:
                continue
            
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            
            # Try small deltas
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                
                old_diffs = _compute_diffs(row)
                row[pos] = new_val
                new_diffs = _compute_diffs(row)
                
                expected_diff_count = len(row) * (len(row) - 1) // 2
                if len(new_diffs) != expected_diff_count:
                    row[pos] = old_val
                    continue
                
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    row[pos] = old_val
                    continue
                
                new_scope = _scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    all_diffs = (all_diffs - old_diffs) | new_diffs
                    row_diffs[ri] = new_diffs
                    break
                else:
                    row[pos] = old_val
        
        return best_rows, best_scope

    # Phase 1: Greedy construction to get a valid starting point
    for seed in range(50):
        if time.time() >= deadline - 20:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]

    # Phase 2: Biased Simulated Annealing
    if best_rows is not None:
        # Run SA with a few different seeds to diversify
        for seed in range(5):
            if time.time() >= deadline - 10:
                break
            candidate_rows, candidate_scope = _biased_sa(best_rows, n_rows, k, deadline, seed=seed)
            if candidate_scope < best_scope:
                best_scope = candidate_scope
                best_rows = candidate_rows

    # Phase 3: Local Polish
    if best_rows is not None and time.time() < deadline - 2:
        best_rows, best_scope = _local_polish(best_rows, n_rows, k, deadline)

    if best_rows is None:
        # Fallback: Construct a valid solution with large scope
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                v = row[-1] + 1
                while True:
                    valid = True
                    for prev in row:
                        d = v - prev
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for prev in row[:-1]:
                            used_diffs.add(v - prev)
                        break
                    v += 1
            rows.append(row)
        return rows

    return best_rows
