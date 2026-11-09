"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows,
    minimizing the largest entry.
    """
    deadline = time.time() + 140  # 140s deadline, leaving 10s safety margin
    total_diffs = n_rows * (k + 1) * k // 2  # total number of differences needed
    
    best_solution = None
    best_scope = float('inf')
    
    def compute_scope(rows):
        return max(max(row) for row in rows) if rows else 0
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def get_used_diffs(rows):
        """Get the set of all differences used."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def greedy_construct(n_rows, k, max_val):
        """Greedy construction with a given max value bound."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                placed = False
                for x in range(1, max_val + 1):
                    if x in row:
                        continue
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        placed = True
                        break
                if not placed:
                    return None
            rows.append(row)
        return rows
    
    def backtrack_construct(n_rows, k, max_val):
        """Backtracking construction with constraint propagation."""
        used = set()
        rows = []
        
        def backtrack(row_idx, current_row, remaining):
            if len(current_row) == k + 1:
                rows.append(current_row)
                if row_idx + 1 == n_rows:
                    return True
                return backtrack(row_idx + 1, [0], remaining)
            
            if not remaining:
                return False
            
            # Try to add next element
            for x in range(current_row[-1] + 1, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if not new_diffs & used:
                    new_used = used | new_diffs
                    new_remaining = remaining - new_diffs
                    if backtrack(row_idx, current_row + [x], new_remaining):
                        return True
            return False
        
        if backtrack(0, [0], set(range(1, max_val + 1))):
            return rows
        return None
    
    def simulated_annealing(n_rows, k, initial_solution, max_time):
        """Simulated annealing to minimize scope."""
        if initial_solution is None:
            return None
        
        current = [list(row) for row in initial_solution]
        current_scope = compute_scope(current)
        best = [list(row) for row in current]
        best_scope = current_scope
        
        temp = max(current_scope * 0.1, 10)
        start_time = time.time()
        no_improve_count = 0
        
        while time.time() - start_time < max_time:
            if time.time() > deadline:
                break
            
            # Generate a random perturbation
            new_solution = [list(row) for row in current]
            
            # Pick a random row and modify it
            row_idx = random.randint(0, n_rows - 1)
            
            # Strategy: try to reduce the max element or shift elements
            strategy = random.random()
            
            if strategy < 0.4:
                # Try to reduce the last element
                row = new_solution[row_idx]
                if len(row) > 1 and row[-1] > row[-2] + 1:
                    old_max = row[-1]
                    # Try smaller values for the last element
                    for new_val in range(row[-2] + 1, old_max):
                        row[-1] = new_val
                        if is_valid(new_solution):
                            break
                        row[-1] = old_max
                    else:
                        row[-1] = old_max
            elif strategy < 0.7:
                # Try to shift a middle element
                row = new_solution[row_idx]
                if len(row) > 2:
                    idx = random.randint(1, len(row) - 2)
                    old_val = row[idx]
                    # Try nearby values
                    for offset in random.sample(range(-5, 6), min(6, 10)):
                        new_val = old_val + offset
                        if new_val > row[idx-1] and new_val < row[idx+1] and new_val > 0:
                            row[idx] = new_val
                            if is_valid(new_solution):
                                break
                            row[idx] = old_val
                    else:
                        row[idx] = old_val
            else:
                # Try to rebuild a row greedily with different starting point
                row = new_solution[row_idx]
                new_row = [0]
                used_other = get_used_diffs([r for i, r in enumerate(new_solution) if i != row_idx])
                for _ in range(k):
                    placed = False
                    for x in range(new_row[-1] + 1, compute_scope(new_solution) + 1):
                        if x in new_row:
                            continue
                        new_diffs = {x - a for a in new_row}
                        if not new_diffs & used_other and not new_diffs & get_used_diffs([new_row]):
                            new_row.append(x)
                            placed = True
                            break
                    if not placed:
                        break
                if len(new_row) == k + 1:
                    new_solution[row_idx] = new_row
            
            new_scope = compute_scope(new_solution)
            delta = new_scope - current_scope
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.1)):
                current = new_solution
                current_scope = new_scope
                no_improve_count = 0
                if current_scope < best_scope:
                    best = [list(row) for row in current]
                    best_scope = current_scope
                    temp = max(temp * 0.95, 1)
            else:
                no_improve_count += 1
            
            # Cool down
            temp = max(temp * 0.999, 1)
        
        return best
    
    # Phase 1: Try to construct a valid solution with increasing max_val
    # Start with a reasonable estimate
    # For 7 rows of 6 elements, we need 7*15 = 105 distinct differences
    # The max value should be at least around 105-120
    
    # First, try greedy with increasing bounds
    for max_val in range(100, 150, 2):
        if time.time() > deadline:
            break
        solution = greedy_construct(n_rows, k, max_val)
        if solution is not None:
            best_solution = solution
            best_scope = compute_scope(solution)
            break
    
    # If greedy didn't work, try backtracking with a higher bound
    if best_solution is None:
        for max_val in range(110, 160, 5):
            if time.time() > deadline:
                break
            solution = backtrack_construct(n_rows, k, max_val)
            if solution is not None:
                best_solution = solution
                best_scope = compute_scope(solution)
                break
    
    # If we still don't have a solution, use a very high bound
    if best_solution is None:
        best_solution = greedy_construct(n_rows, k, 200)
        if best_solution is not None:
            best_scope = compute_scope(best_solution)
    
    # Phase 2: Optimize with simulated annealing
    if best_solution is not None and time.time() < deadline - 10:
        remaining_time = min(deadline - time.time() - 5, 100)
        optimized = simulated_annealing(n_rows, k, best_solution, remaining_time)
        if optimized is not None:
            opt_scope = compute_scope(optimized)
            if opt_scope < best_scope:
                best_solution = optimized
                best_scope = opt_scope
    
    # Final validation
    if best_solution is not None and is_valid(best_solution):
        return best_solution
    
    # Fallback: return a valid solution even if not optimal
    # Use a very high bound to ensure validity
    for max_val in range(150, 300, 10):
        solution = greedy_construct(n_rows, k, max_val)
        if solution is not None and is_valid(solution):
            return solution
    
    # Last resort: construct manually
    # This should never happen, but provide a fallback
    rows = []
    used = set()
    for i in range(n_rows):
        row = [0]
        base = i * 20
        for j in range(1, k + 1):
            val = base + j * 10
            while val in row or any(val - a in used for a in row):
                val += 1
            row.append(val)
            used |= {val - a for a in row}
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
