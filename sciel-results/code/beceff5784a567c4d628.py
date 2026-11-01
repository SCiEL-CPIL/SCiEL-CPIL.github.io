import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Problem: Find n_rows rows of k+1 increasing integers starting at 0
    # such that all differences within each row are distinct across all rows
    # Minimize the scope (largest entry)
    
    # For n_rows=4, k=7: need 4 rows of 8 integers starting at 0
    # Each row contributes C(8,2)=28 differences
    # Total differences: 4*28 = 112
    # All must be distinct, so they must be 1, 2, 3, ..., 112 (or some subset)
    # The maximum possible difference is scope, so we need at least 112 distinct values
    # from 1 to scope, so scope >= 112
    
    def get_all_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def count_total_diffs(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_all_diffs(r))
        return all_diffs
    
    def has_conflicts(rows):
        all_diffs = count_total_diffs(rows)
        return len(all_diffs) != len(set(all_diffs))
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        # Check each row starts at 0, has k+1 elements, strictly increasing
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(1, len(r)):
                if r[i] <= r[i-1]:
                    return False
        # Check all differences are distinct
        all_diffs = count_total_diffs(rows)
        return len(all_diffs) == len(set(all_diffs))
    
    # Greedy construction of a single row given forbidden differences
    def construct_row_greedy(forbidden, max_val):
        row = [0]
        for _ in range(k):
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                valid = True
                for a in row:
                    d = x - a
                    if d in forbidden:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    forbidden.update(x - a for a in row[:-1])
                    found = True
                    break
            if not found:
                return None
        return row
    
    # Main approach: Iterative improvement with backtracking
    # Start with greedy initialization, then try to improve
    
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        found = True
                        break
                    x += 1
            rows.append(row)
        return rows
    
    # Try to construct a valid solution with a given scope limit
    def try_construct_with_scope(max_scope):
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                # Try values from row[-1]+1 up to max_scope
                for x in range(row[-1] + 1, max_scope + 1):
                    # Check if this value creates conflicts
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        # Tentatively add to used_diffs
                        new_diffs = set()
                        for a in row[:-1]:
                            new_diffs.add(x - a)
                        # Check no conflicts within the new set either
                        conflict = False
                        for d in new_diffs:
                            if d in used_diffs:
                                conflict = True
                                break
                        if not conflict:
                            used_diffs.update(new_diffs)
                            found = True
                            break
                        else:
                            # Shouldn't happen if we checked above
                            pass
                if not found:
                    return None
            rows.append(row)
        
        return rows
    
    # Simulated annealing approach
    def simulated_annealing(initial_rows, max_iterations=50000):
        current = [r[:] for r in initial_rows]
        current_scope = get_scope(current)
        best = [r[:] for r in current]
        best_scope = current_scope
        
        T = 10.0  # Initial temperature
        cooling = 0.9995
        
        for it in range(max_iterations):
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            
            # Create a perturbed version
            new_row = current[row_idx][:]
            # Perturb one element
            if len(new_row) > 2:
                pos = random.randint(1, len(new_row) - 1)
                delta = random.randint(-3, 3)
                if delta != 0:
                    new_row[pos] += delta
                    # Fix ordering
                    for j in range(pos + 1, len(new_row)):
                        if new_row[j] <= new_row[j-1]:
                            new_row[j] = new_row[j-1] + 1
                    for j in range(pos - 1, -1, -1):
                        if new_row[j] >= new_row[j+1]:
                            new_row[j] = new_row[j+1] - 1
                    if new_row[0] != 0:
                        new_row[0] = 0
                        for j in range(1, len(new_row)):
                            if new_row[j] <= new_row[j-1]:
                                new_row[j] = new_row[j-1] + 1
            
            # Evaluate
            new_current = [r[:] for r in current]
            new_current[row_idx] = new_row
            new_scope = get_scope(new_current)
            
            # Check validity
            valid = is_valid(new_current)
            if valid:
                new_cost = new_scope
            else:
                # Penalize invalid solutions heavily
                all_diffs = count_total_diffs(new_current)
                n_conflicts = len(all_diffs) - len(set(all_diffs))
                new_cost = new_scope + 1000 * n_conflicts
            
            current_cost = current_scope
            if is_valid(current):
                current_cost = current_scope
            else:
                all_diffs_cur = count_total_diffs(current)
                n_conflicts_cur = len(all_diffs_cur) - len(set(all_diffs_cur))
                current_cost = current_scope + 1000 * n_conflicts_cur
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / max(T, 0.001)):
                current = new_current
                current_scope = new_scope
                if is_valid(current) and new_scope < best_scope:
                    best = [r[:] for r in current]
                    best_scope = new_scope
            
            T *= cooling
        
        return best
    
    # Backtracking search with constraint propagation
    def backtrack_search(max_scope):
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, used):
            if time.time() > deadline - 1.0:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Try next row
                if backtrack(row_idx + 1, [0], used):
                    rows.append(current_row)
                    return True
                return False
            
            # Try adding next value
            start_val = current_row[-1] + 1
            for x in range(start_val, max_scope + 1):
                # Check conflicts
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    # Also check that new differences don't conflict with each other
                    new_diffs = []
                    for a in current_row:
                        new_diffs.append(x - a)
                    if len(new_diffs) != len(set(new_diffs)):
                        continue
                    
                    # Check no conflicts among new diffs
                    conflict = False
                    for d in new_diffs:
                        if d in used:
                            conflict = True
                            break
                    if not conflict:
                        new_used = set(used)
                        new_used.update(new_diffs)
                        if backtrack(row_idx, current_row + [x], new_used):
                            return True
            
            return False
        
        result = backtrack(0, [0], used_diffs)
        if result:
            return rows
        return None
    
    # Start with greedy initialization
    initial = greedy_init()
    initial_scope = get_scope(initial)
    
    # Try backtracking with increasing scope limits
    # The counting bound is 112, best known is 134
    best_solution = None
    best_scope = initial_scope
    
    # First, check if greedy already gives a valid solution
    if is_valid(initial):
        best_solution = initial
        best_scope = initial_scope
    
    # Try backtracking with scope limits
    scope_attempts = [112, 115, 120, 125, 130, 134, 140, 150, 160]
    
    for scope_limit in scope_attempts:
        if time.time() > deadline - 30.0:
            break
        
        if scope_limit <= best_scope and best_solution is not None:
            continue
        
        result = backtrack_search(scope_limit)
        if result and is_valid(result):
            s = get_scope(result)
            if s < best_scope:
                best_solution = result
                best_scope = s
        
        if best_solution is not None and best_scope <= 112:
            break
    
    # If we have a valid solution, try to improve it with simulated annealing
    if best_solution is not None:
        # Run multiple SA runs with different random seeds
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            
            # Create a perturbed initial solution
            perturbed = [r[:] for r in best_solution]
            # Randomly perturb some elements
            for _ in range(5):
                row_idx = random.randint(0, n_rows - 1)
                if len(perturbed[row_idx]) > 2:
                    pos = random.randint(1, len(perturbed[row_idx]) - 1)
                    delta = random.randint(-2, 2)
                    perturbed[row_idx][pos] += delta
                    # Fix ordering
                    for j in range(pos + 1, len(perturbed[row_idx])):
                        if perturbed[row_idx][j] <= perturbed[row_idx][j-1]:
                            perturbed[row_idx][j] = perturbed[row_idx][j-1] + 1
                    if perturbed[row_idx][0] != 0:
                        perturbed[row_idx][0] = 0
                        for j in range(1, len(perturbed[row_idx])):
                            if perturbed[row_idx][j] <= perturbed[row_idx][j-1]:
                                perturbed[row_idx][j] = perturbed[row_idx][j-1] + 1
            
            sa_result = simulated_annealing(perturbed, max_iterations=30000)
            if sa_result is not None and is_valid(sa_result):
                s = get_scope(sa_result)
                if s < best_scope:
                    best_solution = sa_result
                    best_scope = s
    
    # Final: if no valid solution found, use greedy (may be invalid but returns something)
    if best_solution is None:
        best_solution = initial
    
    return best_solution
