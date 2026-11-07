import math
import numpy as np
import time
import sys

def construct():
    n = 256
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # 5s safety margin
    
    # 1. Generate Primes and Prime Powers
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Prime Powers Solution (Baseline)
    pp_solution = [1]
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        pp_solution.append(power)
        
    pp_sum = sum(pp_solution)
    
    # Precompute conflict sets for fast lookup
    conflict_with = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) > 1:
                conflict_with[i].add(j)
                conflict_with[j].add(i)
    
    def is_coprime_with_all(val, sol_set, exclude=None):
        """Check if val is coprime with all in sol_set, optionally excluding one element."""
        for existing in sol_set:
            if exclude is not None and existing == exclude:
                continue
            if val in conflict_with[existing]:
                return False
        return True
    
    # 2. Attempt MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        num_vars = n
        # Objective: Maximize sum -> Minimize -sum
        c = -np.arange(1, n + 1, dtype=float)
        
        # Conflicts
        conflicts = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if math.gcd(i, j) > 1:
                    conflicts.append((i, j))
        
        num_constraints = len(conflicts)
        
        # Sparse matrix construction for efficiency
        rows = []
        cols = []
        vals = []
        
        for k, (i, j) in enumerate(conflicts):
            rows.append(k)
            cols.append(i - 1)
            vals.append(1.0)
            
            rows.append(k)
            cols.append(j - 1)
            vals.append(1.0)
            
        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(num_constraints, num_vars))
        
        b_lower = np.zeros(num_constraints)
        b_upper = np.ones(num_constraints)
        
        linear_constraint = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
        
        # Integrality
        integrality = np.ones(num_vars)
        
        # Warm Start: Prime Powers
        init_sol = np.zeros(num_vars)
        for val in pp_solution:
            init_sol[val - 1] = 1.0
            
        # Check time
        if time.time() - start_time > timeout - 10.0:
            return [int(x) for x in pp_solution]
            
        result = milp(
            c, 
            constraints=linear_constraint, 
            integrality=integrality, 
            bounds=bounds,
            initial_solution=init_sol,
            options={'time_limit': 50, 'output_flag': 0}
        )
        
        best_solution = [int(x) for x in pp_solution]
        best_sum = pp_sum
        
        if result.status == 0:
            sol = result.x
            selected = [int(i + 1) for i in range(num_vars) if sol[i] > 0.5]
            # Verify validity just in case
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if math.gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid and sum(selected) > best_sum:
                best_solution = selected
                best_sum = sum(selected)
                
        # 3. Enhanced Local Search Polish
        if time.time() - start_time < timeout - 3.0:
            current_sol = set(best_solution)
            current_sum = best_sum
            improved = True
            no_improve_count = 0
            
            while improved and time.time() - start_time < timeout - 1.0:
                improved = False
                
                # Try adding a new number (highest value first)
                for new_val in range(n, 0, -1):
                    if new_val in current_sol:
                        continue
                    if is_coprime_with_all(new_val, current_sol):
                        current_sol.add(new_val)
                        current_sum += new_val
                        improved = True
                        break
                
                # Try swapping one element for another (replace smaller with larger)
                if not improved:
                    sorted_sol = sorted(current_sol, reverse=True)
                    found_swap = False
                    for old_val in sorted_sol:
                        # Try replacing old_val with a larger value
                        for new_val in range(n, old_val, -1):
                            if new_val in current_sol:
                                continue
                            if is_coprime_with_all(new_val, current_sol, exclude=old_val):
                                current_sol.remove(old_val)
                                current_sol.add(new_val)
                                current_sum += new_val - old_val
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                
                # Try 2-for-1 swap: replace one element with two non-conflicting elements
                if not improved:
                    sorted_sol = sorted(current_sol, reverse=True)
                    found_2for1 = False
                    for old_val in sorted_sol:
                        # Try replacing old_val with two larger elements that are coprime to each other
                        # and coprime to all others except old_val
                        for new_val1 in range(n, old_val, -1):
                            if new_val1 in current_sol:
                                continue
                            if not is_coprime_with_all(new_val1, current_sol, exclude=old_val):
                                continue
                            # Find a second value that works with new_val1 and all others
                            for new_val2 in range(n, new_val1, -1):
                                if new_val2 in current_sol or new_val2 == new_val1:
                                    continue
                                if math.gcd(new_val1, new_val2) > 1:
                                    continue
                                if not is_coprime_with_all(new_val2, current_sol, exclude=old_val):
                                    continue
                                # Check new_val2 is coprime with new_val1
                                if math.gcd(new_val1, new_val2) > 1:
                                    continue
                                # Both work!
                                gain = new_val1 + new_val2 - old_val
                                if gain > 0:
                                    current_sol.remove(old_val)
                                    current_sol.add(new_val1)
                                    current_sol.add(new_val2)
                                    current_sum += gain
                                    improved = True
                                    found_2for1 = True
                                    break
                            if found_2for1:
                                break
                        if found_2for1:
                            break
                
                # Try 1-for-2 swap: replace two elements with one larger element
                if not improved:
                    sorted_sol = sorted(current_sol, reverse=True)
                    found_1for2 = False
                    for i, old_val1 in enumerate(sorted_sol):
                        for j, old_val2 in enumerate(sorted_sol):
                            if i == j:
                                continue
                            # Try replacing old_val1 and old_val2 with one larger value
                            for new_val in range(n, max(old_val1, old_val2), -1):
                                if new_val in current_sol:
                                    continue
                                # Check if new_val is coprime with all except old_val1 and old_val2
                                if is_coprime_with_all(new_val, current_sol, exclude=old_val1) and \
                                   is_coprime_with_all(new_val, current_sol, exclude=old_val2):
                                    # Need to verify it's coprime with both exclusions handled properly
                                    # Actually, we need new_val coprime with (current_sol - {old_val1, old_val2})
                                    temp_set = set(current_sol)
                                    temp_set.discard(old_val1)
                                    temp_set.discard(old_val2)
                                    if is_coprime_with_all(new_val, temp_set):
                                        gain = new_val - old_val1 - old_val2
                                        if gain > 0:
                                            current_sol.remove(old_val1)
                                            current_sol.remove(old_val2)
                                            current_sol.add(new_val)
                                            current_sum += gain
                                            improved = True
                                            found_1for2 = True
                                            break
                                if found_1for2:
                                    break
                        if found_1for2:
                            break
                
                # Track no-improvement streaks
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > 3:
                        break
                else:
                    no_improve_count = 0
            
            if current_sum > best_sum:
                best_solution = list(current_sol)
                best_sum = current_sum
                
        # 4. Final high-precision polish: try all possible single additions and swaps one more time
        if time.time() - start_time < timeout - 0.5:
            current_sol = set(best_solution)
            current_sum = best_sum
            # One more pass of adding and swapping
            for new_val in range(n, 0, -1):
                if new_val not in current_sol and is_coprime_with_all(new_val, current_sol):
                    current_sol.add(new_val)
                    current_sum += new_val
            
            # Try all single swaps
            sorted_sol = sorted(current_sol, reverse=True)
            for old_val in sorted_sol:
                for new_val in range(n, old_val, -1):
                    if new_val in current_sol:
                        continue
                    if is_coprime_with_all(new_val, current_sol, exclude=old_val):
                        current_sol.remove(old_val)
                        current_sol.add(new_val)
                        current_sum += new_val - old_val
                        break
            
            if current_sum > best_sum:
                best_solution = list(current_sol)
                best_sum = current_sum
                
        return [int(x) for x in best_solution]
            
    except Exception as e:
        # Fallback to Prime Powers if anything goes wrong
        return [int(x) for x in pp_solution]
