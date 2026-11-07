import math
import numpy as np
import time
import sys
import random

def construct():
    n = 256
    start_time = time.time()
    timeout = 85.0
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    pp_solution = [1]
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        pp_solution.append(power)
        
    pp_sum = sum(pp_solution)
    
    forced_primes = [p for p in primes if p > 128]
    base_selected = set(forced_primes + [1])
    
    candidates = [i for i in range(1, n + 1) if i not in base_selected]
    num_candidates = len(candidates)
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    
    best_solution = list(pp_solution)
    best_sum = pp_sum
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        c = -np.array(candidates, dtype=float)
        
        conflicts = []
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                if math.gcd(candidates[i], candidates[j]) > 1:
                    conflicts.append((i, j))
        
        num_constraints = len(conflicts)
        
        if num_constraints > 0:
            rows = []
            cols = []
            vals = []
            
            for k, (i, j) in enumerate(conflicts):
                rows.append(k)
                cols.append(i)
                vals.append(1.0)
                rows.append(k)
                cols.append(j)
                vals.append(1.0)
                
            A = csr_matrix((vals, (rows, cols)), shape=(num_constraints, num_candidates))
            
            b_lower = np.zeros(num_constraints)
            b_upper = np.ones(num_constraints)
            
            linear_constraint = LinearConstraint(A, b_lower, b_upper)
            bounds = Bounds(np.zeros(num_candidates), np.ones(num_candidates))
            integrality = np.ones(num_candidates)
            
            init_sol = np.zeros(num_candidates)
            for val in pp_solution:
                if val in cand_to_idx:
                    init_sol[cand_to_idx[val]] = 1.0
            
            elapsed = time.time() - start_time
            if elapsed > timeout - 15.0:
                best_solution = _local_search(best_solution, base_selected, timeout, start_time)
                return _finalize(best_solution)
            
            milp_time_limit = min(40.0, timeout - elapsed - 5.0)
            
            result = milp(
                c, 
                constraints=linear_constraint, 
                integrality=integrality, 
                bounds=bounds,
                initial_solution=init_sol,
                options={'time_limit': milp_time_limit, 'output_flag': 0}
            )
            
            if result.status == 0:
                sol = result.x
                selected_cands = [candidates[i] for i in range(num_candidates) if sol[i] > 0.5]
                full_sol = list(base_selected) + selected_cands
                valid = True
                for i in range(len(full_sol)):
                    for j in range(i+1, len(full_sol)):
                        if math.gcd(full_sol[i], full_sol[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_solution = full_sol
                    best_sum = sum(full_sol)
            else:
                best_solution = list(pp_solution)
                best_sum = pp_sum
        else:
            best_solution = list(base_selected) + candidates
            best_sum = sum(best_solution)
            
    except Exception as e:
        best_solution = list(pp_solution)
        best_sum = pp_sum
    
    elapsed = time.time() - start_time
    if elapsed < timeout - 3.0:
        best_solution = _local_search(best_solution, base_selected, timeout, start_time)
    
    return _finalize(best_solution)


def _local_search(current_sol, base_selected, timeout, start_time):
    current_set = set(current_sol)
    current_sum = sum(current_sol)
    n = 256
    
    conflict_with = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) > 1:
                conflict_with[i].add(j)
                conflict_with[j].add(i)
    
    best_sol = list(current_sol)
    best_sum = current_sum
    
    max_iters = 3000
    iter_count = 0
    
    while iter_count < max_iters:
        elapsed = time.time() - start_time
        if elapsed > timeout - 2.0:
            break
        iter_count += 1
        
        improved = False
        
        for candidate in range(n, 0, -1):
            if candidate in current_set:
                continue
            if not any(x in conflict_with[candidate] for x in current_set):
                current_set.add(candidate)
                current_sum += candidate
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_sol = list(current_set)
                improved = True
                break
        
        if improved:
            continue
        
        removable = [x for x in current_set if x not in base_selected]
        if not removable:
            break
        
        for rem in sorted(removable, key=lambda x: x):
            if len(removable) > 20:
                break
            current_set.remove(rem)
            current_sum -= rem
            
            best_add = None
            best_gain = 0
            for candidate in range(n, 0, -1):
                if candidate in current_set:
                    continue
                if not any(x in conflict_with[candidate] for x in current_set):
                    gain = candidate - rem
                    if gain > best_gain:
                        best_gain = gain
                        best_add = candidate
            
            if best_add is not None and best_gain > 0:
                current_set.add(best_add)
                current_sum += best_add
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_sol = list(current_set)
                improved = True
                break
            else:
                current_set.add(rem)
                current_sum += rem
        
        if not improved:
            if removable:
                perturb_size = min(2, len(removable))
                to_remove = random.sample(removable, perturb_size)
                for rem in to_remove:
                    if rem in current_set:
                        current_set.remove(rem)
                        current_sum -= rem
                for candidate in range(n, 0, -1):
                    if candidate in current_set:
                        continue
                    if not any(x in conflict_with[candidate] for x in current_set):
                        current_set.add(candidate)
                        current_sum += candidate
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_sol = list(current_set)
            else:
                break
    
    return best_sol


def _finalize(sol):
    sol_set = set(sol)
    valid_sol = [x for x in sol_set if 1 <= x <= 256]
    result = []
    for x in sorted(valid_sol):
        ok = True
        for y in result:
            if math.gcd(x, y) != 1:
                ok = False
                break
        if ok:
            result.append(x)
    if 1 not in result:
        result.append(1)
    return [int(x) for x in result]
