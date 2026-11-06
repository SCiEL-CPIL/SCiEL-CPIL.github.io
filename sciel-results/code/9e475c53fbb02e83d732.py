"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    def init_state(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_unsat_count = [0] * m
        clause_weight = [1.0] * m
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if assignment[vi] != pos:
                    count += 1
            clause_unsat_count[ci] = count
        return assignment, clause_unsat_count, clause_weight
    
    def get_weighted_unsat(clause_unsat_count, clause_weight):
        total = 0.0
        for i in range(m):
            if clause_unsat_count[i] >= 3:
                total += clause_weight[i]
        return total
    
    def find_unsat_clause(clause_unsat_count, clause_weight, rng):
        max_w = -1
        candidates = []
        for ci in range(m):
            if clause_unsat_count[ci] >= 3:
                if clause_weight[ci] > max_w:
                    max_w = clause_weight[ci]
                    candidates = [ci]
                elif clause_weight[ci] == max_w:
                    candidates.append(ci)
        if not candidates:
            return -1
        return rng.choice(candidates)
    
    def flip_var(vi, assignment, clause_unsat_count):
        old_val = assignment[vi]
        assignment[vi] = not old_val
        for ci, pos in var_clauses[vi]:
            if old_val == pos:
                clause_unsat_count[ci] += 1
            else:
                clause_unsat_count[ci] -= 1
    
    def polish_pass(assignment, clause_unsat_count, max_iter=100000):
        assignment = list(assignment)
        clause_unsat_count = list(clause_unsat_count)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            best_var = -1
            best_improvement = 0
            
            candidates = set()
            for ci in range(m):
                if clause_unsat_count[ci] >= 3:
                    for vi, pos in clause_vars[ci]:
                        candidates.add(vi)
            
            for vi in candidates:
                improvement = 0
                for cj, cpos in var_clauses[vi]:
                    if assignment[vi] == cpos:
                        if clause_unsat_count[cj] == 2:
                            improvement -= 1
                    else:
                        if clause_unsat_count[cj] == 3:
                            improvement += 1
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_var = vi
            
            if best_var == -1 or best_improvement <= 0:
                break
            
            flip_var(best_var, assignment, clause_unsat_count)
            
            all_sat = True
            for ci in range(m):
                if clause_unsat_count[ci] >= 3:
                    all_sat = False
                    break
            if all_sat:
                return assignment
        
        return assignment
    
    def smooth_polish(assignment, clause_unsat_count, temperature=0.5, max_iter=50000):
        """
        Smooth surrogate polish: use a soft penalty that rewards partial satisfaction.
        For each clause, the penalty is exp(unsat_count / temperature), which creates
        a smooth landscape where reducing unsat_count from 2->1 is rewarded.
        """
        assignment = list(assignment)
        clause_unsat_count = list(clause_unsat_count)
        
        def smooth_objective():
            total = 0.0
            for ci in range(m):
                if clause_unsat_count[ci] >= 1:
                    total += math.exp(clause_unsat_count[ci] / temperature)
            return total
        
        best_obj = smooth_objective()
        best_assign = list(assignment)
        best_counts = list(clause_unsat_count)
        
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            if best_obj < 1.0:  # All clauses have unsat_count < 1, i.e., 0
                return best_assign
            
            # Find variables in clauses with unsat_count >= 1
            candidates = set()
            for ci in range(m):
                if clause_unsat_count[ci] >= 1:
                    for vi, pos in clause_vars[ci]:
                        candidates.add(vi)
            
            if not candidates:
                return assignment
            
            best_var = -1
            best_delta = 0.0
            
            for vi in candidates:
                # Compute delta in smooth objective if we flip vi
                old_val = assignment[vi]
                delta = 0.0
                for cj, cpos in var_clauses[vi]:
                    old_count = clause_unsat_count[cj]
                    if old_val == cpos:
                        # Literal is true, after flip becomes false, count increases by 1
                        if old_count >= 1:
                            new_count = old_count + 1
                            delta += math.exp(new_count / temperature) - math.exp(old_count / temperature)
                    else:
                        # Literal is false, after flip becomes true, count decreases by 1
                        if old_count >= 1:
                            new_count = max(0, old_count - 1)
                            if new_count == 0:
                                delta += 0.0 - math.exp(old_count / temperature)
                            else:
                                delta += math.exp(new_count / temperature) - math.exp(old_count / temperature)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = vi
            
            if best_var == -1 or best_delta >= 0:
                break
            
            flip_var(best_var, assignment, clause_unsat_count)
            cur_obj = smooth_objective()
            
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_assign = list(assignment)
                best_counts = list(clause_unsat_count)
        
        # Check if all satisfied
        for ci in range(m):
            if best_counts[ci] >= 3:
                return best_assign
        return best_assign
    
    best_assignment = None
    best_weighted_unsat = float('inf')
    
    rng = random.Random(0)
    
    max_restarts = 150
    base_steps = 15000
    max_steps = 200000
    stagnation_limit = 1500
    weight_boost = 2.0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment, clause_unsat_count, clause_weight = init_state(rng)
        current_weighted_unsat = get_weighted_unsat(clause_unsat_count, clause_weight)
        
        if current_weighted_unsat == 0:
            return assignment
        
        if current_weighted_unsat < best_weighted_unsat:
            best_weighted_unsat = current_weighted_unsat
            best_assignment = list(assignment)
        
        steps_in_restart = base_steps + restart * 3000
        steps_in_restart = min(steps_in_restart, max_steps)
        
        steps_without_improvement = 0
        local_best_unsat = current_weighted_unsat
        
        for step in range(steps_in_restart):
            if time.time() > deadline:
                break
            
            unsat_idx = find_unsat_clause(clause_unsat_count, clause_weight, rng)
            if unsat_idx == -1:
                return assignment
            
            cv = clause_vars[unsat_idx]
            
            best_var = -1
            best_delta = float('inf')
            
            for vi, pos in cv:
                delta = 0.0
                for cj, cpos in var_clauses[vi]:
                    if cj == unsat_idx:
                        if clause_unsat_count[cj] == 3:
                            delta -= clause_weight[cj]
                    else:
                        if assignment[vi] == cpos:
                            if clause_unsat_count[cj] == 2:
                                delta += clause_weight[cj]
                        else:
                            if clause_unsat_count[cj] == 3:
                                delta -= clause_weight[cj]
                
                score = delta + rng.random() * 0.01
                if score < best_delta:
                    best_delta = score
                    best_var = vi
            
            if best_var == -1:
                best_var = rng.choice(cv)[0]
            
            flip_var(best_var, assignment, clause_unsat_count)
            
            current_weighted_unsat = get_weighted_unsat(clause_unsat_count, clause_weight)
            
            if current_weighted_unsat < local_best_unsat:
                local_best_unsat = current_weighted_unsat
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            if current_weighted_unsat == 0:
                return assignment
            
            if current_weighted_unsat < best_weighted_unsat:
                best_weighted_unsat = current_weighted_unsat
                best_assignment = list(assignment)
            
            if steps_without_improvement > stagnation_limit:
                for ci in range(m):
                    if clause_unsat_count[ci] >= 3:
                        clause_weight[ci] *= weight_boost
                steps_without_improvement = 0
        
        if time.time() > deadline:
            break
    
    if best_assignment is not None:
        clause_unsat_count = [0] * m
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if best_assignment[vi] != pos:
                    count += 1
            clause_unsat_count[ci] = count
        
        # First: smooth polish to navigate soft landscape
        if time.time() < deadline - 10:
            smooth_result = smooth_polish(best_assignment, clause_unsat_count, temperature=0.5, max_iter=30000)
            # Update best if improved
            smooth_unsat = sum(1 for ci in range(m) if any(
                smooth_result[vi] != pos for vi, pos in clause_vars[ci]
            ) >= 3)
            if smooth_unsat < best_weighted_unsat:
                best_assignment = smooth_result
                # Recompute counts
                clause_unsat_count = [0] * m
                for ci, cv in enumerate(clause_vars):
                    count = 0
                    for vi, pos in cv:
                        if best_assignment[vi] != pos:
                            count += 1
                    clause_unsat_count[ci] = count
        
        # Then: hard polish with exact local search
        if time.time() < deadline - 5:
            polished = polish_pass(best_assignment, clause_unsat_count, max_iter=50000)
            # Check if polished is better
            polished_unsat = sum(1 for ci in range(m) if any(
                polished[vi] != pos for vi, pos in clause_vars[ci]
            ) >= 3)
            if polished_unsat < best_weighted_unsat:
                best_assignment = polished
        
        return best_assignment
    
    rng2 = random.Random(42)
    return [rng2.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
