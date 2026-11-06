# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    m = len(clauses)
    
    # Precompute variable to clause mapping and literal signs
    var_to_clauses = [[] for _ in range(n)]
    clause_var_info = []
    var_signs = [{} for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        info = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            info.append((v, s))
            var_to_clauses[v].append(c_idx)
            var_signs[v][c_idx] = s
        clause_var_info.append(info)
    
    # Deadline: 325 seconds to be safe against the 330s kill
    deadline = time.time() + 325.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    base_noise = 0.15
    max_noise = 0.7
    restart_threshold = 1500
    
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial clause weights and track unsat clauses
        clause_weights = [0] * m
        unsat_list = []
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_list.append(i)
        
        total_unsat = len(unsat_list)
        
        if total_unsat == 0:
            return assignment
        
        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        step_count = 0
        
        for _ in range(300000):
            step_count += 1
            # Check time every 200 steps to reduce overhead
            if step_count % 200 == 0 and time.time() >= deadline:
                break
            
            if total_unsat == 0:
                return assignment
            
            # Track best solution
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment[:]
            
            if total_unsat < min_unsat_seen:
                min_unsat_seen = total_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > restart_threshold:
                break
            
            # Dynamic noise: increase noise as we get stuck
            progress_ratio = min(steps_since_improvement / restart_threshold, 1.0)
            current_noise = base_noise + (max_noise - base_noise) * progress_ratio
            
            # Use list-based selection for speed
            clause_idx = unsat_list[rng.randrange(total_unsat)]
            clause_info = clause_var_info[clause_idx]
            
            if rng.random() < current_noise:
                # Random walk step
                v, s = clause_info[rng.randrange(3)]
                var_idx = v
            else:
                # Greedy step: Find variable in clause that gives best gain
                best_gain = float('inf')
                best_var_idx = -1
                
                for v, s in clause_info:
                    var_idx = v
                    
                    # Calculate delta in total_unsat
                    delta = 0
                    for c_idx in var_to_clauses[var_idx]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_signs[var_idx][c_idx]
                        
                        # Is the literal satisfied currently?
                        is_sat = (assignment[var_idx] == (c_sign == 1))
                        
                        if is_sat:
                            if old_w == 1:
                                delta += 1
                        else:
                            if old_w == 0:
                                delta -= 1
                    
                    if delta < best_gain:
                        best_gain = delta
                        best_var_idx = var_idx
                        if best_gain == -1:
                            break
                            
                var_idx = best_var_idx
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause weights and unsat structures
            for c_idx in var_to_clauses[var_idx]:
                old_w = clause_weights[c_idx]
                c_sign = var_signs[var_idx][c_idx]
                
                # Is the literal satisfied NOW (after flip)?
                is_sat = (assignment[var_idx] == (c_sign == 1))
                
                if is_sat:
                    clause_weights[c_idx] = old_w + 1
                else:
                    clause_weights[c_idx] = old_w - 1
                    
                # Update unsat structures
                if old_w == 0 and clause_weights[c_idx] > 0:
                    # Remove from unsat_list: find and swap with last
                    for j in range(total_unsat):
                        if unsat_list[j] == c_idx:
                            unsat_list[j] = unsat_list[-1]
                            unsat_list.pop()
                            break
                    total_unsat -= 1
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    unsat_list.append(c_idx)
                    total_unsat += 1

    # Final polish pass with the best assignment found
    if best_assignment is not None and best_unsat_count < float('inf'):
        # Try to improve the best assignment with a focused local search
        assignment = best_assignment[:]
        clause_weights = [0] * m
        unsat_list = []
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_list.append(i)
        
        total_unsat = len(unsat_list)
        
        # Focused local search with lower noise
        for _ in range(80000):
            if time.time() >= deadline - 3:
                break
            if total_unsat == 0:
                return assignment
            
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment[:]
            
            # Greedy step with small random perturbation
            if rng.random() < 0.05:
                # Random step
                var_idx = rng.randrange(n)
            else:
                # Greedy: pick variable with best gain
                best_gain = float('inf')
                best_var_idx = -1
                for v in range(n):
                    delta = 0
                    for c_idx in var_to_clauses[v]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_signs[v][c_idx]
                        is_sat = (assignment[v] == (c_sign == 1))
                        if is_sat:
                            if old_w == 1:
                                delta += 1
                        else:
                            if old_w == 0:
                                delta -= 1
                    if delta < best_gain:
                        best_gain = delta
                        best_var_idx = v
                        if best_gain == -1:
                            break
                var_idx = best_var_idx
            
            assignment[var_idx] = not assignment[var_idx]
            
            for c_idx in var_to_clauses[var_idx]:
                old_w = clause_weights[c_idx]
                c_sign = var_signs[var_idx][c_idx]
                is_sat = (assignment[var_idx] == (c_sign == 1))
                
                if is_sat:
                    clause_weights[c_idx] = old_w + 1
                else:
                    clause_weights[c_idx] = old_w - 1
                    
                if old_w == 0 and clause_weights[c_idx] > 0:
                    for j in range(total_unsat):
                        if unsat_list[j] == c_idx:
                            unsat_list[j] = unsat_list[-1]
                            unsat_list.pop()
                            break
                    total_unsat -= 1
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    unsat_list.append(c_idx)
                    total_unsat += 1
    
    return best_assignment
# EVOLVE-BLOCK-END
