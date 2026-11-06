# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin

    # Convert clauses to 0-based signed literals
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    m = len(clauses_0based)

    # Build data structures
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    # === PREPROCESSING: Pure Literal Elimination + Unit Propagation ===
    # This fundamentally changes the problem representation by fixing variables
    fixed = {}  # var -> bool value
    remaining_vars = set(range(n))
    remaining_clauses = list(clauses_0based)
    
    # Iterative pure literal elimination and unit propagation
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        
        # Build literal occurrence map for remaining clauses
        # pos_count[var] = number of clauses where var appears positively (and is not fixed)
        # neg_count[var] = number of clauses where var appears negatively
        pos_count = {}
        neg_count = {}
        for ci, clause in enumerate(remaining_clauses):
            for var, sign in clause:
                if var not in remaining_vars:
                    continue
                if sign > 0:
                    pos_count[var] = pos_count.get(var, 0) + 1
                else:
                    neg_count[var] = neg_count.get(var, 0) + 1
        
        # Pure literal elimination: if a variable appears only in one polarity, assign it
        for var in list(remaining_vars):
            if var not in pos_count and var not in neg_count:
                continue
            if var in pos_count and var not in neg_count:
                # Only appears positively -> set to True
                fixed[var] = True
                remaining_vars.discard(var)
                changed = True
            elif var not in pos_count and var in neg_count:
                # Only appears negatively -> set to False
                fixed[var] = False
                remaining_vars.discard(var)
                changed = True
        
        if changed:
            # Remove satisfied clauses, simplify remaining
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                        # else: literal is false, remove it
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
                    # If simplified is empty, this is a contradiction (shouldn't happen for SAT instances)
            remaining_clauses = new_clauses
            continue
        
        # Unit propagation: find clauses with exactly one unassigned literal
        unit_clauses = []
        for clause in remaining_clauses:
            unassigned = [(var, sign) for var, sign in clause if var in remaining_vars]
            if len(unassigned) == 1:
                unit_clauses.append(unassigned[0])
        
        if unit_clauses:
            for var, sign in unit_clauses:
                if var in remaining_vars:
                    fixed[var] = (sign > 0)
                    remaining_vars.discard(var)
            changed = True
            # Remove satisfied clauses, simplify
            new_clauses = []
            for clause in remaining_clauses:
                sat = False
                simplified = []
                for var, sign in clause:
                    if var in fixed:
                        if fixed[var] == (sign > 0):
                            sat = True
                            break
                    else:
                        simplified.append((var, sign))
                if not sat:
                    if len(simplified) > 0:
                        new_clauses.append(simplified)
            remaining_clauses = new_clauses
    
    n_reduced = len(remaining_vars)
    m_reduced = len(remaining_clauses)
    
    # If all variables are fixed, check if all clauses are satisfied
    if n_reduced == 0:
        # All fixed, verify
        for clause in remaining_clauses:
            sat = False
            for var, sign in clause:
                if var in fixed and fixed[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                # Should not happen for satisfiable instances
                break
        # Build full assignment
        full_assignment = [False] * n
        for var in range(n):
            if var in fixed:
                full_assignment[var] = fixed[var]
            else:
                full_assignment[var] = False
        return full_assignment
    
    # Map reduced variables to indices
    reduced_var_list = list(remaining_vars)
    var_to_reduced_idx = {var: i for i, var in enumerate(reduced_var_list)}
    
    # Convert remaining clauses to use reduced indices
    reduced_clauses = []
    for clause in remaining_clauses:
        reduced_clause = []
        for var, sign in clause:
            if var in remaining_vars:
                reduced_clause.append((var_to_reduced_idx[var], sign))
        if reduced_clause:
            reduced_clauses.append(reduced_clause)
    
    n_r = n_reduced
    m_r = len(reduced_clauses)
    
    # Build var_clauses for reduced instance
    reduced_var_clauses = [[] for _ in range(n_r)]
    for ci, clause in enumerate(reduced_clauses):
        for var, sign in clause:
            reduced_var_clauses[var].append((ci, sign))
    
    def _check_satisfied(assignment, clauses_list, var_clauses_list):
        """Check if all clauses are satisfied."""
        for clause in clauses_list:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def _count_unsat(assignment, clauses_list):
        """Count unsatisfied clauses."""
        count = 0
        for clause in clauses_list:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def _walksat_reduced(rng, max_restarts=100, steps_per_restart=5000):
        """Fast WalkSAT with clause-literal counts for O(1) flip updates."""
        best_assignment = None
        best_unsat = m_r + 1
        
        # Precompute: for each clause, list of (var, sign)
        # For each variable, list of (clause_idx, sign)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initial assignment
            assignment = [rng.random() < 0.5 for _ in range(n_r)]
            
            # Compute sat_count[ci] = number of satisfying literals in clause ci
            sat_count = np.zeros(m_r, dtype=np.int32)
            for ci, clause in enumerate(reduced_clauses):
                cnt = 0
                for var, sign in clause:
                    if assignment[var] == (sign > 0):
                        cnt += 1
                sat_count[ci] = cnt
            
            clause_unsat = (sat_count == 0)
            unsat_count = int(np.sum(clause_unsat))
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # WalkSAT with noise schedule
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return best_assignment
                
                if unsat_count == 0:
                    return assignment
                
                # Pick random unsatisfied clause
                unsat_indices = np.where(clause_unsat)[0]
                if len(unsat_indices) == 0:
                    return assignment
                ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
                clause = reduced_clauses[ci]
                
                # Choose variable to flip: greedy with noise
                # Score = (# unsat clauses that would be satisfied) - (# sat clauses that would break)
                # Add noise for exploration
                best_var = -1
                best_score = -float('inf')
                
                for var, sign in clause:
                    # Compute delta if we flip var
                    delta = 0
                    for other_ci, other_sign in reduced_var_clauses[var]:
                        if sat_count[other_ci] > 0:
                            # Currently satisfied
                            if other_sign == sign:
                                # Flipping var would make this literal false
                                # Check if it's the only satisfying literal
                                if sat_count[other_ci] == 1:
                                    delta += 1  # Would become unsat
                        else:
                            # Currently unsat
                            if other_sign != sign:
                                # Flipping var would make this literal true
                                delta -= 1  # Would become sat
                    # Noise: more noise early, less later
                    noise_scale = 1.0 * (1.0 - step / steps_per_restart) + 0.1
                    score = delta + rng.gauss(0, noise_scale)
                    
                    if score > best_score:
                        best_score = score
                        best_var = var
                
                if best_var == -1:
                    break
                
                # Flip the variable
                assignment[best_var] = not assignment[best_var]
                
                # Update sat_count for all clauses containing best_var
                for other_ci, other_sign in reduced_var_clauses[best_var]:
                    # The literal (best_var, other_sign) changes value
                    if assignment[best_var] == (other_sign > 0):
                        sat_count[other_ci] += 1
                    else:
                        sat_count[other_ci] -= 1
                    
                    was_unsat = (sat_count[other_ci] == 0)
                    if was_unsat:
                        clause_unsat[other_ci] = True
                        unsat_count += 1
                    else:
                        clause_unsat[other_ci] = False
                        unsat_count -= 1
                
                if unsat_count == 0:
                    return assignment
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
        
        return best_assignment if best_assignment is not None else [False] * n_r
    
    def _simulated_annealing_reduced(rng, max_iters=50000):
        """Fast simulated annealing with sat_count tracking."""
        assignment = [rng.random() < 0.5 for _ in range(n_r)]
        
        sat_count = np.zeros(m_r, dtype=np.int32)
        for ci, clause in enumerate(reduced_clauses):
            cnt = 0
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    cnt += 1
            sat_count[ci] = cnt
        
        unsat_count = int(np.sum(sat_count == 0))
        if unsat_count == 0:
            return assignment
        
        best_assignment = assignment[:]
        best_unsat = unsat_count
        
        T0 = 2.0
        T_end = 0.01
        alpha = (T_end / T0) ** (1.0 / max_iters)
        T = T0
        
        for step in range(max_iters):
            if time.time() > deadline:
                return best_assignment
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            var = rng.randrange(n_r)
            old_val = assignment[var]
            assignment[var] = not old_val
            
            delta = 0
            for other_ci, other_sign in reduced_var_clauses[var]:
                old_sat = sat_count[other_ci]
                if assignment[var] == (other_sign > 0):
                    sat_count[other_ci] = old_sat + 1
                else:
                    sat_count[other_ci] = old_sat - 1
                
                new_sat = sat_count[other_ci]
                if old_sat == 0 and new_sat > 0:
                    delta -= 1
                elif old_sat > 0 and new_sat == 0:
                    delta += 1
            
            new_unsat = unsat_count + delta
            
            if new_unsat <= unsat_count:
                unsat_count = new_unsat
            elif delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                unsat_count = new_unsat
            else:
                # Revert
                assignment[var] = old_val
                for other_ci, other_sign in reduced_var_clauses[var]:
                    if old_val == (other_sign > 0):
                        sat_count[other_ci] -= 1
                    else:
                        sat_count[other_ci] += 1
            
            T *= alpha
        
        return best_assignment
    
    # Run solvers on reduced instance with multiple seeds for diversity
    best_full_assignment = None
    best_unsat_global = m + 1
    
    def _map_and_verify(reduced_result):
        """Map reduced result back to full assignment and verify."""
        full_assign = [False] * n
        for var in range(n):
            if var in fixed:
                full_assign[var] = fixed[var]
            elif var in var_to_reduced_idx:
                full_assign[var] = reduced_result[var_to_reduced_idx[var]]
            else:
                full_assign[var] = False
        
        # Count unsatisfied clauses
        unsat = 0
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if full_assign[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return full_assign, unsat
    
    # Try WalkSAT with multiple seeds
    seeds = [42, 123, 456, 789, 1024]
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(seed)
        reduced_result = _walksat_reduced(rng, max_restarts=30, steps_per_restart=10000)
        
        full_assign, unsat = _map_and_verify(reduced_result)
        
        if unsat == 0:
            return full_assign
        
        if unsat < best_unsat_global:
            best_unsat_global = unsat
            best_full_assignment = full_assign
    
    # Try simulated annealing if time permits
    if time.time() < deadline - 10:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 5:
                break
            
            reduced_result2 = _simulated_annealing_reduced(random.Random(seed), max_iters=40000)
            
            full_assign2, unsat2 = _map_and_verify(reduced_result2)
            
            if unsat2 == 0:
                return full_assign2
            
            if unsat2 < best_unsat_global:
                best_unsat_global = unsat2
                best_full_assignment = full_assign2
    
    # Return best result found
    if best_full_assignment is not None:
        return best_full_assignment
    
    # Fallback: return all False
    return [False] * n
# EVOLVE-BLOCK-END
