import time
import random
import sys
import math
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin

    if m == 0:
        return [True] * n

    # Preprocessing: Convert to 0-based, build structures
    # clause_data: list of lists of (var, sign)
    # var_occ: list of lists of (clause_idx, sign)
    clause_data = []
    var_occ = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cd = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cd.append((v, sign))
            var_occ[v].append((i, sign))
        clause_data.append(cd)

    # Unit Propagation and Pure Literal Elimination
    assignment = [None] * n
    clause_active = [True] * m
    sat_counts = [0] * m
    
    def propagate():
        changed = True
        while changed:
            changed = False
            # Pure Literal Elimination
            for v in range(n):
                if assignment[v] is not None:
                    continue
                has_pos = False
                has_neg = False
                for (ci, sign) in var_occ[v]:
                    if not clause_active[ci]:
                        continue
                    if sign == 1:
                        has_pos = True
                    else:
                        has_neg = True
                if has_pos and not has_neg:
                    assignment[v] = True
                    changed = True
                elif has_neg and not has_pos:
                    assignment[v] = False
                    changed = True
            
            # Unit Propagation
            for ci in range(m):
                if not clause_active[ci]:
                    continue
                unassigned = []
                true_count = 0
                for (v, sign) in clause_data[ci]:
                    if assignment[v] is None:
                        unassigned.append((v, sign))
                    else:
                        if sign == 1:
                            if assignment[v]:
                                true_count += 1
                        else:
                            if not assignment[v]:
                                true_count += 1
                if true_count > 0:
                    clause_active[ci] = False
                    sat_counts[ci] = true_count
                    changed = True
                elif len(unassigned) == 1:
                    v, sign = unassigned[0]
                    assignment[v] = (sign == 1)
                    changed = True
        
        # Update sat_counts for active clauses
        for ci in range(m):
            if clause_active[ci]:
                sc = 0
                for (v, sign) in clause_data[ci]:
                    if assignment[v] is None:
                        continue
                    if sign == 1:
                        if assignment[v]:
                            sc += 1
                    else:
                        if not assignment[v]:
                            sc += 1
                sat_counts[ci] = sc

    propagate()
    
    # Check if solved
    if all(not clause_active[ci] or sat_counts[ci] > 0 for ci in range(m)):
        return [a if a is not None else True for a in assignment]

    # Initialize unassigned variables
    rng = random.Random(42)
    unassigned_vars = [v for v in range(n) if assignment[v] is None]
    for v in unassigned_vars:
        assignment[v] = rng.random() < 0.5

    # Recompute state
    for ci in range(m):
        sc = 0
        for (v, sign) in clause_data[ci]:
            if sign == 1:
                if assignment[v]:
                    sc += 1
            else:
                if not assignment[v]:
                    sc += 1
        sat_counts[ci] = sc
        clause_active[ci] = (sc == 0)

    unsat_clauses = [ci for ci in range(m) if clause_active[ci]]
    clause_pos = [-1] * m
    for i, ci in enumerate(unsat_clauses):
        clause_pos[ci] = i
    
    if not unsat_clauses:
        return assignment[:]

    best_unsat = len(unsat_clauses)
    best_assignment = assignment[:]

    # Efficient Local Search (WalkSAT/SA Hybrid)
    # Data structures for O(1) or amortized O(1) updates
    # var_unsat_count[v]: number of unsat clauses containing v
    var_unsat_count = [0] * n
    for ci in unsat_clauses:
        for (v, sign) in clause_data[ci]:
            var_unsat_count[v] += 1

    max_restarts = 200
    max_steps_per_restart = 20000  # Can be higher if fast
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize or Perturb
        if restart == 0:
            current_assignment = best_assignment[:]
        else:
            # Perturb the best assignment found so far
            current_assignment = best_assignment[:]
            # Flip a random subset of variables
            num_flips = max(1, n // 10)
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                current_assignment[v] = not current_assignment[v]
        
        # Recompute state for current_assignment
        for ci in range(m):
            sc = 0
            for (v, sign) in clause_data[ci]:
                if sign == 1:
                    if current_assignment[v]:
                        sc += 1
                else:
                    if not current_assignment[v]:
                        sc += 1
            sat_counts[ci] = sc
            clause_active[ci] = (sc == 0)
        
        unsat_clauses = [ci for ci in range(m) if clause_active[ci]]
        for i, ci in enumerate(unsat_clauses):
            clause_pos[ci] = i
        for ci in range(m):
            if not clause_active[ci]:
                clause_pos[ci] = -1
                
        current_cost = len(unsat_clauses)
        
        if current_cost == 0:
            return current_assignment
        
        if current_cost < best_unsat:
            best_unsat = current_cost
            best_assignment = current_assignment[:]

        # Rebuild var_unsat_count
        var_unsat_count = [0] * n
        for ci in unsat_clauses:
            for (v, sign) in clause_data[ci]:
                var_unsat_count[v] += 1

        # SA Parameters
        T_start = 1.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max_steps_per_restart)
        T = T_start
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return current_assignment
            
            # Update best
            if current_cost < best_unsat:
                best_unsat = current_cost
                best_assignment = current_assignment[:]
            
            # Select a variable to flip
            # 1. Pick a random unsat clause
            if len(unsat_clauses) == 0:
                return current_assignment
            
            idx = rng.randint(0, len(unsat_clauses) - 1)
            ci = unsat_clauses[idx]
            
            # Variable selection heuristic:
            # Pick the variable in this clause with the highest var_unsat_count.
            cd_c = clause_data[ci]
            best_v = cd_c[0][0]
            best_pressure = var_unsat_count[best_v]
            
            for (v, sign) in cd_c[1:]:
                p = var_unsat_count[v]
                if p > best_pressure:
                    best_pressure = p
                    best_v = v
            
            v_to_flip = best_v
            
            # Calculate delta cost
            # Flip v_to_flip
            old_val = current_assignment[v_to_flip]
            new_val = not old_val
            
            delta = 0
            for (ci_v, sign) in var_occ[v_to_flip]:
                if sign == 1:
                    # Literal is v
                    was_true = old_val
                    is_true = new_val
                else:
                    # Literal is -v
                    was_true = not old_val
                    is_true = not new_val
                
                if was_true and not is_true:
                    # Clause becomes unsatisfied
                    if sat_counts[ci_v] == 1:
                        delta += 1
                elif not was_true and is_true:
                    # Clause becomes satisfied
                    if sat_counts[ci_v] == 0:
                        delta -= 1
            
            # Acceptance Probability
            if delta <= 0:
                accept = True
            else:
                # Metropolis Criterion
                if T > 0:
                    p = math.exp(-delta / T)
                    accept = (rng.random() < p)
                else:
                    accept = False
            
            if accept:
                # Apply flip
                current_assignment[v_to_flip] = new_val
                current_cost += delta
                
                # Update sat_counts, clause_active, unsat_clauses, var_unsat_count
                for (ci_v, sign) in var_occ[v_to_flip]:
                    if sign == 1:
                        old_lit_true = old_val
                        new_lit_true = new_val
                    else:
                        old_lit_true = not old_val
                        new_lit_true = not new_val
                    
                    if old_lit_true and not new_lit_true:
                        sat_counts[ci_v] -= 1
                        if sat_counts[ci_v] == 0:
                            # Clause became unsatisfied
                            if not clause_active[ci_v]:
                                clause_active[ci_v] = True
                                # Add to unsat_clauses
                                unsat_clauses.append(ci_v)
                                clause_pos[ci_v] = len(unsat_clauses) - 1
                                # Update var_unsat_count for this clause's vars
                                for (v_u, s_u) in clause_data[ci_v]:
                                    var_unsat_count[v_u] += 1
                    elif not old_lit_true and new_lit_true:
                        sat_counts[ci_v] += 1
                        if sat_counts[ci_v] == 1:
                            # Clause became satisfied
                            if clause_active[ci_v]:
                                clause_active[ci_v] = False
                                # Remove from unsat_clauses
                                pos = clause_pos[ci_v]
                                last = unsat_clauses[-1]
                                unsat_clauses[pos] = last
                                clause_pos[last] = pos
                                unsat_clauses.pop()
                                clause_pos[ci_v] = -1
                                # Update var_unsat_count for this clause's vars
                                for (v_u, s_u) in clause_data[ci_v]:
                                    var_unsat_count[v_u] -= 1
                
            # Cool down
            T *= cooling_rate

    if best_assignment is not None:
        return best_assignment
    
    return [True] * n
