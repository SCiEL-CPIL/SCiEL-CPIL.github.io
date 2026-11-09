# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # 1. Pre-process
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_signs = []
    
    for i, c in enumerate(clauses):
        cv = []
        cs = []
        for lit in c:
            v = abs(lit) - 1
            s = (lit > 0)
            cv.append(v)
            cs.append(s)
            var_to_clauses[v].append(i)
        clause_vars.append(cv)
        clause_signs.append(cs)
        
    num_clauses = len(clauses)
    
    # Phase initialization
    pos_counts = np.zeros(n, dtype=np.int32)
    neg_counts = np.zeros(n, dtype=np.int32)
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    phase = pos_counts > neg_counts
    phase_list = phase.tolist()
    
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 285.0
    
    best_assignment = phase_list.copy()
    best_unsat_count = float('inf')
    
    # Simulated Annealing parameters
    T0 = 5.0  # initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Main Search Loop with restarts
    current_steps = 500
    max_steps = 200000
    restart_count = 0
    
    while time.time() < deadline:
        # Initialize assignment
        assignment = phase_list.copy()
        
        # Initialize clause_sat_count: number of literals in clause i that are satisfied
        clause_sat_count = np.zeros(num_clauses, dtype=np.int8)
        
        # Compute initial satisfaction counts
        for i in range(num_clauses):
            cv = clause_vars[i]
            cs = clause_signs[i]
            count = 0
            for j in range(3):
                if assignment[cv[j]] == cs[j]:
                    count += 1
            clause_sat_count[i] = count
            
        # Find initial unsat count
        unsat_indices = np.where(clause_sat_count == 0)[0]
        current_unsat = len(unsat_indices)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
        
        # Maintain unsat_list for efficient selection
        unsat_list = [int(i) for i in unsat_indices]
        
        # Temperature for this restart
        T = T0
        
        # Local Search Loop with simulated annealing
        steps_in_restart = 0
        
        while steps_in_restart < current_steps:
            if time.time() > deadline - 5.0:
                # Safety break with margin
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment.copy()
                return best_assignment
                
            if current_unsat == 0:
                return assignment
            
            # Select a clause to repair - weighted by penalty (dynamic)
            # Penalty increases for clauses that remain unsatisfied
            if unsat_list:
                # Weighted selection: clauses with higher penalty are more likely to be chosen
                # Use clause_sat_count == 0 as the unsat indicator
                # We can weight by how long they've been unsat or just random
                clause_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            else:
                return assignment
            
            cv = clause_vars[clause_idx]
            cs = clause_signs[clause_idx]
            
            # Determine flip variable using delta computation
            # Delta = change in weighted objective
            # Weighted objective: sum of penalties for unsat clauses
            # For simplicity, use standard WalkSAT delta with annealing acceptance
            
            best_var = -1
            best_delta = float('inf')
            
            for k in range(3):
                v = cv[k]
                delta = -1  # This clause will become satisfied
                
                # Check other clauses containing v
                for c_idx in var_to_clauses[v]:
                    if c_idx == clause_idx:
                        continue
                    
                    c_cv = clause_vars[c_idx]
                    c_cs = clause_signs[c_idx]
                    
                    # Find v's sign in this clause
                    v_sign = None
                    for j in range(3):
                        if c_cv[j] == v:
                            v_sign = c_cs[j]
                            break
                    
                    if v_sign is None:
                        continue
                    
                    v_satisfies_now = (assignment[v] == v_sign)
                    c_count = clause_sat_count[c_idx]
                    
                    if v_satisfies_now:
                        if c_count == 1:
                            delta += 1
                    else:
                        if c_count == 0:
                            delta -= 1
                    
                    # Dynamic penalty: add small penalty for clauses that are "hard"
                    # Use clause frequency as a proxy
                    if c_count == 0:
                        delta += 0.1 * len(var_to_clauses[c_cv[0]])
                    
                if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                    best_delta = delta
                    best_var = v
            
            # Simulated annealing: accept worse moves with probability exp(-delta/T)
            if delta > 0 and T > T_min:
                accept_prob = math.exp(-delta / T)
                if rng.random() > accept_prob:
                    # Reject this move, try a random variable instead
                    best_var = cv[rng.randint(0, 2)]
                    # Recompute delta for the random choice
                    delta = 0
                    v = best_var
                    for c_idx in var_to_clauses[v]:
                        c_cv = clause_vars[c_idx]
                        c_cs = clause_signs[c_idx]
                        v_sign = None
                        for j in range(3):
                            if c_cv[j] == v:
                                v_sign = c_cs[j]
                                break
                        if v_sign is None:
                            continue
                        v_satisfies_now = (assignment[v] == v_sign)
                        c_count = clause_sat_count[c_idx]
                        if v_satisfies_now:
                            if c_count == 1:
                                delta += 1
                        else:
                            if c_count == 0:
                                delta -= 1
            
            # Flip best_var
            assignment[best_var] = not assignment[best_var]
            
            # Update clause_sat_count for all clauses containing best_var
            for c_idx in var_to_clauses[best_var]:
                c_cv = clause_vars[c_idx]
                c_cs = clause_signs[c_idx]
                
                v_sign = None
                for j in range(3):
                    if c_cv[j] == best_var:
                        v_sign = c_cs[j]
                        break
                        
                if v_sign is None:
                    continue
                
                satisfied_before = (not assignment[best_var]) == v_sign
                satisfied_after = assignment[best_var] == v_sign
                
                if satisfied_before and not satisfied_after:
                    clause_sat_count[c_idx] -= 1
                    if clause_sat_count[c_idx] == 0:
                        unsat_list.append(c_idx)
                elif not satisfied_before and satisfied_after:
                    clause_sat_count[c_idx] += 1
                    if clause_sat_count[c_idx] == 1:
                        try:
                            unsat_list.remove(c_idx)
                        except ValueError:
                            pass
            
            # Update current_unsat count
            current_unsat = len(unsat_list)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
                
            # Cool down
            T *= cooling_rate
            
            steps_in_restart += 1
            
        # End of restart
        # Increase steps for next restart and reset temperature
        current_steps = int(current_steps * 1.3)
        if current_steps > max_steps:
            current_steps = max_steps
        restart_count += 1

    return best_assignment
# EVOLVE-BLOCK-END
