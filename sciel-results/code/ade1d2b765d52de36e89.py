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

    num_clauses = len(clauses)
    
    # 1. Pre-process: Build efficient data structures
    # var_to_clauses[v] = list of clause indices containing v
    var_to_clauses = [[] for _ in range(n)]
    # clause_vars[i] = [v0, v1, v2] (0-indexed)
    # clause_signs[i] = [s0, s1, s2] (True if positive, False if negative)
    # clause_var_signs[i][j] = sign of var j in clause i? No, let's use a dict or parallel arrays.
    # For speed, let's store for each clause, the variables and their signs.
    clause_vars = []
    clause_signs = []
    
    # Precompute: for each clause i and position j, we have var and sign.
    # To speed up delta calculation, we need to know: for a variable v, what is its sign in clause c?
    # We can store a dictionary: clause_var_sign[c][v] = sign (1 or -1)
    # But n=50, num_clauses can be large. A list of dicts or a 2D array (sparse) is good.
    # Since max vars per clause is 3, we can just store the 3 pairs.
    
    # Let's use a structure: 
    # clause_info[i] = [(v0, s0), (v1, s1), (v2, s2)]
    clause_info = []
    
    for i, c in enumerate(clauses):
        cv = []
        cs = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append(v)
            cs.append(s)
            var_to_clauses[v].append(i)
        clause_info.append(list(zip(cv, cs)))
        
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
    deadline = start_time + 290.0 # 10s margin
    
    best_assignment = phase_list.copy()
    best_unsat_count = float('inf')
    
    # Simulated Annealing parameters
    T0 = 10.0
    T_min = 0.001
    cooling_rate = 0.999
    
    # Weights for clauses (for conflict-driven learning)
    # Initial weight 1.0
    clause_weights = np.ones(num_clauses, dtype=np.float32)
    
    current_steps = 1000
    max_steps = 500000
    restart_count = 0
    
    while time.time() < deadline:
        # Initialize assignment
        assignment = phase_list.copy()
        
        # Initialize clause_sat_count: number of literals in clause i that are satisfied
        clause_sat_count = np.zeros(num_clauses, dtype=np.int8)
        
        # Compute initial satisfaction counts
        for i in range(num_clauses):
            count = 0
            for v, s in clause_info[i]:
                if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                    count += 1
            clause_sat_count[i] = count
            
        unsat_indices = np.where(clause_sat_count == 0)[0]
        current_unsat = len(unsat_indices)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
            
        # Maintain unsat_list
        unsat_list = [int(i) for i in unsat_indices]
        
        # Reset clause weights? Or keep them? 
        # Keeping them across restarts might be bad if the restart is far away.
        # Let's reset weights on restart to avoid bias from previous failed attempts.
        clause_weights[:] = 1.0
        
        T = T0
        steps_in_restart = 0
        
        while steps_in_restart < current_steps:
            if time.time() > deadline - 5.0:
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment.copy()
                return best_assignment
                
            if current_unsat == 0:
                return assignment
            
            # Select a clause to repair
            # Weighted selection: clauses with higher weights are more likely to be chosen
            # This focuses on "hard" clauses that have resisted satisfaction
            if len(unsat_list) > 1:
                # Sample using weights
                weights = clause_weights[unsat_list]
                # Normalize not strictly necessary for numpy choice if we use cumulative sum, 
                # but let's just pick random for speed if weights are similar, 
                # or use weighted choice.
                # For speed, let's just pick random from unsat_list initially, 
                # and rely on delta to guide. 
                # Actually, weighted selection is key for CDCL-like behavior.
                clause_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            else:
                clause_idx = unsat_list[0]
                
            # Get clause details
            c_info = clause_info[clause_idx]
            
            # Determine best variable to flip
            # Delta = change in total weighted unsat count
            # Objective: Minimize sum( w_c * I(unsat_c) )
            # Flipping v changes the state of all clauses containing v.
            
            best_var = -1
            best_delta = float('inf')
            
            # We evaluate flipping each of the 3 variables in the chosen clause
            for v, s in c_info:
                delta = 0.0
                
                # For each clause containing v
                for c_idx in var_to_clauses[v]:
                    if c_idx == clause_idx:
                        # This clause is currently unsat (since we picked it from unsat_list)
                        # Flipping v (which is in this clause) will make it satisfied?
                        # v's sign is s. 
                        # If we flip v, does it satisfy the clause?
                        # Currently clause_idx is unsat, so all literals are false.
                        # Literal (v, s) is false. 
                        # If we flip v, literal becomes true. So clause becomes satisfied.
                        # Change in cost: -w[clause_idx]
                        delta -= clause_weights[c_idx]
                        continue
                    
                    # For other clauses, we need to check if flipping v changes their sat status
                    # Find v's sign in c_idx
                    # We stored clause_info[c_idx] as list of (v, s)
                    # We need to find the entry for v.
                    # Since max 3 vars, linear search is fast.
                    v_sign_in_c = None
                    for vv, ss in clause_info[c_idx]:
                        if vv == v:
                            v_sign_in_c = ss
                            break
                            
                    if v_sign_in_c is None:
                        continue # Should not happen if data structures are correct
                    
                    # Current state of literal (v, v_sign_in_c) in clause c_idx
                    # satisfied = (assignment[v] and v_sign_in_c == 1) or (not assignment[v] and v_sign_in_c == -1)
                    # New state after flip:
                    # new_assignment[v] = not assignment[v]
                    
                    current_sat = (assignment[v] and v_sign_in_c == 1) or (not assignment[v] and v_sign_in_c == -1)
                    new_sat = (not assignment[v] and v_sign_in_c == 1) or (assignment[v] and v_sign_in_c == -1)
                    
                    c_count = clause_sat_count[c_idx]
                    
                    if current_sat and not new_sat:
                        # Literal was satisfied, now unsatisfied
                        # Clause sat count decreases by 1
                        # If it was 1, it becomes 0 (unsat) -> Cost increases by w
                        # If it was >1, it stays sat -> Cost unchanged
                        if c_count == 1:
                            delta += clause_weights[c_idx]
                    elif not current_sat and new_sat:
                        # Literal was unsatisfied, now satisfied
                        # Clause sat count increases by 1
                        # If it was 0, it becomes 1 (sat) -> Cost decreases by w
                        # If it was >0, it stays sat -> Cost unchanged
                        if c_count == 0:
                            delta -= clause_weights[c_idx]
                    
                # Add a small random noise to break ties and encourage exploration
                # delta += rng.gauss(0, 0.1)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
            # Simulated Annealing: Accept worse moves
            if best_delta > 0 and T > T_min:
                accept_prob = math.exp(-best_delta / T)
                if rng.random() > accept_prob:
                    # Reject best move, pick a random variable from the clause
                    best_var = c_info[rng.randint(0, 2)][0]
                    # Recompute delta for this random choice? 
                    # For speed, we can just accept the risk or recompute. 
                    # Recomputing is safer for state consistency.
                    delta = 0.0
                    v = best_var
                    for c_idx in var_to_clauses[v]:
                        if c_idx == clause_idx:
                            delta -= clause_weights[c_idx]
                            continue
                        v_sign_in_c = None
                        for vv, ss in clause_info[c_idx]:
                            if vv == v:
                                v_sign_in_c = ss
                                break
                        if v_sign_in_c is None: continue
                        current_sat = (assignment[v] and v_sign_in_c == 1) or (not assignment[v] and v_sign_in_c == -1)
                        new_sat = (not assignment[v] and v_sign_in_c == 1) or (assignment[v] and v_sign_in_c == -1)
                        c_count = clause_sat_count[c_idx]
                        if current_sat and not new_sat:
                            if c_count == 1: delta += clause_weights[c_idx]
                        elif not current_sat and new_sat:
                            if c_count == 0: delta -= clause_weights[c_idx]
                    best_delta = delta

            # Flip best_var
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update clause_sat_count and unsat_list
            for c_idx in var_to_clauses[best_var]:
                # Find sign of best_var in c_idx
                v_sign_in_c = None
                for vv, ss in clause_info[c_idx]:
                    if vv == best_var:
                        v_sign_in_c = ss
                        break
                if v_sign_in_c is None: continue
                
                # Determine if literal was satisfied before and after
                # Before: assignment was old_val
                sat_before = (old_val and v_sign_in_c == 1) or (not old_val and v_sign_in_c == -1)
                # After: assignment is not old_val
                sat_after = (not old_val and v_sign_in_c == 1) or (old_val and v_sign_in_c == -1)
                
                if sat_before and not sat_after:
                    clause_sat_count[c_idx] -= 1
                    if clause_sat_count[c_idx] == 0:
                        unsat_list.append(c_idx)
                        # Increase weight of this newly unsat clause?
                        # Or increase weight of clauses that stay unsat?
                        # Let's increase weight of clauses that are unsat at the end of step?
                        # Simpler: Increase weight of clause if it becomes unsat.
                        clause_weights[c_idx] *= 1.1
                elif not sat_before and sat_after:
                    clause_sat_count[c_idx] += 1
                    if clause_sat_count[c_idx] == 1:
                        try:
                            unsat_list.remove(c_idx)
                        except ValueError:
                            pass
                        # Decrease weight if it becomes sat?
                        clause_weights[c_idx] = max(1.0, clause_weights[c_idx] * 0.9)
                        
            current_unsat = len(unsat_list)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
                
            T *= cooling_rate
            steps_in_restart += 1

        # End of restart
        current_steps = int(current_steps * 1.2)
        if current_steps > max_steps:
            current_steps = max_steps
        restart_count += 1

    return best_assignment
# EVOLVE-BLOCK-END
