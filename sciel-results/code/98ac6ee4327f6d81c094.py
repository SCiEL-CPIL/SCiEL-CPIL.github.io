# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # 1. Pre-process
    # Clause data: list of lists of ints (signed literals)
    # We need:
    # - var_to_clauses: dict var_idx -> list of clause_indices
    # - clause_vars: list of lists of var_idx
    # - clause_signs: list of lists of bool (True for positive)
    
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
    # Convert to bool list for easy manipulation
    phase_list = phase.tolist()
    
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_assignment = phase_list.copy()
    best_unsat_count = float('inf')
    
    # Main Search Loop
    # We will use a restart strategy.
    # Each restart:
    #   - Reset assignment to phase (or random?)
    #   - Run local search for X steps or until solved
    #   - If solved, return
    #   - If time runs out, break
    
    # Adaptive parameters
    current_steps = 100
    max_steps = 100000
    
    while time.time() < deadline:
        # Initialize assignment
        assignment = phase_list.copy()
        
        # Initialize clause_sat_count: number of literals in clause i that are satisfied
        clause_sat_count = np.zeros(num_clauses, dtype=np.int8)
        
        # Initialize clause_weight: for weighted selection of unsat clauses
        clause_weight = np.zeros(num_clauses, dtype=np.float64)
        
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
            
        # Local Search Loop
        steps_in_restart = 0
        
        while steps_in_restart < current_steps:
            if time.time() > deadline:
                # Safety break
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment.copy()
                return best_assignment
                
            if current_unsat == 0:
                return assignment
                
            # Select a clause to repair
            # Weighted random choice among unsat clauses
            # Weights are incremented every time a clause is picked and fails? 
            # Or simply use the count of times it has been unsat?
            # Standard approach: increment weight of the selected clause.
            
            # To make it faster, we can just pick a random unsat clause from the list.
            # But maintaining a list of unsat clauses dynamically is tricky with incremental updates.
            # Alternative: Pick a random clause index. If it's sat, retry? No, that's inefficient if many are sat.
            # Better: Maintain a list of unsat clause indices.
            # Updating the list on every flip is O(degree).
            
            # Let's maintain `unsat_list` as a Python list of indices.
            # Initialization:
            if not hasattr(solve, '_unsat_list'): # This is bad practice for state, let's do it locally
                 pass
            
            # Re-initialize unsat_list at start of restart
            unsat_list = [i for i in range(num_clauses) if clause_sat_count[i] == 0]
            
            if not unsat_list:
                return assignment
                
            # Weighted selection:
            # We can use the clause_weight array.
            # For simplicity and speed, let's just pick randomly from unsat_list first.
            # If we want weighting, we need to sum weights of unsat clauses.
            # Let's try a simple "walk" probability.
            # Standard WalkSAT: 
            # 1. Pick random unsat clause.
            # 2. Pick variable to flip.
            #    - With probability p (e.g. 0.5), pick a variable that satisfies the clause (if any).
            #    - With probability 1-p, pick a random variable from the clause.
            
            clause_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            
            # Update weight of this clause
            clause_weight[clause_idx] += 1.0
            
            cv = clause_vars[clause_idx]
            cs = clause_signs[clause_idx]
            
            # Determine flip variable
            # Calculate delta for each variable in the clause
            # Delta = change in unsat count
            # If we flip v, for each clause c containing v:
            #   - If c is currently sat (count > 0):
            #       - If v is satisfying c (assignment[v]==sign), and count==1, then c becomes unsat (delta +1)
            #       - If v is NOT satisfying c, or count>1, c remains sat (delta 0)
            #   - If c is currently unsat (count == 0):
            #       - If v is NOT satisfying c (assignment[v]!=sign), then c becomes sat (delta -1)
            #       - If v IS satisfying c (impossible if count==0), c remains unsat
            
            best_var = -1
            best_delta = float('inf')
            
            # Optimization: Pre-compute which variables in cv are currently satisfying the clause?
            # Since clause is unsat, NO variable in cv is satisfying it.
            # So for the selected clause, flipping ANY variable will satisfy it (delta -1 for this clause).
            
            for k in range(3):
                v = cv[k]
                delta = -1 # This clause will become satisfied
                
                # Check other clauses containing v
                for c_idx in var_to_clauses[v]:
                    if c_idx == clause_idx:
                        continue
                    
                    # Check if v is currently satisfying c_idx
                    # We need the sign of v in c_idx
                    # We have clause_signs[c_idx] and clause_vars[c_idx]
                    # Find index of v in c_idx? That's O(3).
                    # To avoid this, we can store a map (clause_idx, var) -> sign? 
                    # Or just loop through the 3 literals.
                    
                    c_cv = clause_vars[c_idx]
                    c_cs = clause_signs[c_idx]
                    
                    # Find v's sign in this clause
                    v_sign = None
                    for j in range(3):
                        if c_cv[j] == v:
                            v_sign = c_cs[j]
                            break
                    
                    if v_sign is None:
                        continue # Should not happen
                    
                    # Is v currently satisfying c_idx?
                    v_satisfies_now = (assignment[v] == v_sign)
                    
                    # Current sat count of c_idx
                    c_count = clause_sat_count[c_idx]
                    
                    if v_satisfies_now:
                        # v is helping this clause
                        if c_count == 1:
                            # If we flip v, this clause becomes unsat
                            delta += 1
                        # else: c_count > 1, flipping v leaves it sat (delta 0)
                    else:
                        # v is NOT helping this clause
                        # If c_count == 0 (unsat), flipping v makes it sat (delta -1)
                        # If c_count > 0 (sat), flipping v leaves it sat (delta 0)
                        if c_count == 0:
                            delta -= 1
                        # else: delta 0
                    
                # Tie-breaking: random
                if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                    best_delta = delta
                    best_var = v
                    
            # WalkSAT heuristic:
            # With probability p (e.g. 0.3-0.5), pick a random variable from the clause instead of the best one.
            # This helps escape local minima.
            p_walk = 0.3
            if rng.random() < p_walk:
                best_var = cv[rng.randint(0, 2)]
                
            # Flip best_var
            assignment[best_var] = not assignment[best_var]
            
            # Update clause_sat_count for all clauses containing best_var
            for c_idx in var_to_clauses[best_var]:
                c_cv = clause_vars[c_idx]
                c_cs = clause_signs[c_idx]
                
                # Find sign of best_var in this clause
                v_sign = None
                for j in range(3):
                    if c_cv[j] == best_var:
                        v_sign = c_cs[j]
                        break
                        
                if v_sign is None:
                    continue
                
                # Did best_var satisfy the clause before flip?
                # assignment[best_var] is now the NEW value.
                # So "before" value was not assignment[best_var].
                # If (not assignment[best_var]) == v_sign, then it satisfied before.
                
                satisfied_before = (not assignment[best_var]) == v_sign
                satisfied_after = assignment[best_var] == v_sign
                
                if satisfied_before and not satisfied_after:
                    clause_sat_count[c_idx] -= 1
                    if clause_sat_count[c_idx] == 0:
                        # Became unsat, add to unsat_list
                        unsat_list.append(c_idx)
                elif not satisfied_before and satisfied_after:
                    clause_sat_count[c_idx] += 1
                    if clause_sat_count[c_idx] == 1:
                        # Became sat, remove from unsat_list
                        # Removing from list is O(N). 
                        # For small N, we can just filter? Or mark as inactive?
                        # Filtering is expensive if done every step.
                        # Let's use a "valid" flag or just clean up periodically?
                        # Or use a set? Set removal is O(1).
                        # But we need weighted random selection from set?
                        # Let's stick to list and accept O(N) removal cost, or just rebuild unsat_list occasionally?
                        # Given N=50 vars, C~250-500, removal is fast.
                        try:
                            unsat_list.remove(c_idx)
                        except ValueError:
                            pass # Already removed?
            
            # Update current_unsat count
            current_unsat = len(unsat_list)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
                
            steps_in_restart += 1
            
            # Periodically clean up unsat_list if it gets too large with duplicates?
            # The logic above adds on 0->-1 transition and removes on -1->0 transition.
            # So it should be exact.
            
        # End of restart
        # Increase steps for next restart
        current_steps = int(current_steps * 1.5)
        if current_steps > max_steps:
            current_steps = max_steps

    return best_assignment
# EVOLVE-BLOCK-END
