"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Preprocess clauses into numpy arrays for vectorization
    # Convert 1-based signed literals to 0-based indices and signs
    # indices: shape (m, 3), values in [0, n-1]
    # signs: shape (m, 3), values in {+1, -1}
    
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
            
    # Variable occurrence lists for fast gain calculation
    # var_to_clauses[var] = list of clause indices containing var
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_to_clauses[v].append(i)
    
    # Convert to numpy arrays of lists for easier handling? 
    # Actually, lists of lists are fine for the inner loop if we optimize the gain calc.
    # But we can precompute the "other literals" for each variable occurrence to speed up gain.
    
    # state: boolean array of assignments
    # sat_counts: integer array of number of satisfied literals per clause (0, 1, 2, 3)
    
    def initialize_assignment():
        return np.random.rand(n) < 0.5

    def calculate_sat_counts(assignment):
        # assignment: (n,) bool
        # indices: (m, 3) int
        # signs: (m, 3) int
        # lit_val = assignment[indices] * signs
        # satisfied if lit_val == 1
        # assignment[indices] is (m, 3) bool
        # signs is (m, 3) int
        # We need: assignment[indices] == (signs > 0)
        # Let's just compute:
        # val = np.take(assignment, indices) # (m, 3) bool
        # is_sat = (val == (signs > 0)) # (m, 3) bool
        # return np.sum(is_sat, axis=1)
        
        val = np.take(assignment, indices)
        # signs > 0 is a bool array where positive literals are True
        # if sign is +1, we want assignment to be True.
        # if sign is -1, we want assignment to be False.
        # So satisfied if (sign==1 and val==True) or (sign==-1 and val==False)
        # This is equivalent to: (val == (signs > 0))
        is_sat = (val == (signs > 0))
        return np.sum(is_sat, axis=1)

    def flip_variable(var, assignment, sat_counts):
        """
        Flip variable var, update assignment and sat_counts.
        Returns new assignment and sat_counts.
        """
        # Find all clauses containing var
        cl_idx = var_to_clauses[var]
        if not cl_idx:
            return assignment, sat_counts
            
        cl_idx_arr = np.array(cl_idx, dtype=np.int32)
        
        # Get current values of the other 2 literals in these clauses
        # We need to know if the clause was satisfied by var, and if it will be satisfied by others.
        
        # Current state of var in these clauses:
        # var_signs = signs[cl_idx_arr, :]
        # var_positions = np.where(var_signs == indices[cl_idx_arr, :]... no, we need positions
        
        # Efficient way:
        # For each clause in cl_idx_arr, find the position j where indices[i, j] == var
        # pos_mask = (indices[cl_idx_arr] == var) # (len(cl_idx), 3) bool
        # pos_idx = np.argmax(pos_mask, axis=1) # (len(cl_idx),)
        
        # Other literals are at the other 2 positions.
        # We can just recompute sat counts for these specific clauses using the new assignment.
        # This is O(K) where K is occurrences of var.
        # Since we update sat_counts, we just need to recompute for affected clauses.
        
        # Update assignment
        assignment[var] = not assignment[var]
        
        # Recompute sat count for affected clauses
        # new_sat = calculate_sat_counts(assignment[indices[cl_idx_arr]])
        # But calculate_sat_counts expects full arrays. Let's make a helper or inline.
        
        val = np.take(assignment, indices[cl_idx_arr])
        is_sat = (val == (signs[cl_idx_arr] > 0))
        new_counts = np.sum(is_sat, axis=1)
        
        sat_counts[cl_idx_arr] = new_counts
        
        return assignment, sat_counts

    # Main Loop
    best_assignment = None
    best_unsat = m + 1
    
    # Adaptive parameters
    noise = 0.5
    flip_budget = 5000
    
    while time.time() < deadline:
        # Restart logic
        assignment = initialize_assignment()
        sat_counts = calculate_sat_counts(assignment)
        
        unsat_clauses = np.where(sat_counts == 0)[0]
        current_unsat = len(unsat_clauses)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            
        if current_unsat == 0:
            return assignment.tolist()
            
        steps = 0
        while steps < flip_budget and time.time() < deadline:
            if len(unsat_clauses) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_clauses[random.randint(0, len(unsat_clauses) - 1)]
            
            # Pick a literal to flip
            # WalkSAT heuristic: flip with probability noise randomly, otherwise pick best gain
            if random.random() < noise:
                # Random flip from clause
                pos = random.randint(0, 2)
                var = indices[clause_idx, pos]
            else:
                # Find var in clause that maximizes gain
                # Gain = (# unsat literals in other clauses containing var that become sat) - 
                #        (# sat literals in other clauses containing var that become unsat)
                
                best_gain = -100
                best_var = -1
                
                for pos in range(3):
                    var = indices[clause_idx, pos]
                    # Calculate gain for flipping var
                    # This requires checking all clauses containing var.
                    # We can approximate or calculate exactly.
                    # Exact calculation is expensive if var appears in many clauses.
                    # However, n=150, m is likely ~400-500 (3SAT ratio).
                    # Occurrences per var ~ 3*m/n ~ 10.
                    # So checking ~10 clauses is fast.
                    
                    gain = 0
                    for ci in var_to_clauses[var]:
                        if ci == clause_idx:
                            continue
                        # Check if this clause is currently satisfied
                        if sat_counts[ci] > 0:
                            # If it's satisfied, does it depend on var?
                            # If var is currently making it satisfied, flipping it might make it unsat.
                            # Find pos of var in ci
                            # If var is satisfied in ci, and we flip it, it becomes unsat.
                            # If other literals in ci are satisfied, it stays sat.
                            # So loss = 1 if (sat_counts[ci] == 1 and var is the only sat literal) else 0
                            # We need to know if var is the *only* satisfied literal.
                            # We don't have that info directly in sat_counts.
                            # We can check:
                            # val_ci = np.take(assignment, indices[ci])
                            # is_sat_ci = (val_ci == (signs[ci] > 0))
                            # if np.sum(is_sat_ci) == 1:
                            #    # check if var is the one
                            #    pos_var_in_ci = np.where(indices[ci] == var)[0]
                            #    if len(pos_var_in_ci) > 0 and is_sat_ci[pos_var_in_ci[0]]:
                            #        gain -= 1
                            
                            # Optimization: Only subtract if var is critical.
                            # Let's just do the full check for the few clauses var is in.
                            pass # Placeholder for logic below
                        else:
                            # Clause is unsat. If we flip var to satisfy it, gain +1.
                            # var is in clause ci.
                            # We flip var.
                            # Current value of var: assignment[var]
                            # Desired value for var in ci: (signs[ci, pos] > 0)
                            # If we flip, new value is not assignment[var].
                            # If (not assignment[var]) == desired, then we satisfy this literal.
                            # Since clause is unsat, satisfying one literal makes it sat (gain +1).
                            # Note: If multiple literals of var are in the clause? No, 3-SAT distinct vars usually.
                            
                            # Find pos of var in ci
                            # We can precompute positions?
                            # Let's just loop through the 3 literals of ci.
                            for k in range(3):
                                if indices[ci, k] == var:
                                    desired = signs[ci, k] > 0
                                    # New value of var after flip
                                    new_val = not assignment[var]
                                    if new_val == desired:
                                        gain += 1
                                    break
                            
                    # Now handle the loss for satisfied clauses
                    # We skipped the loss logic above. Let's implement it properly.
                    # Actually, let's just compute the delta directly.
                    # Old sat count for ci: sat_counts[ci]
                    # New sat count for ci: 
                    #   If we flip var, does the sat count change?
                    #   It changes by +1 if var was unsat and becomes sat.
                    #   It changes by -1 if var was sat and becomes unsat.
                    
                    # Let's re-calculate gain properly.
                    gain = 0
                    for ci in var_to_clauses[var]:
                        # Find pos of var in ci
                        # Since ci is small (3 literals), linear search is fine
                        pos_var = -1
                        for k in range(3):
                            if indices[ci, k] == var:
                                pos_var = k
                                break
                        if pos_var == -1:
                            continue
                        
                        # Current state of var in ci
                        desired_old = signs[ci, pos_var] > 0
                        old_sat = assignment[var] == desired_old
                        
                        # New state of var in ci
                        new_val = not assignment[var]
                        new_sat = new_val == desired_old
                        
                        if old_sat and not new_sat:
                            # Var was satisfying, now it's not.
                            # Clause ci loses a satisfied literal.
                            # If it had exactly 1 satisfied literal (which was var), it becomes unsat.
                            if sat_counts[ci] == 1:
                                gain -= 1
                        elif not old_sat and new_sat:
                            # Var was not satisfying, now it is.
                            # Clause ci gains a satisfied literal.
                            # If it had 0 satisfied literals, it becomes sat.
                            if sat_counts[ci] == 0:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = indices[clause_idx, 0]
                    
            # Perform flip
            assignment, sat_counts = flip_variable(var, assignment, sat_counts)
            
            # Update unsat_clauses list
            # We can just recompute it or update incrementally.
            # Recomputing is O(m). m is small.
            unsat_clauses = np.where(sat_counts == 0)[0]
            
            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                
            steps += 1
            
            # Adapt noise
            if steps % 100 == 0:
                noise = max(0.1, noise - 0.01)
                if current_unsat > m * 0.5: # If stuck in high unsat, increase noise?
                    noise = min(0.9, noise + 0.05)

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
