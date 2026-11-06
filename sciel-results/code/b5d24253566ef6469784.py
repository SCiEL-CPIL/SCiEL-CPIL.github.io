# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290
    
    # Preprocess
    m = len(clauses)
    # Convert to 0-based
    # Store as list of lists of (var, is_pos)
    # For NumPy, we can create a matrix of literals
    # But since clauses are small (size 3), a list of tuples might still be faster for the "flip" logic if we don't vectorize the flip.
    # However, finding unsat clauses is the bottleneck.
    
    # Let's use a hybrid: NumPy for counting, Python for the step.
    
    # Build arrays for fast NumPy evaluation
    # vars_matrix: shape (m, 3), values 0..n-1
    # signs_matrix: shape (m, 3), values 0 (negative) or 1 (positive)
    vars_matrix = np.zeros((m, 3), dtype=np.int32)
    signs_matrix = np.zeros((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            vars_matrix[i, j] = abs(l) - 1
            signs_matrix[i, j] = 1 if l > 0 else 0

    # Build variable index for quick access to clauses containing a variable
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[vars_matrix[i, j]].append(i)

    rng = random.Random(42)
    
    def get_unsat_clauses(assignment):
        """
        Returns list of indices of unsatisfied clauses.
        assignment is a list or numpy array of 0/1.
        """
        # assignment_np should be a numpy array of 0/1
        # lit_val = assignment[vars] == signs
        # clause_sat = any(lit_val)
        # We want clauses where clause_sat is False
        
        # Vectorized check
        # a = np.array(assignment)
        # matches = (a[vars_matrix] == signs_matrix)
        # sat_flags = matches.any(axis=1)
        # return np.where(~sat_flags)[0]
        
        # This creates arrays of size m.
        # Let's do it efficiently.
        a = np.asarray(assignment, dtype=np.int32)
        matches = (a[vars_matrix] == signs_matrix)
        sat_counts = matches.sum(axis=1)
        unsat_indices = np.where(sat_counts == 0)[0]
        return unsat_indices.tolist()

    def count_unsat(assignment):
        a = np.asarray(assignment, dtype=np.int32)
        matches = (a[vars_matrix] == signs_matrix)
        sat_counts = matches.sum(axis=1)
        return int(np.sum(sat_counts == 0))

    def unit_propagate(assignment):
        """
        Simple unit propagation.
        Returns modified assignment and success flag.
        """
        changed = False
        # Iterate until no change
        # We can do this iteratively, but for 3-SAT, a few passes are enough.
        # Let's do a queue-based approach for efficiency.
        
        # First, find all unit clauses in current assignment
        # A clause is unit if it has exactly one false literal (and 2 true) OR 
        # wait, in SAT, "unit" usually means 1 unassigned. Here all are assigned.
        # In local search context, "unit propagation" often means: 
        # If a clause is unsatisfied and has only 1 literal that is False, flipping it satisfies the clause.
        # If a clause is satisfied, we don't care.
        # So, find unsat clauses. If an unsat clause has only 1 false literal, that literal's variable MUST be flipped to satisfy this clause.
        
        # Let's implement a standard "repair" step:
        # While there are unsatisfied clauses:
        #   Pick an unsatisfied clause.
        #   If it has 1 false literal, flip it.
        #   Else, break (or handle in main loop).
        
        # To be efficient, we can do a few rounds of this.
        
        # Actually, the main loop handles the flips. We can just initialize with a random assignment and let the main loop do the work.
        # However, starting with a "good" assignment helps.
        
        # Let's try to satisfy as many clauses as possible greedily first?
        # No, just random start is standard for WalkSAT.
        
        return assignment, True

    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 1000
    base_steps = 1000
    step_growth = 1.1
    
    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # Random initial assignment
        assignment = np.random.randint(0, 2, n, dtype=np.int32).tolist()
        
        # Optional: Light local search / repair to get a good start
        # Let's do a few steps of "flip the variable that fixes the most unsat clauses"
        
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)
        
        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 20000)
        
        # WalkSAT Loop
        # Probability of random flip vs min-conflicts flip
        p_random = 0.5
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            unsat_indices = get_unsat_clauses(assignment)
            
            if not unsat_indices:
                return assignment
            
            current_unsat_count = len(unsat_indices)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = list(assignment)
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            
            # Get the literals in this clause
            c_vars = vars_matrix[ci]
            c_signs = signs_matrix[ci]
            
            # Identify false literals in this clause
            # A literal is false if assignment[var] != sign
            false_vars = []
            for k in range(3):
                v = c_vars[k]
                s = c_signs[k]
                if assignment[v] != s:
                    false_vars.append((v, s))
            
            if not false_vars:
                # Should not happen for an unsat clause
                continue
            
            # Choose which variable to flip
            if rng.random() < p_random:
                # Random pick
                var_to_flip, _ = rng.choice(false_vars)
            else:
                # Min-conflicts: pick the variable that, when flipped, results in the fewest newly unsatisfied clauses.
                # Flipping var v:
                # 1. Satisfies clause ci.
                # 2. May break other clauses where v was the ONLY satisfying literal.
                # 3. May satisfy other clauses where v was false and now becomes true.
                
                best_var = -1
                min_cost = float('inf')
                
                for v, s in false_vars:
                    # Calculate cost of flipping v
                    # Current state: v is assigned !s (since it's false in clause ci, and s is the sign required by ci)
                    # Wait, assignment[v] != s. So assignment[v] is 1-s.
                    # New state: assignment[v] becomes s.
                    
                    # Clauses containing v:
                    # For each clause c' in var_to_clauses[v]:
                    #   Check if c' is currently satisfied.
                    #   If yes, check if v is the ONLY satisfying literal. If so, flipping v breaks it. Cost +1.
                    #   If no, check if flipping v satisfies it. If so, cost -1 (or just track net change).
                    
                    # Efficiently:
                    # We can pre-calculate the "satisfaction count" for each clause?
                    # Or just check the specific clauses containing v.
                    
                    cost = 0
                    for ci2 in var_to_clauses[v]:
                        if ci2 == ci:
                            continue # We know this one becomes satisfied (cost -1 effectively, but we compare relative)
                        
                        c2_vars = vars_matrix[ci2]
                        c2_signs = signs_matrix[ci2]
                        
                        # Check if clause ci2 is currently satisfied
                        sat_now = False
                        only_sat_var = -1
                        sat_count = 0
                        for k in range(3):
                            vv = c2_vars[k]
                            ss = c2_signs[k]
                            if assignment[vv] == ss:
                                sat_now = True
                                sat_count += 1
                                only_sat_var = vv
                                
                        if sat_now:
                            # If v is the only satisfying literal, flipping it breaks the clause
                            if sat_count == 1 and only_sat_var == v:
                                cost += 1
                        else:
                            # Clause is unsatisfied.
                            # Check if flipping v satisfies it.
                            # It is satisfied if the literal (v, s) is present in ci2?
                            # No, flipping v to s means the literal (v, s) becomes true.
                            # So if ci2 contains (v, s), it becomes satisfied.
                            # We need to check if ci2 contains v with sign s.
                            for k in range(3):
                                if c2_vars[k] == v and c2_signs[k] == s:
                                    cost -= 1
                                    break
                            
                    if cost < min_cost:
                        min_cost = cost
                        best_var = v
                
                if best_var == -1:
                    var_to_flip = false_vars[0][0]
                else:
                    var_to_flip = best_var
            
            # Perform the flip
            assignment[var_to_flip] = 1 - assignment[var_to_flip]
            
        restart += 1

    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
