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
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
    
    # Variable occurrence lists for fast gain calculation
    # var_positions[var] = list of (clause_idx, pos_in_clause)
    var_positions = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_positions[v].append((i, j))
    
    # Convert var_positions to numpy arrays for faster access
    var_clause_idx = []
    var_pos_in_clause = []
    for v in range(n):
        if var_positions[v]:
            arr = np.array(var_positions[v], dtype=np.int32)
            var_clause_idx.append(arr[:, 0])
            var_pos_in_clause.append(arr[:, 1])
        else:
            var_clause_idx.append(np.empty(0, dtype=np.int32))
            var_pos_in_clause.append(np.empty(0, dtype=np.int32))
    
    def initialize_assignment():
        return np.random.rand(n) < 0.5

    def calculate_sat_counts(assignment):
        val = np.take(assignment, indices)
        is_sat = (val == (signs > 0))
        return np.sum(is_sat, axis=1)

    def flip_variable(var, assignment, sat_counts, unsat_set):
        """
        Flip variable var, update assignment, sat_counts, and unsat_set incrementally.
        Returns updated sat_counts and unsat_set.
        """
        ci_arr = var_clause_idx[var]
        pos_arr = var_pos_in_clause[var]
        
        if len(ci_arr) == 0:
            return sat_counts, unsat_set
        
        # Update assignment
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        # Vectorized update of sat counts for affected clauses
        old_counts = sat_counts[ci_arr].copy()
        
        # For each affected clause, check if the literal we're changing is satisfied
        # old_sat: was the literal satisfied before flip?
        # new_sat: is the literal satisfied after flip?
        lit_desired = signs[ci_arr, pos_arr] > 0  # shape (k,)
        old_sat = (old_val == lit_desired)  # shape (k,)
        new_sat = (new_val == lit_desired)  # shape (k,)
        
        # delta_sat: +1 if newly satisfied, -1 if newly unsatisfied, 0 otherwise
        delta_sat = np.zeros(len(ci_arr), dtype=np.int32)
        # Newly satisfied: was not sat, now sat
        newly_sat = (~old_sat) & new_sat
        # Newly unsatisfied: was sat, now not sat
        newly_unsat = old_sat & (~new_sat)
        delta_sat[newly_sat] = 1
        delta_sat[newly_unsat] = -1
        
        new_counts = old_counts + delta_sat
        sat_counts[ci_arr] = new_counts
        
        # Update unsat set
        # Clauses that became satisfied (old_count==0, new_count>0)
        became_sat = (old_counts == 0) & (new_counts > 0)
        if np.any(became_sat):
            for ci in ci_arr[became_sat].tolist():
                unsat_set.discard(ci)
        
        # Clauses that became unsatisfied (old_count>0, new_count==0)
        became_unsat = (old_counts > 0) & (new_counts == 0)
        if np.any(became_unsat):
            for ci in ci_arr[became_unsat].tolist():
                unsat_set.add(ci)
        
        return sat_counts, unsat_set

    # Main Loop
    best_assignment = None
    best_unsat = m + 1
    
    # Adaptive parameters
    base_noise = 0.3
    base_flip_budget = 50000
    restart_count = 0
    
    # Track best unsat for biasing restarts
    best_unsat_clauses_list = None
    
    while time.time() < deadline:
        # Restart logic: bias toward best assignment when stuck
        if restart_count == 0 or best_assignment is None:
            assignment = initialize_assignment()
        elif random.random() < 0.3:
            # Perturb best assignment
            assignment = best_assignment.copy()
            # Flip some random variables
            num_flips = max(1, n // 20)
            flip_vars = random.sample(range(n), num_flips)
            for v in flip_vars:
                assignment[v] = not assignment[v]
        else:
            assignment = initialize_assignment()
        
        sat_counts = calculate_sat_counts(assignment)
        
        # Build unsat set incrementally
        unsat_indices = np.where(sat_counts == 0)[0]
        unsat_set = set(unsat_indices.tolist())
        current_unsat = len(unsat_set)
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive flip budget
        if restart_count == 0:
            flip_budget = base_flip_budget
        else:
            flip_budget = int(base_flip_budget * (0.8 + 0.4 * random.random()))
        
        steps = 0
        no_improvement_steps = 0
        local_best_unsat = current_unsat
        noise = base_noise
        
        while steps < flip_budget and time.time() < deadline:
            if len(unsat_set) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            unsat_list = list(unsat_set)
            clause_idx = unsat_list[random.randint(0, len(unsat_list) - 1)]
            
            # Pick a literal to flip
            if random.random() < noise:
                # Random flip from clause
                pos = random.randint(0, 2)
                var = indices[clause_idx, pos]
            else:
                # Find var in clause that maximizes gain (vectorized)
                best_gain = -100
                best_var = -1
                
                for pos in range(3):
                    var = indices[clause_idx, pos]
                    
                    # Calculate gain for this variable
                    ci_arr = var_clause_idx[var]
                    pos_arr = var_pos_in_clause[var]
                    
                    if len(ci_arr) == 0:
                        gain = 0
                    else:
                        current_val = assignment[var]
                        new_val = not current_val
                        
                        # Vectorized gain calculation
                        lit_desired = signs[ci_arr, pos_arr] > 0
                        old_sat = (current_val == lit_desired)
                        new_sat = (new_val == lit_desired)
                        
                        # Clauses that will become unsatisfied (were sat, now not)
                        newly_unsat = old_sat & (~new_sat)
                        # Clauses that will become satisfied (were not sat, now sat)
                        newly_sat = (~old_sat) & new_sat
                        
                        # Only count if it changes the sat/unsat status
                        # A clause becomes unsat if it had exactly 1 sat literal and that one is lost
                        # A clause becomes sat if it had 0 sat literals and gains one
                        cur_counts = sat_counts[ci_arr]
                        
                        # Newly unsatisfied: was the only sat literal
                        becomes_unsat = newly_unsat & (cur_counts == 1)
                        # Newly satisfied: was the only way to satisfy
                        becomes_sat = newly_sat & (cur_counts == 0)
                        
                        gain = int(np.sum(becomes_sat)) - int(np.sum(becomes_unsat))
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = indices[clause_idx, 0]
                    
            # Perform flip
            sat_counts, unsat_set = flip_variable(var, assignment, sat_counts, unsat_set)
            
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improvement_steps = 0
            
            steps += 1
            
            # Adapt noise: decrease gradually, but increase if stuck
            if steps % 100 == 0:
                if no_improvement_steps > 300:
                    noise = min(0.7, noise + 0.05)
                else:
                    noise = max(0.1, noise - 0.02)
        
        restart_count += 1

    # Final polishing pass: if we're close to a solution, do intensive local search
    if best_assignment is not None and best_unsat <= max(5, m // 100):
        # Try to polish the best assignment found
        polish_deadline = deadline - 2.0
        assignment = best_assignment.copy()
        sat_counts = calculate_sat_counts(assignment)
        unsat_indices = np.where(sat_counts == 0)[0]
        unsat_set = set(unsat_indices.tolist())
        current_unsat = len(unsat_set)
        
        # Intensive local search with low noise
        polish_steps = 0
        max_polish_steps = 20000
        while polish_steps < max_polish_steps and time.time() < polish_deadline and len(unsat_set) > 0:
            if len(unsat_set) == 0:
                return assignment.tolist()
            
            unsat_list = list(unsat_set)
            clause_idx = unsat_list[random.randint(0, len(unsat_list) - 1)]
            
            # Always use best-gain flip for polishing
            best_gain = -100
            best_var = -1
            
            for pos in range(3):
                var = indices[clause_idx, pos]
                ci_arr = var_clause_idx[var]
                pos_arr = var_pos_in_clause[var]
                
                if len(ci_arr) == 0:
                    gain = 0
                else:
                    current_val = assignment[var]
                    new_val = not current_val
                    lit_desired = signs[ci_arr, pos_arr] > 0
                    old_sat = (current_val == lit_desired)
                    new_sat = (new_val == lit_desired)
                    newly_unsat = old_sat & (~new_sat)
                    newly_sat = (~old_sat) & new_sat
                    cur_counts = sat_counts[ci_arr]
                    becomes_unsat = newly_unsat & (cur_counts == 1)
                    becomes_sat = newly_sat & (cur_counts == 0)
                    gain = int(np.sum(becomes_sat)) - int(np.sum(becomes_unsat))
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            var = best_var
            if var == -1:
                var = indices[clause_idx, 0]
            
            sat_counts, unsat_set = flip_variable(var, assignment, sat_counts, unsat_set)
            current_unsat = len(unsat_set)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
            
            polish_steps += 1
    
    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()

# EVOLVE-BLOCK-END
