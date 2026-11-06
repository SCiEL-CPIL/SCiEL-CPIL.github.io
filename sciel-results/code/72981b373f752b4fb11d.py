"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 315.0  # Safety margin before 330s limit
    
    if m == 0:
        return [True] * n
    
    # Preprocess: build clause array for vectorized operations
    # clauses is a list of lists of signed 1-based literals
    # Convert to numpy array with padding for vectorization
    max_clause_len = 3
    clause_arr = np.zeros((m, max_clause_len), dtype=np.int32)
    clause_len_arr = np.zeros(m, dtype=np.int32)
    
    for i, clause in enumerate(clauses):
        clause_len_arr[i] = len(clause)
        for j, lit in enumerate(clause):
            clause_arr[i, j] = lit
    
    def check_satisfied(assignment):
        """Check if all clauses are satisfied. assignment is a numpy bool array of length n."""
        # For each clause, check if any literal is true
        # Literal l is true if (l > 0 and assignment[abs(l)-1]) or (l < 0 and not assignment[abs(l)-1])
        
        # Build literal truth values
        # For each literal in each clause position
        abs_lits = np.abs(clause_arr)  # (m, 3)
        signs = np.sign(clause_arr)  # (m, 3)
        
        # Get variable values
        # Need to handle -1 padding (literal 0 means not present)
        var_idx = abs_lits - 1  # (m, 3)
        # Mask out invalid positions
        valid_mask = (abs_lits > 0)  # (m, 3)
        
        # Clamp var_idx to valid range for indexing
        safe_var_idx = np.clip(var_idx, 0, n - 1)
        var_values = assignment[safe_var_idx]  # (m, 3)
        
        # For positive literals, truth = var_values
        # For negative literals, truth = ~var_values
        # For padding (0), truth = False
        literal_truth = np.where(signs > 0, var_values, 
                                  np.where(signs < 0, ~var_values, 
                                           False))
        # Zero out padding positions
        literal_truth = np.where(valid_mask, literal_truth, False)
        
        # Clause is satisfied if any literal is true
        clause_sat = np.any(literal_truth, axis=1)  # (m,)
        return clause_sat
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        clause_sat = check_satisfied(assignment)
        return np.sum(~clause_sat)
    
    # Phase 1: Try random initialization and simple WalkSAT with vectorized evaluation
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # Try multiple random restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        assignment = np.random.random(n) < 0.5
        assignment = assignment.astype(bool)
        
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # WalkSAT with adaptive parameters
        # Use a mix of greedy and random flips
        max_steps = 5000
        p_greedy = 0.5
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            clause_sat = check_satisfied(assignment)
            unsat_mask = ~clause_sat
            unsat_count = np.sum(unsat_mask)
            
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            clause = clause_arr[clause_idx]
            clause_len = clause_len_arr[clause_idx]
            
            # Get literals in this clause
            lit_indices = []
            for j in range(clause_len):
                lit = clause[j]
                if lit != 0:
                    lit_indices.append(lit)
            
            if not lit_indices:
                continue
            
            # Decide flip strategy
            if rng.random() < p_greedy:
                # Greedy: pick literal that maximizes gain
                best_var = None
                best_gain = -1
                for lit in lit_indices:
                    var_idx = abs(lit) - 1
                    # Calculate gain if we flip this variable
                    # Need to check impact on all clauses containing this variable
                    # For simplicity in this phase, just pick randomly from clause
                    # But let's do a quick local gain calculation
                    gain = 0
                    # This variable is currently:
                    current_val = assignment[var_idx]
                    # If we flip it, how many clauses does it satisfy/unsatisfy?
                    # For each clause containing this variable:
                    # If the literal in that clause is currently false, flipping might satisfy it
                    # If the literal is currently true, flipping might unsatisfy it
                    # But we only care about clauses that are currently unsatisfied
                    # For now, just pick randomly
                    pass
                
                # Simplified greedy: pick a random literal from the clause
                lit = lit_indices[rng.randint(0, len(lit_indices) - 1)]
            else:
                # Random: pick any variable
                lit = lit_indices[rng.randint(0, len(lit_indices) - 1)]
            
            var_idx = abs(lit) - 1
            assignment[var_idx] = not assignment[var_idx]
        
        # After each restart, try a polish phase on the best assignment
        if best_assignment is not None and best_unsat_count > 0 and best_unsat_count <= 5:
            if time.time() > deadline:
                break
            # Try targeted repair
            polish_assignment = best_assignment.copy()
            for polish_step in range(1000):
                if time.time() > deadline:
                    break
                clause_sat = check_satisfied(polish_assignment)
                unsat_mask = ~clause_sat
                unsat_count = np.sum(unsat_mask)
                if unsat_count == 0:
                    return polish_assignment.tolist()
                
                # Pick an unsatisfied clause
                unsat_indices = np.where(unsat_mask)[0]
                if len(unsat_indices) == 0:
                    break
                clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
                clause = clause_arr[clause_idx]
                clause_len = clause_len_arr[clause_idx]
                
                lit_indices = []
                for j in range(clause_len):
                    lit = clause[j]
                    if lit != 0:
                        lit_indices.append(lit)
                
                if not lit_indices:
                    continue
                
                # Try flipping each variable and see if it reduces unsat count
                best_gain = 0
                best_var = None
                for lit in lit_indices:
                    var_idx = abs(lit) - 1
                    old_val = polish_assignment[var_idx]
                    polish_assignment[var_idx] = not old_val
                    new_unsat = count_unsat(polish_assignment)
                    gain = unsat_count - new_unsat
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var_idx
                    polish_assignment[var_idx] = old_val
                
                if best_var is not None and best_gain > 0:
                    polish_assignment[best_var] = not polish_assignment[best_var]
                else:
                    # No improvement, try random flip
                    lit = lit_indices[rng.randint(0, len(lit_indices) - 1)]
                    var_idx = abs(lit) - 1
                    polish_assignment[var_idx] = not polish_assignment[var_idx]
            
            # Update best if improved
            final_unsat = count_unsat(polish_assignment)
            if final_unsat < best_unsat_count:
                best_unsat_count = final_unsat
                best_assignment = polish_assignment.copy()
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
