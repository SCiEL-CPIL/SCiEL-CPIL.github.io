"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    timeout = 315.0  # Leave margin for overhead
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays
    # clause_sgn: shape (m, 3), values are +1 or -1
    # clause_var: shape (m, 3), values are var indices (0-based)
    clause_sgn = np.empty((m, 3), dtype=np.int8)
    clause_var = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_sgn[i, j] = 1 if lit > 0 else -1
            clause_var[i, j] = abs(lit) - 1
    
    # Build variable-to-clause incidence
    # For each variable, list of (clause_idx, position)
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_var[i, j]
            var_clauses[v].append((i, j))
    
    # Precompute for each variable: array of clause indices and signs
    var_clause_idx = [np.array([x[0] for x in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in var_clauses]
    var_clause_pos = [np.array([x[1] for x in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in var_clauses]
    
    def evaluate(assignment):
        """Return unsatisfied clause indices."""
        # assignment: boolean array of length n
        # For each clause, check if any literal is satisfied
        # lit_val = clause_sgn[i] * (2*assignment[clause_var[i]] - 1)
        # satisfied if any lit_val == 1
        vals = clause_sgn * (2.0 * assignment[clause_var] - 1.0)
        sat = np.any(vals > 0, axis=1)
        return np.where(~sat)[0]
    
    def is_satisfied(assignment):
        return len(evaluate(assignment)) == 0
    
    def find_best_flip(assignment, unsat_clauses):
        """Find variable to flip that minimizes number of unsatisfied clauses."""
        if len(unsat_clauses) == 0:
            return None
        
        # Pick a random unsatisfied clause
        clause_idx = unsat_clauses[np.random.randint(len(unsat_clauses))]
        
        # For each variable in this clause, compute the change in unsat count
        best_var = -1
        best_score = float('inf')
        best_ties = []
        
        for j in range(3):
            v = clause_var[clause_idx, j]
            sgn = clause_sgn[clause_idx, j]
            
            # Current state of this literal
            current_val = 2.0 * assignment[v] - 1.0
            # After flip, the value negates
            # The clause is currently unsatisfied, so flipping v changes its status
            # We need to count how many clauses become unsatisfied vs satisfied
            
            # Clauses containing variable v
            cidx = var_clause_idx[v]
            cpos = var_clause_pos[v]
            
            if len(cidx) == 0:
                continue
            
            # Get signs and other variables for these clauses
            signs = clause_sgn[cidx, cpos]  # sign of v in each clause
            other_vars = clause_var[cidx]  # all 3 vars in each clause
            
            # For each clause, check if it's currently satisfied (by other literals)
            # A clause is satisfied if any literal is true
            # After flipping v, the literal for v changes sign
            # We need to check: was the clause satisfied by v? Is it satisfied by others?
            
            # Current values of all literals in these clauses
            all_vals = clause_sgn[cidx] * (2.0 * assignment[other_vars] - 1.0)
            # all_vals shape: (len(cidx), 3)
            
            # Currently satisfied?
            currently_sat = np.any(all_vals > 0, axis=1)
            
            # After flipping v, the value of v's literal negates
            # new_all_vals = all_vals.copy(); new_all_vals[:, pos_of_v] *= -1
            # But we need to know the position of v in each clause
            # cpos gives us the position of v in each clause
            new_all_vals = all_vals.copy()
            rows = np.arange(len(cidx))
            new_all_vals[rows, cpos] *= -1
            
            new_sat = np.any(new_all_vals > 0, axis=1)
            
            # Change in unsat count:
            # Clauses that were unsat and become sat: -1 each
            # Clauses that were sat and become unsat: +1 each
            delta = np.sum(new_sat) - np.sum(currently_sat)
            # delta = (new_sat_count - old_sat_count) = -(new_unsat - old_unsat)
            # So change in unsat = -delta
            
            if delta > best_score:  # maximizing delta = minimizing unsat change
                best_score = delta
                best_var = v
                best_ties = [v]
            elif delta == best_score:
                best_ties.append(v)
        
        if best_var == -1:
            # Random flip from the clause
            v = clause_var[clause_idx, np.random.randint(3)]
            return v
        elif len(best_ties) > 1:
            return best_ties[np.random.randint(len(best_ties))]
        else:
            return best_var
    
    # WalkSAT with restarts and adaptive parameters
    rng = np.random.RandomState(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 500
    base_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Random initial assignment
        assignment = (rng.rand(n) < 0.5).astype(np.float64)
        # Convert to 0/1 for internal use, but we work with float for vectorization
        
        unsat = evaluate(assignment)
        unsat_count = len(unsat)
        
        if unsat_count == 0:
            return assignment.astype(bool).tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # Adaptive step budget
        steps = base_steps + restart * 500
        
        # WalkSAT loop
        for step in range(steps):
            if time.time() - start_time > timeout:
                break
            
            if len(unsat) == 0:
                return assignment.astype(bool).tolist()
            
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment.copy()
            
            # Choose flip
            # Use score-based selection with some randomness
            clause_idx = unsat[rng.randint(len(unsat))]
            
            # Try all 3 variables in the clause, pick best
            best_v = -1
            best_delta = float('-inf')
            candidates = []
            
            for j in range(3):
                v = clause_var[clause_idx, j]
                cidx = var_clause_idx[v]
                cpos = var_clause_pos[v]
                
                if len(cidx) == 0:
                    continue
                
                signs = clause_sgn[cidx, cpos]
                other_vars = clause_var[cidx]
                
                all_vals = clause_sgn[cidx] * (2.0 * assignment[other_vars] - 1.0)
                currently_sat = np.any(all_vals > 0, axis=1)
                
                new_all_vals = all_vals.copy()
                rows = np.arange(len(cidx))
                new_all_vals[rows, cpos] *= -1
                new_sat = np.any(new_all_vals > 0, axis=1)
                
                delta = np.sum(new_sat) - np.sum(currently_sat)
                
                if delta > best_delta:
                    best_delta = delta
                    best_v = v
                    candidates = [v]
                elif delta == best_delta:
                    candidates.append(v)
            
            if best_v == -1:
                v = clause_var[clause_idx, rng.randint(3)]
            elif len(candidates) > 1:
                v = candidates[rng.randint(len(candidates))]
            else:
                v = best_v
            
            assignment[v] = 1.0 - assignment[v]
            
            # Update unsat clauses efficiently
            # Only clauses containing v can change status
            cidx = var_clause_idx[v]
            if len(cidx) == 0:
                unsat = evaluate(assignment)
            else:
                # Recompute only affected clauses
                all_vals = clause_sgn[cidx] * (2.0 * assignment[clause_var[cidx]] - 1.0)
                sat = np.any(all_vals > 0, axis=1)
                
                # Update the unsat set
                # Clauses in cidx that are now sat should be removed from unsat
                # Clauses in cidx that are now unsat should be added to unsat
                
                # Create a set of unsat clause indices for O(1) lookup
                unsat_set = set(unsat.tolist())
                
                for k, ci in enumerate(cidx):
                    if sat[k]:
                        unsat_set.discard(int(ci))
                    else:
                        unsat_set.add(int(ci))
                
                unsat = np.array(list(unsat_set), dtype=np.int32)
        
        # Check after walk
        if len(unsat) == 0:
            return assignment.astype(bool).tolist()
        if len(unsat) < best_unsat_count:
            best_unsat_count = len(unsat)
            best_assignment = assignment.copy()
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.astype(bool).tolist()
    else:
        return (rng.rand(n) < 0.5).tolist()
# EVOLVE-BLOCK-END
