"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Satisfying assignment for a locally generated planted satisfiable 3-CNF.
    
    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans (length n) such that every clause is satisfied.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: evaluator kills at 330s; keep a safety margin.
    start = time.time()
    deadline = start + 285.0  # 45s margin
    
    # Precompute clause structure for fast evaluation
    # For each clause, store the variable indices and whether each literal is positive
    clause_vars = np.zeros((m, 3), dtype=np.int32)  # 0-based variable indices
    clause_signs = np.zeros((m, 3), dtype=np.int8)  # +1 for positive, -1 for negative
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            var = abs(lit) - 1  # 0-based
            sign = 1 if lit > 0 else -1
            clause_vars[i, j] = var
            clause_signs[i, j] = sign
    
    # Also store as flat arrays for faster access
    # For each clause position (i, j), we need var and sign
    # We'll use a different approach: for each variable, store which clause positions it appears in
    
    # Build variable-to-clause-position mapping
    # var_clause_pos[var] = list of (clause_idx, position_in_clause)
    var_clause_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var = clause_vars[i, j]
            var_clause_pos[var].append((i, j))
    
    # For faster lookup, convert to numpy arrays per variable
    var_clause_idx = [np.array([p[0] for p in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in var_clause_pos]
    var_clause_pos_arr = [np.array([p[1] for p in lst], dtype=np.int8) if lst else np.array([], dtype=np.int8) for lst in var_clause_pos]
    
    # State: assignment as boolean array
    # sat_count[i] = number of satisfied literals in clause i (0 to 3)
    # A clause is satisfied if sat_count[i] >= 1
    
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    def compute_sat_counts(assignment):
        """Compute satisfaction count for each clause given assignment."""
        sat_counts = np.zeros(m, dtype=np.int32)
        for var in range(n):
            if assignment[var]:
                # Positive literals are satisfied, negative are not
                cidx = var_clause_idx[var]
                cpos = var_clause_pos_arr[var]
                if len(cidx) > 0:
                    signs = clause_signs[cidx, cpos]
                    # Count where sign is +1
                    sat_counts[cidx] += (signs == 1).astype(np.int32)
            else:
                # Negative literals are satisfied, positive are not
                cidx = var_clause_idx[var]
                cpos = var_clause_pos_arr[var]
                if len(cidx) > 0:
                    signs = clause_signs[cidx, cpos]
                    # Count where sign is -1
                    sat_counts[cidx] += (signs == -1).astype(np.int32)
        return sat_counts
    
    def count_unsat(sat_counts):
        return int(np.sum(sat_counts == 0))
    
    def try_solve(assignment, max_steps):
        """Run WalkSAT-like local search from given assignment."""
        nonlocal best_assignment, best_unsat_count, deadline
        
        sat_counts = compute_sat_counts(assignment)
        unsat = count_unsat(sat_counts)
        
        if unsat == 0:
            best_assignment = assignment.copy()
            best_unsat_count = 0
            return True
        
        if unsat < best_unsat_count:
            best_unsat_count = unsat
            best_assignment = assignment.copy()
        
        # WalkSAT with clause selection and variable flipping
        # Use a noise parameter that adapts
        noise = 0.3
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = np.where(sat_counts == 0)[0]
            if len(unsat_clauses) == 0:
                best_assignment = assignment.copy()
                best_unsat_count = 0
                return True
            
            # Select a random unsatisfied clause
            clause_idx = rng.choice(unsat_clauses.tolist())
            
            # Get the three literals in this clause
            cvars = clause_vars[clause_idx]
            csigns = clause_signs[clause_idx]
            
            # Count how many literals are currently satisfied in this clause
            # Since the clause is unsatisfied, all 3 literals are unsatisfied
            # We need to pick a variable to flip
            
            # Strategy: with probability (1-noise), pick a variable that would satisfy the clause
            # (i.e., flip it to make its literal true)
            # With probability noise, pick a random variable from the clause
            
            # Find which variables, if flipped, would satisfy the clause
            # A literal is satisfied if assignment[var] == (sign == 1)
            # Currently all are unsatisfied, so for each var in clause:
            #   if sign == 1, we need assignment[var] = True, so flip if currently False
            #   if sign == -1, we need assignment[var] = False, so flip if currently True
            
            satisfying_vars = []
            for j in range(3):
                var = cvars[j]
                sign = csigns[j]
                if sign == 1:
                    # Need assignment[var] = True
                    if not assignment[var]:
                        satisfying_vars.append(var)
                else:
                    # Need assignment[var] = False
                    if assignment[var]:
                        satisfying_vars.append(var)
            
            if rng.random() < (1.0 - noise) and satisfying_vars:
                # Pick a satisfying variable
                var_to_flip = rng.choice(satisfying_vars)
            else:
                # Pick a random variable from the clause
                var_to_flip = rng.choice(cvars.tolist())
            
            # Flip the variable
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val
            
            # Update sat_counts for all clauses containing this variable
            cidx = var_clause_idx[var_to_flip]
            cpos = var_clause_pos_arr[var_to_flip]
            if len(cidx) > 0:
                signs = clause_signs[cidx, cpos]
                # For each affected clause, the old value contributed to sat_count
                # if the literal was satisfied. Now it's flipped, so:
                # old contribution: 1 if (old_val and sign==1) or (not old_val and sign==-1)
                # new contribution: 1 if (not old_val and sign==1) or (old_val and sign==-1)
                
                old_satisfied = np.where((old_val & (signs == 1)) | ((~old_val) & (signs == -1)), 1, 0).astype(np.int32)
                new_satisfied = np.where((~old_val & (signs == 1)) | (old_val & (signs == -1)), 1, 0).astype(np.int32)
                
                sat_counts[cidx] += (new_satisfied - old_satisfied)
            
            unsat = count_unsat(sat_counts)
            
            if unsat == 0:
                best_assignment = assignment.copy()
                best_unsat_count = 0
                return True
            
            if unsat < best_unsat_count:
                best_unsat_count = unsat
                best_assignment = assignment.copy()
        
        return best_unsat_count == 0
    
    # Try multiple random initial assignments with restarts
    num_restarts = 0
    while time.time() < deadline:
        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Determine step budget based on remaining time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Estimate time per step: roughly 100-200 microseconds per step
        # Be conservative: assume 200us per step
        max_steps = int(min(50000, max(1000, remaining * 4000)))
        
        solved = try_solve(assignment, max_steps)
        num_restarts += 1
        
        if solved:
            break
        
        # If we haven't made progress, try a different strategy
        if num_restarts % 5 == 0:
            # Try a greedy initialization: satisfy as many clauses as possible
            # by setting variables to satisfy the most unsatisfied clauses
            pass
    
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        # Fallback: return the last assignment tried
        return assignment.tolist()
# EVOLVE-BLOCK-END
