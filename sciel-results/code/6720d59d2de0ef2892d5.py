"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    # Deadline guard: 330s wall-clock, use 325s for safety
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess clauses into numpy arrays
    # clause_signs: shape (m, 3), values are +1 or -1
    # clause_vars: shape (m, 3), values are 0-based variable indices
    clause_signs = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_signs[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause incidence for efficient tracking
    # For each variable, store list of (clause_idx, sign, position)
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append((i, clause_signs[i, j], j))
    
    # Random initial assignment
    rng = np.random.RandomState(42)
    assignment = rng.rand(n) < 0.5  # True/False array
    
    # Compute initial clause satisfaction
    # clause_sat[i] = number of satisfied literals in clause i
    clause_sat = np.zeros(m, dtype=np.int8)
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            s = clause_signs[i, j]
            if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                clause_sat[i] += 1
    
    def is_satisfied():
        return np.all(clause_sat > 0)
    
    def evaluate_flip(var_idx):
        """Return the change in total unsatisfied count if we flip var_idx.
        Positive means more unsatisfied (bad), negative means fewer (good)."""
        # For each clause containing var_idx, check if flipping helps or hurts
        # A clause is satisfied if sat_count > 0.
        # If we flip var_idx, for each clause containing it:
        #   - If the literal was satisfied before and not after: sat_count decreases by 1
        #   - If the literal was not satisfied before and is after: sat_count increases by 1
        # Net effect on total unsatisfied = sum over clauses of (change in unsat)
        # A clause goes from unsat to sat: -1 to total unsat
        # A clause goes from sat to unsat: +1 to total unsat
        
        total_change = 0
        for (ci, sign, pos) in var_clauses[var_idx]:
            was_sat = (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1)
            now_sat = (not assignment[var_idx] and sign == 1) or (assignment[var_idx] and sign == -1)
            
            if was_sat and not now_sat:
                # This literal was contributing to satisfaction
                if clause_sat[ci] == 1:
                    # Clause was satisfied only by this literal -> becomes unsat
                    total_change += 1
                else:
                    # Clause still satisfied by other literals
                    total_change += 0
            elif not was_sat and now_sat:
                # This literal now satisfies the clause
                if clause_sat[ci] == 0:
                    # Clause was unsat -> becomes sat
                    total_change -= 1
                else:
                    # Clause already satisfied
                    total_change += 0
            # else: no change
        return total_change
    
    def apply_flip(var_idx):
        """Flip var_idx and update clause_sat."""
        for (ci, sign, pos) in var_clauses[var_idx]:
            was_sat = (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1)
            now_sat = (not assignment[var_idx] and sign == 1) or (assignment[var_idx] and sign == -1)
            if was_sat and not now_sat:
                clause_sat[ci] -= 1
            elif not was_sat and now_sat:
                clause_sat[ci] += 1
        assignment[var_idx] = not assignment[var_idx]
    
    # WalkSAT with smart variable selection
    # Track best assignment found
    best_assignment = assignment.copy()
    best_unsat = np.sum(clause_sat == 0)
    
    max_steps = 500000
    restarts = 0
    max_restarts = 20
    
    while time.time() < deadline:
        # Restart if stuck
        if restarts >= max_restarts:
            break
        
        # Random restart
        if restarts > 0:
            assignment = rng.rand(n) < 0.5
            clause_sat = np.zeros(m, dtype=np.int8)
            for i in range(m):
                for j in range(3):
                    v = clause_vars[i, j]
                    s = clause_signs[i, j]
                    if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                        clause_sat[i] += 1
            restarts += 1
        
        # Main search loop
        for step in range(max_steps):
            if time.time() >= deadline - 5.0:
                break
            
            unsat_mask = (clause_sat == 0)
            unsat_count = np.sum(unsat_mask)
            
            if unsat_count == 0:
                # Found satisfying assignment
                return assignment.tolist()
            
            # Update best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            # Choose an unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            ci = unsat_indices[rng.randint(0, len(unsat_indices))]
            
            # Choose variable to flip from this clause
            # Use a weighted heuristic: prefer variables that appear in many unsat clauses
            # and where flipping would reduce unsat count
            best_var = -1
            best_score = float('inf')
            
            for j in range(3):
                v = clause_vars[ci, j]
                score = evaluate_flip(v)
                # Add small random tiebreaker
                score += rng.rand() * 0.1
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var == -1:
                # Fallback: random variable from clause
                best_var = clause_vars[ci, rng.randint(0, 2)]
            
            apply_flip(best_var)
        
        # If we exhausted steps without solving, restart
        restarts += 1
    
    # Return best assignment found
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
