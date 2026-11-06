"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # Preprocess for NumPy
    # Convert to 0-based indices
    # lits: (M, 3) array of variable indices (0-based)
    # signs: (M, 3) array of booleans (True for positive, False for negative)
    M = len(clauses)
    lits = np.zeros((M, 3), dtype=np.int32)
    signs = np.zeros((M, 3), dtype=bool)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            lits[i, j] = abs(lit) - 1
            signs[i, j] = lit > 0

    # Build mapping from variable to list of clause indices it appears in
    # var_to_clauses[var] = list of clause indices
    var_to_clauses = [[] for _ in range(n)]
    for i in range(M):
        for j in range(3):
            v = lits[i, j]
            var_to_clauses[v].append(i)

    rng = random.Random(42)
    
    # Phase saving
    phase = [False] * n
    
    # Initial assignment
    # Use phase saving or random
    assignment = np.array([phase[i] if rng.random() < 0.8 else rng.random() < 0.5 for i in range(n)], dtype=bool)
    
    # Initialize satisfied array
    # satisfied[i] is True if clause i is satisfied
    def compute_satisfied(assignment):
        # Get values of variables in clauses
        vals = assignment[lits] # (M, 3) boolean
        # Check if literal matches
        # If sign is True (positive), we need val to be True.
        # If sign is False (negative), we need val to be False.
        # So match = (vals == signs)
        matches = vals == signs # (M, 3) boolean
        # Clause is satisfied if any literal matches
        return np.any(matches, axis=1) # (M,) boolean

    satisfied = compute_satisfied(assignment)
    unsat_count = np.sum(~satisfied)
    
    if unsat_count == 0:
        return assignment.tolist()
    
    best_unsat = unsat_count
    best_assignment = assignment.copy()
    
    max_flips = 1000000 # 1 million flips
    p = 0.5 # Probability of random flip
    
    # Precompute unsat clause indices for fast access
    # We will maintain a list of unsat clause indices
    
    for flip in range(max_flips):
        if unsat_count == 0:
            return assignment.tolist()
        
        # Pick a random unsatisfied clause
        unsat_indices = np.where(~satisfied)[0]
        if len(unsat_indices) == 0:
            return assignment.tolist()
            
        ci = rng.choice(unsat_indices.tolist())
        clause_lits = lits[ci]
        clause_signs = signs[ci]
        
        # Choose variable to flip
        if rng.random() < p:
            # Random variable from the clause
            j = rng.randint(0, 2)
            v = clause_lits[j]
        else:
            # Choose variable that minimizes unsat count (maximizes gain)
            best_v = -1
            best_unsat = unsat_count + 1 # Infinity
            
            for j in range(3):
                v = clause_lits[j]
                # Calculate gain by flipping v
                # Only clauses containing v are affected
                affected_clauses = var_to_clauses[v]
                
                # Current unsat count is known.
                # We need to calculate the change in unsat count.
                # For each affected clause, check if it's currently satisfied, 
                # and if it will be satisfied after flip.
                
                # Current state of affected clauses
                cur_sat = satisfied[affected_clauses]
                
                # New value of v
                new_val = not assignment[v]
                
                # For each affected clause, determine if the literal for v in that clause is satisfied by new_val
                # We need to know the sign of v in each affected clause.
                # This is tricky because v might appear multiple times in a clause? No, 3-SAT usually distinct.
                # But v appears in multiple clauses.
                
                # Let's construct the new satisfaction for the affected clauses.
                # We need the specific literal value for v in each affected clause.
                # We can get the signs for v in those clauses.
                
                # Find positions of v in the affected clauses? 
                # Or simpler: 
                # For each affected clause index `k`:
                #   Check if clause k is satisfied by other literals (not v).
                #   If yes, it stays satisfied.
                #   If no, it depends on the literal for v.
                
                # This "other literals" check is expensive if done per clause in Python.
                # Let's vectorize it.
                
                # Get all literals in affected clauses
                aff_lits = lits[affected_clauses] # (K, 3)
                aff_signs = signs[affected_clauses] # (K, 3)
                
                # Get values of these literals
                aff_vals = assignment[aff_lits] # (K, 3)
                aff_matches = aff_vals == aff_signs # (K, 3)
                
                # Identify which position in the clause is variable v
                # v_pos: (K, 3) boolean mask where True if aff_lits[k, j] == v
                v_pos = aff_lits == v
                
                # Satisfied by other literals: any match in columns where v_pos is False
                other_matches = np.where(v_pos, False, aff_matches)
                sat_by_other = np.any(other_matches, axis=1)
                
                # For clauses not satisfied by others, check if the literal for v is satisfied by new_val
                # The literal for v in clause k is at position where v_pos[k, :] is True
                # We need the sign of that literal.
                # v_sign = aff_signs[aff_lits == v] -> This doesn't work directly for 2D.
                
                # Let's extract the sign of v in each affected clause.
                # Since v appears exactly once in each clause (assumption for standard 3-SAT),
                # we can sum the signs where v_pos is True? No, signs are bool.
                # We can use: v_sign_mask = v_pos & aff_signs
                # Then v_is_positive = np.any(v_sign_mask, axis=1)
                
                v_is_positive = np.any(v_pos & aff_signs, axis=1)
                
                # Literal is satisfied if (v_is_positive AND new_val) OR (NOT v_is_positive AND NOT new_val)
                lit_satisfied = (v_is_positive == new_val) # If new_val is True, pos lit is sat. If new_val False, neg lit is sat.
                # Wait: 
                # If v_is_positive (True) and new_val (True) -> Sat.
                # If v_is_positive (True) and new_val (False) -> Unsat.
                # If v_is_positive (False) and new_val (True) -> Unsat.
                # If v_is_positive (False) and new_val (False) -> Sat.
                # So lit_satisfied = (v_is_positive == new_val) is correct.
                
                # New satisfaction for affected clauses:
                new_sat = sat_by_other | lit_satisfied
                
                # Calculate change in unsat count
                # Old unsat in affected: sum(~cur_sat)
                # New unsat in affected: sum(~new_sat)
                old_unsat_aff = np.sum(~cur_sat)
                new_unsat_aff = np.sum(~new_sat)
                
                # Total new unsat = global_unsat - old_unsat_aff + new_unsat_aff
                new_total_unsat = unsat_count - old_unsat_aff + new_unsat_aff
                
                if new_total_unsat < best_unsat:
                    best_unsat = new_total_unsat
                    best_v = v
            
            v = best_v
        
        # Perform the flip
        old_val = assignment[v]
        assignment[v] = not old_val
        phase[v] = assignment[v]
        
        # Update satisfied array and unsat_count
        affected_clauses = var_to_clauses[v]
        
        # Recompute satisfaction for affected clauses
        aff_lits = lits[affected_clauses]
        aff_signs = signs[affected_clauses]
        aff_vals = assignment[aff_lits]
        aff_matches = aff_vals == aff_signs
        new_sat = np.any(aff_matches, axis=1)
        
        # Update global satisfied
        old_sat = satisfied[affected_clauses]
        
        # Update unsat count
        old_unsat_aff = np.sum(~old_sat)
        new_unsat_aff = np.sum(~new_sat)
        unsat_count = unsat_count - old_unsat_aff + new_unsat_aff
        
        # Update satisfied array
        satisfied[affected_clauses] = new_sat
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
            
        # Early termination if we hit 0
        if unsat_count == 0:
            return assignment.tolist()

    return best_assignment.tolist()
# EVOLVE-BLOCK-END
