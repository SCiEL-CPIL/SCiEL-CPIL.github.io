"""
Improved 3-SAT solver using Score-based WalkSAT with Phase Heuristic Initialization.
"""
import numpy as np
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    deadline = time.time() + 285  # 15s margin for safety
    
    # Precomputation
    # clause_vars: shape (m, 3), 0-indexed variable indices
    # clause_signs: shape (m, 3), 1 for positive, -1 for negative
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int32)
    
    # var_to_clauses: list of lists, var_to_clauses[v] contains indices of clauses containing v
    var_to_clauses = [[] for _ in range(n)]
    
    # Phase counts for initialization
    pos_count = np.zeros(n, dtype=np.int32)
    neg_count = np.zeros(n, dtype=np.int32)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            clause_vars[i, j] = v
            if lit > 0:
                clause_signs[i, j] = 1
                pos_count[v] += 1
            else:
                clause_signs[i, j] = -1
                neg_count[v] += 1
            var_to_clauses[v].append(i)

    # Helper to evaluate clause satisfaction for a given assignment
    # assignment: 0/1 array of length n
    # Returns: boolean array of length m (True if sat)
    def get_clause_satisfaction(assignment):
        # lit_val = sign * (2*val - 1)
        # If sign=1, val=1 -> 1*1=1 (True)
        # If sign=1, val=0 -> 1*(-1)=-1 (False)
        # If sign=-1, val=1 -> -1*1=-1 (False)
        # If sign=-1, val=0 -> -1*(-1)=1 (True)
        # So literal is true if sign * (2*val - 1) == 1
        # Which is equivalent to: (val == 1 and sign == 1) or (val == 0 and sign == -1)
        # Or simply: val == (1 if sign==1 else 0)
        
        # Vectorized check:
        # For each clause i, check if any literal j is true.
        # lit_true[i, j] = (assignment[clause_vars[i, j]] == (1 if clause_signs[i, j]==1 else 0))
        
        # Let's use the arithmetic trick:
        # val is 0 or 1. sign is 1 or -1.
        # target_val = (sign + 1) // 2. (1->1, -1->0)
        # lit_true = (assignment[clause_vars] == target_val)
        
        target_vals = (clause_signs + 1) // 2
        lit_vals = assignment[clause_vars]
        lit_true = (lit_vals == target_vals)
        
        # Clause is sat if any literal is true
        clause_sat = lit_true.any(axis=1)
        return clause_sat

    # Main solver loop
    best_unsat = m + 1
    best_assignment = None
    
    # Parameters
    max_restarts = 1000
    max_steps_per_restart = 5000
    p_walk = 0.2  # Probability of random walk step
    
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment using Phase Heuristic
        # If pos_count[v] > neg_count[v], set True (1), else False (0)
        # If equal, random
        assignment = np.zeros(n, dtype=np.int32)
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                assignment[v] = 1
            elif neg_count[v] > pos_count[v]:
                assignment[v] = 0
            else:
                assignment[v] = rng.randint(0, 1)
        
        # Initial evaluation
        clause_sat = get_clause_satisfaction(assignment)
        unsat_count = m - clause_sat.sum()
        
        if unsat_count == 0:
            return assignment.tolist()
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
            
        # Local Search
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment.tolist()
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
                
            # Pick a random unsatisfied clause
            unsat_indices = np.where(~clause_sat)[0]
            c_idx = unsat_indices[rng.randint(len(unsat_indices))]
            
            # Get variables in this clause
            c_vars = clause_vars[c_idx]
            
            # Score each variable in the clause
            # Score = Delta in unsat_count if we flip this variable
            # We want to minimize the score (most negative is best)
            
            best_var = -1
            best_score = float('inf')
            candidates = []
            
            for j in range(3):
                v = c_vars[j]
                
                # Calculate score for flipping v
                # Current state of v: assignment[v]
                # New state: 1 - assignment[v]
                
                # Iterate through all clauses containing v
                delta = 0
                
                for k_idx in var_to_clauses[v]:
                    # Check if clause k_idx is currently satisfied
                    was_sat = clause_sat[k_idx]
                    
                    # Determine if clause k_idx will be satisfied after flipping v
                    # We need to check if any other literal in k_idx is true,
                    # or if the literal for v becomes true.
                    
                    # Literal for v in clause k_idx:
                    # Find position of v in clause k_idx
                    # Since we know v is in k_idx, we can find its sign
                    # But we stored clause_signs[k_idx] as an array of 3 signs.
                    # We need to map v to its sign in this clause.
                    
                    # Optimization: Precompute var_sign_in_clause?
                    # Or just check:
                    # The literal for v in clause k_idx is true if:
                    # (new_val == 1 and sign == 1) or (new_val == 0 and sign == -1)
                    # Let's find the sign of v in clause k_idx.
                    
                    # We can search for v in clause_vars[k_idx]
                    # Since m is small, linear search is fine, but we can do better.
                    # Let's just iterate the 3 literals of clause k_idx.
                    
                    new_val = 1 - assignment[v]
                    is_v_lit_true = False
                    for jj in range(3):
                        if clause_vars[k_idx, jj] == v:
                            sign = clause_signs[k_idx, jj]
                            # Check if literal is true with new_val
                            if (sign == 1 and new_val == 1) or (sign == -1 and new_val == 0):
                                is_v_lit_true = True
                            break
                    
                    # Check if any OTHER literal in k_idx is true
                    # We can use the current clause_sat if the clause was sat, but we need to know if it STAYS sat.
                    # If was_sat is True:
                    #   It stays sat if (is_v_lit_true is True) OR (any other lit is true).
                    #   It becomes unsat if (is_v_lit_true is False) AND (no other lit is true).
                    #   "No other lit is true" means the clause was satisfied ONLY by v.
                    
                    # If was_sat is False:
                    #   It becomes sat if is_v_lit_true is True.
                    #   It stays unsat if is_v_lit_true is False.
                    
                    # To be precise, let's check all literals in k_idx.
                    any_other_true = False
                    for jj in range(3):
                        if clause_vars[k_idx, jj] == v:
                            continue
                        var_jj = clause_vars[k_idx, jj]
                        sign_jj = clause_signs[k_idx, jj]
                        val_jj = assignment[var_jj]
                        if (sign_jj == 1 and val_jj == 1) or (sign_jj == -1 and val_jj == 0):
                            any_other_true = True
                            break
                    
                    now_sat = is_v_lit_true or any_other_true
                    
                    if was_sat and not now_sat:
                        delta += 1  # Became unsat
                    elif not was_sat and now_sat:
                        delta -= 1  # Became sat
                    # else: no change
                    
                # The clause c_idx itself:
                # It was unsat. After flipping v, is it sat?
                # Yes, because v is in c_idx and we are flipping it.
                # Wait, does flipping v always satisfy c_idx?
                # No, only if the literal for v in c_idx becomes true.
                # But we are picking v from an unsat clause c_idx.
                # If we flip v, does the literal for v in c_idx become true?
                # Yes, because c_idx is unsat, so all literals are false.
                # If we flip v, the literal for v (which was false) becomes true?
                # Let's check:
                # If lit was false:
                #   Case 1: sign=1, val=0. Flip to 1. Lit becomes true.
                #   Case 2: sign=-1, val=1. Flip to 0. Lit becomes true.
                # So yes, flipping any variable in an unsat clause makes that literal true,
                # thus satisfying the clause.
                # So delta for c_idx is -1.
                
                total_delta = delta - 1
                
                if total_delta < best_score:
                    best_score = total_delta
                    best_var = v
                    candidates = [v]
                elif total_delta == best_score:
                    candidates.append(v)
            
            # Choose variable to flip
            if rng.random() < p_walk:
                # Random walk: pick random var from clause
                j_rand = rng.randint(3)
                v_flip = c_vars[j_rand]
            else:
                # Score-based: pick best var
                if candidates:
                    v_flip = candidates[rng.randint(len(candidates))]
                else:
                    # Fallback (should not happen)
                    v_flip = c_vars[rng.randint(3)]
            
            # Apply flip
            assignment[v_flip] = 1 - assignment[v_flip]
            
            # Update clause satisfaction for all clauses containing v_flip
            for k_idx in var_to_clauses[v_flip]:
                # Recalculate satisfaction for clause k_idx
                # Check if any literal is true
                sat = False
                for jj in range(3):
                    var_jj = clause_vars[k_idx, jj]
                    sign_jj = clause_signs[k_idx, jj]
                    val_jj = assignment[var_jj]
                    if (sign_jj == 1 and val_jj == 1) or (sign_jj == -1 and val_jj == 0):
                        sat = True
                        break
                
                if sat != clause_sat[k_idx]:
                    if sat:
                        unsat_count -= 1
                    else:
                        unsat_count += 1
                    clause_sat[k_idx] = sat
        
        # End of restart
        if unsat_count == 0:
            return assignment.tolist()
            
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
