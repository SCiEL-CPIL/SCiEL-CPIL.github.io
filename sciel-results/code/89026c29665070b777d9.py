# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: 290s safety margin
    deadline = time.time() + 290.0

    # Precompute clause literal arrays
    # lits: shape (m, 3), signed 1-based literals
    lits = np.array(clauses, dtype=np.int32)
    # var_idx: 0-based variable indices
    var_idx = np.abs(lits) - 1
    # sign: +1 if literal is positive, -1 if negative
    sign = np.where(lits > 0, 1, -1)

    # Precompute variable to clauses mapping for fast updates
    # var_to_clauses[v] = list of (clause_idx, literal_sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for c in range(m):
        for j in range(3):
            v = var_idx[c, j]
            s = sign[c, j]
            var_to_clauses[v].append((c, s))

    rng = np.random.RandomState(42)
    
    # Helper to check if a literal is satisfied given assignment
    # assignment is 0/1 array
    # literal (v, s) is satisfied if (assignment[v] == 1 and s == 1) or (assignment[v] == 0 and s == -1)
    # Equivalent: assignment[v] == (s == 1)
    
    def is_lit_satisfied(assignment, v, s):
        return assignment[v] == (1 if s == 1 else 0)

    best_assignment = None
    best_unsat = m + 1

    # Try a few restarts
    for restart in range(50):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = (rng.rand(n) < 0.5).astype(np.int32)
        
        # Initialize clause_sat_count and unsat_clauses
        clause_sat_count = np.zeros(m, dtype=np.int32)
        unsat_clauses = set()
        
        for c in range(m):
            sat_count = 0
            for j in range(3):
                v = var_idx[c, j]
                s = sign[c, j]
                if is_lit_satisfied(assignment, v, s):
                    sat_count += 1
            clause_sat_count[c] = sat_count
            if sat_count == 0:
                unsat_clauses.add(c)
        
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment.tolist()
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()

        # Local Search (WalkSAT variant)
        # Parameters
        noise_prob = 0.05 # Probability of random flip
        max_steps = 100000 # Max steps per restart
        
        for step in range(max_steps):
            if time.time() > deadline:
                return best_assignment.tolist() if best_assignment is not None else [False] * n
            
            if not unsat_clauses:
                return assignment.tolist()
            
            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.copy()
                
            # Pick a random unsatisfied clause
            # Using list(unsat_clauses) is O(U). We can keep a list for faster random access if needed, 
            # but for small U it's fine. To be safe, convert to list occasionally or just use next(iter()).
            # Actually, picking uniformly from a set is not O(1). 
            # Let's maintain a list of unsat clauses for O(1) random access.
            
            # Optimization: Rebuild list of unsat clauses if it changed significantly? 
            # Or just use random.choice on list(unsat_clauses). 
            # Given n=150, m~500, U is small. list(unsat_clauses) is fast.
            unsat_list = list(unsat_clauses)
            c = unsat_list[rng.randint(len(unsat_list))]
            
            # Get variables in this clause
            c_vars = var_idx[c]
            c_signs = sign[c]
            
            # Determine which literals are false
            false_vars = []
            for j in range(3):
                v = c_vars[j]
                s = c_signs[j]
                if not is_lit_satisfied(assignment, v, s):
                    false_vars.append(v)
            
            # Pick variable to flip
            if not false_vars:
                # Should not happen for unsat clause, but safety
                v_to_flip = c_vars[rng.randint(3)]
            else:
                if rng.rand() < noise_prob:
                    v_to_flip = c_vars[rng.randint(3)]
                else:
                    v_to_flip = false_vars[rng.randint(len(false_vars))]
            
            # Flip variable
            old_val = assignment[v_to_flip]
            new_val = 1 - old_val
            assignment[v_to_flip] = new_val
            
            # Update clause statuses
            # For each clause containing v_to_flip
            for c_idx, s in var_to_clauses[v_to_flip]:
                # Check old status of this literal
                was_sat = is_lit_satisfied(assignment, v_to_flip, s) # This checks NEW value, we need OLD
                # Wait, assignment is already updated. 
                # We need to know if the literal was satisfied BEFORE flip.
                # Old value was old_val.
                # Literal (v, s) sat iff old_val == (1 if s==1 else 0)
                was_sat = (old_val == (1 if s == 1 else 0))
                # New value is new_val.
                # Literal (v, s) sat iff new_val == (1 if s==1 else 0)
                is_sat = (new_val == (1 if s == 1 else 0))
                
                if was_sat and not is_sat:
                    # Lost satisfaction
                    clause_sat_count[c_idx] -= 1
                    if clause_sat_count[c_idx] == 0:
                        unsat_clauses.add(c_idx)
                elif not was_sat and is_sat:
                    # Gained satisfaction
                    clause_sat_count[c_idx] += 1
                    if clause_sat_count[c_idx] > 0 and c_idx in unsat_clauses:
                        unsat_clauses.remove(c_idx)
                # If both sat or both unsat, count doesn't change, and unsat status doesn't change (since it was 0 or >0)
                # Actually, if it was unsat (0) and stays unsat (0), no change.
                # If it was sat (>0) and stays sat (>0), no change.
                # The logic above handles the transitions correctly.

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
