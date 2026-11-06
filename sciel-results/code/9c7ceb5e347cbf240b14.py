import numpy as np
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [True] * n

    # Precompute clause structure
    # indices: (m, 3), signs: (m, 3)
    indices = np.zeros((m, 3), dtype=np.int32)
    signs = np.zeros((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            indices[i, j] = abs(l) - 1
            signs[i, j] = 1 if l > 0 else -1

    deadline = time.time() + 290.0
    rng = np.random.default_rng(0)
    
    # Min-Conflicts with Random Restarts
    # We will use a hybrid: mostly Min-Conflicts, some WalkSAT noise.
    
    best_unsat_count = m + 1
    best_assignment = None

    while time.time() < deadline:
        # Random restart
        assignment = rng.integers(0, 2, size=n, dtype=np.int8) # 0=False, 1=True
        
        # Initial check
        # lit_values: (m, 3)
        # assignment[indices] gives values of variables
        # A literal is true if (value == 1 and sign == 1) or (value == 0 and sign == -1)
        # This is equivalent to: (value * sign) == 1 ? No.
        # If sign is 1, we want value 1. If sign is -1, we want value 0.
        # So, satisfied if (value == 1) == (sign == 1)
        # Or: (value * sign) == 1 is NOT correct for 0/-1.
        # Let's use: 
        # var_vals = assignment[indices] (0 or 1)
        # lit_true = (var_vals == 1) if sign==1 else (var_vals == 0)
        # Easier: 
        # lit_true = (var_vals * sign) + (1 - sign) ? No.
        # If sign=1: lit_true = var_vals
        # If sign=-1: lit_true = 1 - var_vals
        # So lit_true = (sign * var_vals + (1 - sign)) / 2 ? 
        # Let's just use boolean logic.
        
        # Vectorized satisfaction check:
        # var_vals: (m, 3)
        # signs: (m, 3)
        # lit_satisfied = (var_vals == 1) & (signs == 1) | (var_vals == 0) & (signs == -1)
        # This can be simplified: 
        # If we map 0->0, 1->1.
        # lit_satisfied = (var_vals * signs) + (1 - signs) * (1 - var_vals) ?
        # Let's stick to:
        # satisfied_lit = (var_vals == 1) * (signs == 1) + (var_vals == 0) * (signs == -1)
        # This is (m,3) boolean array.
        
        unsat_count = m
        prev_unsat = m + 1
        
        # Local search loop
        for step in range(10000):
            if time.time() > deadline:
                break
                
            var_vals = assignment[indices]
            # lit_satisfied: True if the literal is true
            # sign 1 -> var 1. sign -1 -> var 0.
            # lit_satisfied = (var_vals == signs) if we mapped signs to 0/1?
            # Let's map signs to 0/1: 1->1, -1->0.
            # Then lit_satisfied = (var_vals == signs_mapped)
            
            signs_mapped = (signs + 1) // 2 # 1->1, -1->0
            lit_satisfied = (var_vals == signs_mapped)
            
            clause_satisfied = np.any(lit_satisfied, axis=1)
            unsat_clauses = np.where(~clause_satisfied)[0]
            unsat_count = len(unsat_clauses)
            
            if unsat_count == 0:
                return [bool(x) for x in assignment]
                
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
                
            # If we haven't improved in a long time, restart
            if unsat_count >= prev_unsat:
                # Simple stagnation check
                pass # We handle restarts via outer loop or max steps
            
            prev_unsat = unsat_count

            # Choose variable to flip
            if rng.random() < 0.1:
                # WalkSAT: pick random var from random unsat clause
                random_clause_idx = unsat_clauses[rng.integers(0, len(unsat_clauses))]
                random_var_idx = indices[random_clause_idx, rng.integers(0, 3)]
                var_to_flip = random_var_idx
            else:
                # Min-Conflicts: pick var appearing in most unsat clauses
                # Get all variables in unsat clauses
                unsat_vars = indices[unsat_clauses].ravel()
                # Count occurrences
                counts = np.bincount(unsat_vars, minlength=n)
                var_to_flip = np.argmax(counts)
                
            assignment[var_to_flip] = 1 - assignment[var_to_flip]
            
            # Optional: Early break if we've been stuck
            # But the restart logic in the outer loop handles it.
            # We can add a step limit per restart.
            
        # End of restart loop
        # If we exhausted steps without finding solution, restart again
        if best_assignment is not None and best_unsat_count < m:
             # Keep the best found so far? 
             # Usually restarts are better than keeping a bad state.
             pass

    # Return best assignment found if not solved
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    
    return [False] * n
