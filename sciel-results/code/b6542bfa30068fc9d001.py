import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute clause literal arrays
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8) # 1 for pos, -1 for neg
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1

    # Build incidence matrices
    # M_pos[v, c] = 1 if clause c contains +v
    # M_neg[v, c] = 1 if clause c contains -v
    M_pos = np.zeros((n, m), dtype=np.int32)
    M_neg = np.zeros((n, m), dtype=np.int32)
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                M_pos[v, i] = 1
            else:
                M_neg[v, i] = 1

    # Also store list of clauses per variable for efficient updates
    # var_clauses[v] = list of clause indices containing variable v
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            if i not in var_clauses[v]:
                var_clauses[v].append(i)
    
    # Convert to numpy arrays for faster access if needed, but lists are fine for small n
    # Actually, let's keep it simple with lists.

    rng = np.random.RandomState(42)
    
    # Try multiple random restarts
    max_restarts = 10
    max_iters_per_restart = 20000
    
    for restart in range(max_restarts):
        # Random initial assignment
        assignment = rng.rand(n) < 0.5

        # Compute initial clause_sat
        # lit_sat[i, j] = (assignment[var] == (sign==1))
        a = assignment[clause_vars] # (m, 3)
        s = (clause_signs == 1) # (m, 3)
        lit_sat = (a == s)
        clause_sat = lit_sat.any(axis=1)
        
        # Count satisfied literals per clause (for efficient updates)
        # sat_count[c] = number of satisfied literals in clause c
        sat_count = lit_sat.sum(axis=1).astype(np.int32)

        if clause_sat.all():
            return assignment.tolist()

        num_unsat = m - np.sum(clause_sat)
        
        for _ in range(max_iters_per_restart):
            if num_unsat == 0:
                return assignment.tolist()

            # Compute gains for each variable and value
            # gain_true[v] = (unsat clauses with +v) - (sat clauses with -v)
            # gain_false[v] = (unsat clauses with -v) - (sat clauses with +v)
            
            unsat_mask = ~clause_sat
            
            # Use matrix multiplication for speed
            unsat_counts_pos = M_pos @ unsat_mask
            unsat_counts_neg = M_neg @ unsat_mask
            
            sat_counts_pos = M_pos @ clause_sat
            sat_counts_neg = M_neg @ clause_sat
            
            gain_true = unsat_counts_pos - sat_counts_neg
            gain_false = unsat_counts_neg - sat_counts_pos
            
            # Find best flip
            max_gain_true = np.max(gain_true)
            max_gain_false = np.max(gain_false)
            
            if max_gain_true > max_gain_false:
                best_gain = max_gain_true
                candidates = np.where(gain_true == max_gain_true)[0]
                best_var = candidates[rng.randint(len(candidates))]
                best_val = True
            elif max_gain_false > max_gain_true:
                best_gain = max_gain_false
                candidates = np.where(gain_false == max_gain_false)[0]
                best_var = candidates[rng.randint(len(candidates))]
                best_val = False
            else:
                # Tie: pick randomly between the two sets
                if rng.rand() < 0.5:
                    best_gain = max_gain_true
                    candidates = np.where(gain_true == max_gain_true)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = True
                else:
                    best_gain = max_gain_false
                    candidates = np.where(gain_false == max_gain_false)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = False

            # Escape mechanism: if best_gain <= 0, use WalkSAT-style step
            if best_gain <= 0 and rng.rand() < 0.5:
                # Pick a random unsat clause
                unsat_indices = np.where(unsat_mask)[0]
                if len(unsat_indices) == 0:
                    break
                ci = unsat_indices[rng.randint(len(unsat_indices))]
                # Pick a random variable in this clause
                j = rng.randint(3)
                v = int(clause_vars[ci, j])
                best_var = v
                # With prob 0.5, set to value that satisfies clause, else random
                if rng.rand() < 0.5:
                    best_val = (clause_signs[ci, j] == 1)
                else:
                    best_val = not assignment[v]

            # Apply flip
            if best_val != assignment[best_var]:
                assignment[best_var] = best_val
                
                # Update clause_sat and sat_count for affected clauses
                for ci in var_clauses[best_var]:
                    old_sat = clause_sat[ci]
                    
                    # Recompute satisfaction for this clause
                    new_sat_count = 0
                    for j in range(3):
                        v = int(clause_vars[ci, j])
                        s = (clause_signs[ci, j] == 1)
                        if assignment[v] == s:
                            new_sat_count += 1
                    
                    new_sat = new_sat_count > 0
                    sat_count[ci] = new_sat_count
                    
                    if new_sat != old_sat:
                        clause_sat[ci] = new_sat
                        if new_sat:
                            num_unsat -= 1
                        else:
                            num_unsat += 1

    return assignment.tolist()
