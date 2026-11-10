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

    # Build M_pos and M_neg: (n, m)
    # M_pos[v, c] = 1 if clause c contains +v
    # M_neg[v, c] = 1 if clause c contains -v
    M_pos = np.zeros((n, m), dtype=np.float32)
    M_neg = np.zeros((n, m), dtype=np.float32)
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                M_pos[v, i] = 1
            else:
                M_neg[v, i] = 1

    rng = np.random.RandomState(42)
    assignment = rng.rand(n) < 0.5

    # Compute initial clause_sat
    # lit_sat[i, j] = (assignment[var] == (sign==1))
    a = assignment[clause_vars] # (m, 3)
    s = (clause_signs == 1) # (m, 3)
    lit_sat = (a == s)
    clause_sat = lit_sat.any(axis=1)

    if clause_sat.all():
        return assignment.tolist()

    max_iters = 50000
    p_walk = 0.15 # Probability to do a random walk step when stuck or always?
    
    for _ in range(max_iters):
        if clause_sat.all():
            break
            
        unsat_mask = ~clause_sat
        num_unsat = np.sum(unsat_mask)
        
        if num_unsat == 0:
            break

        # Compute gains
        # gain_true[v] = unsat clauses with +v - sat clauses with -v
        # gain_false[v] = unsat clauses with -v - sat clauses with +v
        
        # unsat_counts_pos[v] = sum over c of M_pos[v,c] * unsat_mask[c]
        unsat_counts_pos = M_pos @ unsat_mask
        unsat_counts_neg = M_neg @ unsat_mask
        
        sat_counts_pos = M_pos @ clause_sat
        sat_counts_neg = M_neg @ clause_sat
        
        gain_true = unsat_counts_pos - sat_counts_neg
        gain_false = unsat_counts_neg - sat_counts_pos
        
        # We want to choose v and val to maximize gain.
        # Create a combined gain array where we can pick the best (v, val) pair.
        # Flatten gains? Or just compare max(gain_true) and max(gain_false).
        
        best_gain = -np.inf
        best_var = -1
        best_val = False
        
        max_gain_true = np.max(gain_true)
        max_gain_false = np.max(gain_false)
        
        if max_gain_true > max_gain_false:
            best_gain = max_gain_true
            # Find all vars with this gain, pick one randomly
            candidates = np.where(gain_true == max_gain_true)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = True
        elif max_gain_false > max_gain_true:
            best_gain = max_gain_false
            candidates = np.where(gain_false == max_gain_false)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = False
        else:
            # Tie or both low. Random tie break between the two sets or just pick one
            # If both are -inf or equal, pick randomly
            if max_gain_true == max_gain_false:
                if rng.rand() < 0.5:
                    candidates = np.where(gain_true == max_gain_true)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = True
                    best_gain = max_gain_true
                else:
                    candidates = np.where(gain_false == max_gain_false)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = False
                    best_gain = max_gain_false
            else:
                # Should not happen if logic is correct, but fallback
                best_var = int(rng.randint(0, n))
                best_val = not assignment[best_var]
                best_gain = 0

        # Escape mechanism: If best_gain <= 0, we might be stuck.
        # With probability p_walk, do a standard WalkSAT step instead.
        # Or, if best_gain is very low, increase noise.
        # Let's use: if best_gain <= 0, with prob 0.5, do random walk step.
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
                # Set to value that satisfies this literal
                best_val = (clause_signs[ci, j] == 1)
            else:
                best_val = not assignment[v]

        # Apply flip
        if best_val != assignment[best_var]:
            assignment[best_var] = best_val
            
            # Update clause_sat for clauses containing best_var
            # We need to recompute sat status for clauses involving best_var.
            # Find clauses containing best_var
            # We can use M_pos and M_neg to find them?
            # M_pos[best_var, :] == 1 or M_neg[best_var, :] == 1
            affected_clauses = np.where((M_pos[best_var, :] == 1) | (M_neg[best_var, :] == 1))[0]
            
            for ci in affected_clauses:
                # Recompute satisfaction
                is_sat = False
                for j in range(3):
                    v = int(clause_vars[ci, j])
                    s = (clause_signs[ci, j] == 1)
                    if assignment[v] == s:
                        is_sat = True
                        break
                clause_sat[ci] = is_sat

    return assignment.tolist()
