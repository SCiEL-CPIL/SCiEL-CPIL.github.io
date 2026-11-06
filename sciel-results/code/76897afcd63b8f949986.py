import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin
    
    # Preprocessing: Pure literal elimination
    pos_counts = [0] * n
    neg_counts = [0] * n
    
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
    
    assignment = [None] * n
    
    # Pure literal elimination
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if assignment[v] is not None:
                continue
            if pos_counts[v] > 0 and neg_counts[v] == 0:
                assignment[v] = True
                changed = True
            elif neg_counts[v] > 0 and pos_counts[v] == 0:
                assignment[v] = False
                changed = True
    
    # Apply assignments, simplify
    new_clauses = []
    for c in clauses:
        new_c = []
        satisfied = False
        for lit in c:
            v = abs(lit) - 1
            if assignment[v] is not None:
                if assignment[v] == (lit > 0):
                    satisfied = True
                    break
                else:
                    continue
            else:
                new_c.append(lit)
        if not satisfied:
            if len(new_c) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            new_clauses.append(new_c)
    
    clauses = new_clauses
    
    if not clauses:
        final_assignment = [False] * n
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
            else:
                final_assignment[i] = random.random() < 0.5
        return final_assignment
    
    # Map to compact indices
    var_map = {}
    new_n = 0
    for v in range(n):
        if assignment[v] is None:
            var_map[v] = new_n
            new_n += 1
    
    remapped_clauses = []
    for c in clauses:
        new_c = []
        for lit in c:
            v = abs(lit) - 1
            new_v = var_map[v]
            new_lit = new_v + 1
            if lit < 0:
                new_lit = -new_lit
            new_c.append(new_lit)
        remapped_clauses.append(new_c)
    
    m = len(remapped_clauses)
    
    # Build NumPy arrays
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, c in enumerate(remapped_clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    # var_to_clauses: list of clause indices for each variable
    var_to_clauses = [[] for _ in range(new_n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
    
    # Convert to numpy arrays for vectorized gain calculation
    var_to_clauses_np = [np.array(vtc, dtype=np.int32) for vtc in var_to_clauses]
    
    rng = random.Random(42)
    
    # Initial random assignment
    curr = np.array([rng.random() < 0.5 for _ in range(new_n)], dtype=bool)
    
    def get_unsat_count(ass):
        true_val = ass[clause_vars]
        lit_true = np.where(clause_signs > 0, true_val, ~true_val)
        clause_sat = lit_true.any(axis=1)
        return np.sum(~clause_sat), ~clause_sat

    curr_unsat, unsat_mask = get_unsat_count(curr)
    best_unsat = curr_unsat
    best_assignment = curr.copy()
    
    if best_unsat == 0:
        final_assignment = [False] * n
        rev_map = {v: k for k, v in var_map.items()}
        for new_v, val in enumerate(best_assignment):
            orig_v = rev_map[new_v]
            final_assignment[orig_v] = bool(val)
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
        return final_assignment
    
    # WalkSAT with Luby restarts
    # Luby sequence generator
    def luby_sequence():
        seq = [1]
        while True:
            yield seq[0]
            new_seq = [1]
            for s in seq:
                new_seq.append(s * 2)
            seq = new_seq

    luby_gen = luby_sequence()
    walk_prob = 0.4
    
    while time.time() < deadline:
        steps = next(luby_gen)
        # Ensure we do at least some steps, but don't exceed deadline
        # If steps is huge, cap it to avoid long loops that ignore deadline
        if steps > 10000:
            steps = 10000
            
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            curr_unsat, unsat_mask = get_unsat_count(curr)
            
            if curr_unsat == 0:
                best_assignment = curr.copy()
                best_unsat = 0
                break
            
            if curr_unsat < best_unsat:
                best_unsat = curr_unsat
                best_assignment = curr.copy()
                
            if best_unsat == 0:
                break
            
            # Pick random unsat clause
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                break
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            
            # Pick variable from clause
            if rng.random() < walk_prob:
                j = rng.randrange(3)
            else:
                # Calculate gains for the 3 variables in the clause
                gains = np.zeros(3, dtype=np.int32)
                for j in range(3):
                    v = clause_vars[ci, j]
                    clauses_v = var_to_clauses_np[v]
                    if len(clauses_v) == 0:
                        continue
                    cv = clause_vars[clauses_v]
                    cs = clause_signs[clauses_v]
                    tv = curr[cv]
                    lt = np.where(cs > 0, tv, ~tv)
                    csat = lt.any(axis=1)
                    
                    # Gain = count(unsat) - count(sat where v is only true)
                    unsat_count = np.sum(~csat)
                    
                    # For sat clauses, check if v is the only true literal
                    # v is in position 'pos_v' in these clauses
                    # We need to check for each clause if lt[clause, pos_v] is true AND all others are false
                    # Since v is in these clauses, we can find its position
                    # pos_v = np.where(cv == v)[0] # This is (k, 3) boolean
                    # Actually, cv contains v at some position.
                    # Let's vectorize:
                    # For each clause k, let p be the index where cv[k, p] == v.
                    # Then v_lit = lt[k, p].
                    # other_lits = lt[k, :] with lt[k, p] removed.
                    # If v_lit is True and any(other_lits) is False -> v is only true.
                    
                    # Efficient way:
                    # Find positions of v in cv
                    pos_v = (cv == v) # (k, 3)
                    # v_lit_true = lt[np.arange(len(clauses_v)), pos_v.nonzero()[1]] # This is tricky if v appears multiple times? No, 3-SAT usually distinct vars in clause? 
                    # The problem statement says "signed 1-based literals". It doesn't explicitly say distinct, but standard 3-SAT assumes distinct.
                    # Assuming distinct:
                    v_idx = pos_v.nonzero()
                    if len(v_idx[0]) == 0:
                        gains[j] = 0
                        continue
                    # v_idx[0] are row indices, v_idx[1] are col indices
                    v_lit_vals = lt[v_idx[0], v_idx[1]]
                    
                    # For clauses where v is true, check if others are false
                    # We only care about clauses that are currently SAT.
                    # If a clause is SAT and v is the ONLY true literal, flipping v makes it UNSAT (loss -1).
                    # If a clause is UNSAT, flipping v (which is currently false) makes it SAT (gain +1).
                    
                    # Gain from unsat clauses:
                    gain_unsat = np.sum(~csat)
                    
                    # Loss from sat clauses:
                    # Mask for sat clauses: csat
                    # For these, check if v is the only true one.
                    sat_mask = csat
                    if np.any(sat_mask):
                        # Get v_lit for sat clauses
                        # We need to align v_idx with sat_mask
                        # v_idx covers all clauses in clauses_v.
                        # Create a boolean mask for v_idx rows that are sat
                        v_rows = v_idx[0]
                        v_cols = v_idx[1]
                        
                        # For each clause in clauses_v, if it is sat:
                        # Check if lt[row, col] is True (v is true) AND sum(lt[row, :]) == 1 (only v is true)
                        # Note: if v is false, it can't be the only true one.
                        
                        v_true_in_sat = v_lit_vals[sat_mask[v_rows]]
                        # Sum of literals in those clauses
                        sat_clause_sums = np.sum(lt[sat_mask, :], axis=1)
                        
                        # v is only true if v_lit is True AND sum is 1
                        only_true = (v_true_in_sat) & (sat_clause_sums == 1)
                        loss_sat = np.sum(only_true)
                    else:
                        loss_sat = 0
                        
                    gains[j] = gain_unsat - loss_sat

                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                j = rng.choice(candidates)
            
            v = clause_vars[ci, j]
            curr[v] = not curr[v]
        
        if best_unsat == 0:
            break
        
        # Adaptive walk prob
        if best_unsat > 10:
            walk_prob = min(0.8, walk_prob * 1.05)
        else:
            walk_prob = max(0.1, walk_prob * 0.95)
    
    # Map back
    final_assignment = [False] * n
    rev_map = {v: k for k, v in var_map.items()}
    for new_v, val in enumerate(best_assignment):
        orig_v = rev_map[new_v]
        final_assignment[orig_v] = bool(val)
    for i in range(n):
        if assignment[i] is not None:
            final_assignment[i] = assignment[i]
    
    return final_assignment
