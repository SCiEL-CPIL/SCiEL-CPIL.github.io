import numpy as np
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    rng = random.Random(42)
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)
    
    # Preprocess
    indices = np.zeros((m, 3), dtype=np.int32)
    signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            indices[i, j] = var_idx
            signs[i, j] = sign
            
    # Precompute clause lists for each variable
    # pos_clauses[v]: list of (clause_idx, lit_idx) where v is positive
    # neg_clauses[v]: list of (clause_idx, lit_idx) where v is negative
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            if signs[i, j] == 1:
                pos_clauses[v].append((i, j))
            else:
                neg_clauses[v].append((i, j))
                
    def compute_satisfaction(assignment):
        lit_true = (assignment[indices] == (signs > 0))
        return lit_true.any(axis=1)
        
    is_sat = compute_satisfaction(assignment)
    
    start_time = time.time()
    deadline = 325.0
    
    for step in range(200000):
        if time.time() > deadline:
            break
            
        if not is_sat.any():
            break
            
        unsat_indices = np.where(~is_sat)[0]
        c_idx = rng.choice(unsat_indices)
        
        # Candidates: the 3 literals in clause c_idx
        min_breaks = float('inf')
        best_var = -1
        candidates = []
        
        for j in range(3):
            v = indices[c_idx, j]
            current_val = assignment[v]
            
            breaks = 0
            
            # Check clauses where v appears
            # If current_val is 0, v is False, -v is True.
            # Clauses with -v (neg_clauses[v]) might break.
            # If current_val is 1, v is True, -v is False.
            # Clauses with v (pos_clauses[v]) might break.
            
            if current_val == 0:
                # Check neg_clauses[v]
                for c, k in neg_clauses[v]:
                    if not is_sat[c]:
                        continue
                    # Clause c is satisfied. Check if -v was the only true literal.
                    # Check other 2 literals in c
                    is_only = True
                    for jj in range(3):
                        if jj == k:
                            continue
                        # Check if literal jj in clause c is true
                        # lit_val = (assignment[indices[c, jj]] == (signs[c, jj] > 0))
                        if assignment[indices[c, jj]] == (signs[c, jj] > 0):
                            is_only = False
                            break
                    if is_only:
                        breaks += 1
            else:
                # Check pos_clauses[v]
                for c, k in pos_clauses[v]:
                    if not is_sat[c]:
                        continue
                    is_only = True
                    for jj in range(3):
                        if jj == k:
                            continue
                        if assignment[indices[c, jj]] == (signs[c, jj] > 0):
                            is_only = False
                            break
                    if is_only:
                        breaks += 1
                        
            if breaks < min_breaks:
                min_breaks = breaks
                best_var = v
                candidates = [v]
            elif breaks == min_breaks:
                candidates.append(v)
                
        if best_var == -1:
            continue
            
        v_to_flip = rng.choice(candidates)
        assignment[v_to_flip] = 1 - assignment[v_to_flip]
        
        # Update is_sat
        is_sat = compute_satisfaction(assignment)
        
    return assignment.tolist()
