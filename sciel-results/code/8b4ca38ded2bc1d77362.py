"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    
    # Preprocessing: build data structures
    m = len(clauses)
    clause_vars = []
    clause_signs = []
    var_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        vars_list = []
        signs_list = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_list.append(v)
            signs_list.append(s)
            var_clauses[v].append(ci)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    # Precompute for each clause: which variables and signs
    # For fast satisfaction: clause is sat if any (v,s) has assignment[v]==(s==1)
    # Encode: for clause ci with literals (v1,s1),(v2,s2),(v3,s3)
    # sat iff (a[v1]==(s1==1)) or (a[v2]==(s2==1)) or (a[v3]==(s3==1))
    
    # Build numpy arrays for vectorized clause evaluation
    # clause_mat: (m, 3) int array of variable indices (0-based), pad with -1 if <3
    # clause_sig: (m, 3) int array of signs
    max_w = max(len(c) for c in clauses) if m > 0 else 0
    clause_mat = np.full((m, max_w), -1, dtype=np.int32)
    clause_sig = np.zeros((m, max_w), dtype=np.int8)
    for ci, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_mat[ci, j] = abs(lit) - 1
            clause_sig[ci, j] = 1 if lit > 0 else -1
    
    # For gain computation: for each variable v, precompute list of (clause_idx, sign_in_clause)
    var_clause_info = []
    for v in range(n):
        info = []
        for ci in var_clauses[v]:
            # Find sign of v in clause ci
            for j in range(max_w):
                if clause_mat[ci, j] == v:
                    info.append((ci, clause_sig[ci, j]))
                    break
        var_clause_info.append(info)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def compute_clause_sat_vec(a):
        """Vectorized clause satisfaction check. a is bool array of length n."""
        if m == 0:
            return np.array([], dtype=bool), 0
        # For each clause, check if any literal is satisfied
        # A literal (v, s) is satisfied if a[v] == (s == 1)
        # i.e., if s==1: a[v] is True; if s==-1: a[v] is False
        # Compute: for each (ci, j), satisfied = (clause_sig[ci,j]==1) == a[clause_mat[ci,j]]
        valid = clause_mat >= 0
        # Gather assignment values
        safe_mat = np.where(valid, clause_mat, 0)
        gathered = a[safe_mat]  # (m, max_w) bool array
        # Satisfied: (sig==1 and gathered==True) or (sig==-1 and gathered==False)
        sig_pos = clause_sig == 1
        sig_neg = clause_sig == -1
        sat_elem = (sig_pos & gathered) | (sig_neg & ~gathered)
        sat_elem = sat_elem & valid  # mask out padding
        clause_sat = sat_elem.any(axis=1)
        num_unsat = int(np.sum(~clause_sat))
        return clause_sat, num_unsat
    
    # Main search loop with adaptive WalkSAT
    max_restarts = 5000
    base_steps = 800
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        a = np.array(assignment, dtype=bool)
        clause_sat, num_unsat = compute_clause_sat_vec(a)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment[:]
        
        # Adaptive steps: more steps when close to solution
        steps = base_steps + int(200 * (1.0 - min(num_unsat / max(m, 1), 1.0)))
        
        # WalkSAT loop
        for step in range(steps):
            if time.time() >= deadline:
                break
                
            if num_unsat == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            if num_unsat < m:
                unsat_mask = ~clause_sat
                # Find random unsat clause
                unsat_indices = np.nonzero(unsat_mask)[0]
                ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            else:
                ci = rng.randrange(m)
            
            # Compute gain for each literal in this clause
            best_var = -1
            best_score = float('inf')
            
            for j in range(max_w):
                v = clause_mat[ci, j]
                if v < 0:
                    break
                s = clause_sig[ci, j]
                
                # Calculate gain if we flip v
                gain = 0
                for (c_idx, s_in_c) in var_clause_info[v]:
                    # Check if clause c_idx is satisfied by literals other than v
                    satisfied_by_others = False
                    for j2 in range(max_w):
                        v2 = clause_mat[c_idx, j2]
                        if v2 == v or v2 < 0:
                            continue
                        s2 = clause_sig[c_idx, j2]
                        if s2 == 1:
                            if a[v2]:
                                satisfied_by_others = True
                                break
                        else:
                            if not a[v2]:
                                satisfied_by_others = True
                                break
                    
                    if satisfied_by_others:
                        continue
                    else:
                        if clause_sat[c_idx]:
                            gain -= 1
                        else:
                            gain += 1
                
                # Adaptive noise: more noise when stuck
                noise = rng.random() * 0.5
                score = -gain + noise
                
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var < 0:
                v = rng.randrange(n)
            else:
                v = best_var
                
            # Flip v
            a[v] = not a[v]
            assignment[v] = not assignment[v]
            
            # Update clause_sat for clauses containing v
            for (c_idx, s_in_c) in var_clause_info[v]:
                # Recompute satisfaction of clause c_idx
                new_sat = False
                for j2 in range(max_w):
                    v2 = clause_mat[c_idx, j2]
                    if v2 < 0:
                        break
                    s2 = clause_sig[c_idx, j2]
                    if s2 == 1:
                        if a[v2]:
                            new_sat = True
                            break
                    else:
                        if not a[v2]:
                            new_sat = True
                            break
                
                old_sat = clause_sat[c_idx]
                
                if old_sat and not new_sat:
                    num_unsat += 1
                elif not old_sat and new_sat:
                    num_unsat -= 1
                
                clause_sat[c_idx] = new_sat
            
            if num_unsat == 0:
                return assignment
                
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment[:]
                
        if time.time() >= deadline:
            break

    if best_assignment is not None:
        return best_assignment
        
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
