"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0
    
    m = len(clauses)
    if m == 0:
        return [False] * n
    
    # Build data structures
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
    
    max_w = max(len(c) for c in clauses) if m > 0 else 0
    clause_mat = np.full((m, max_w), -1, dtype=np.int32)
    clause_sig = np.zeros((m, max_w), dtype=np.int8)
    for ci, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_mat[ci, j] = abs(lit) - 1
            clause_sig[ci, j] = 1 if lit > 0 else -1
    
    # For each variable, list of (clause_idx, sign_in_clause)
    var_clause_info = []
    for v in range(n):
        info = []
        for ci in var_clauses[v]:
            for j in range(max_w):
                if clause_mat[ci, j] == v:
                    info.append((ci, clause_sig[ci, j]))
                    break
        var_clause_info.append(info)
    
    def check_satisfied(assignment):
        """Check if assignment satisfies all clauses using numpy."""
        a = np.array(assignment, dtype=bool)
        valid = clause_mat >= 0
        safe_mat = np.where(valid, clause_mat, 0)
        gathered = a[safe_mat]
        sig_pos = clause_sig == 1
        sig_neg = clause_sig == -1
        sat_elem = (sig_pos & gathered) | (sig_neg & ~gathered)
        sat_elem = sat_elem & valid
        clause_sat = sat_elem.any(axis=1)
        return int(np.sum(~clause_sat)) == 0
    
    def count_unsat_vec(assignment):
        """Count unsatisfied clauses using numpy."""
        a = np.array(assignment, dtype=bool)
        valid = clause_mat >= 0
        safe_mat = np.where(valid, clause_mat, 0)
        gathered = a[safe_mat]
        sig_pos = clause_sig == 1
        sig_neg = clause_sig == -1
        sat_elem = (sig_pos & gathered) | (sig_neg & ~gathered)
        sat_elem = sat_elem & valid
        clause_sat = sat_elem.any(axis=1)
        return int(np.sum(~clause_sat))
    
    # DPLL solver with unit propagation and pure literal elimination
    def dpll_solve(assignment, remaining_clauses):
        """
        DPLL solver.
        assignment: list of (None, True, False) for each variable
        remaining_clauses: list of clause indices still to be satisfied
        Returns: True if satisfiable, and updates assignment if so
        """
        if time.time() >= deadline:
            return False
        
        # Check if all clauses are satisfied
        if not remaining_clauses:
            return True
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_remaining = []
            for ci in remaining_clauses:
                clause = clauses[ci]
                # Determine which literals are still undecided
                undecided = []
                satisfied = False
                for lit in clause:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if val is None:
                        undecided.append(lit)
                    else:
                        expected = (lit > 0)
                        if val == expected:
                            satisfied = True
                            break
                if satisfied:
                    continue
                if len(undecided) == 0:
                    # Clause is falsified
                    return False
                if len(undecided) == 1:
                    # Unit clause: must assign this literal
                    lit = undecided[0]
                    v = abs(lit) - 1
                    assignment[v] = (lit > 0)
                    changed = True
                    # Don't add to new_remaining, it's now satisfied
                else:
                    new_remaining.append(ci)
            remaining_clauses = new_remaining
        
        if not remaining_clauses:
            return True
        
        if time.time() >= deadline:
            return False
        
        # Pure literal elimination
        changed = True
        while changed:
            changed = False
            # For each unassigned variable, check if it appears with only one sign
            var_signs = {}
            for ci in remaining_clauses:
                clause = clauses[ci]
                for lit in clause:
                    v = abs(lit) - 1
                    if assignment[v] is None:
                        s = 1 if lit > 0 else -1
                        if v not in var_signs:
                            var_signs[v] = set()
                        var_signs[v].add(s)
            
            # Assign pure literals
            for v, signs in var_signs.items():
                if len(signs) == 1:
                    assignment[v] = (list(signs)[0] == 1)
                    changed = True
            
            if changed:
                # Re-filter clauses
                new_remaining = []
                for ci in remaining_clauses:
                    clause = clauses[ci]
                    satisfied = False
                    for lit in clause:
                        v = abs(lit) - 1
                        val = assignment[v]
                        if val is not None:
                            expected = (lit > 0)
                            if val == expected:
                                satisfied = True
                                break
                    if not satisfied:
                        new_remaining.append(ci)
                remaining_clauses = new_remaining
        
        if not remaining_clauses:
            return True
        
        if time.time() >= deadline:
            return False
        
        # Choose a variable to branch on
        # Use degree heuristic: pick variable appearing in most remaining clauses
        var_count = {}
        for ci in remaining_clauses:
            for lit in clauses[ci]:
                v = abs(lit) - 1
                if assignment[v] is None:
                    var_count[v] = var_count.get(v, 0) + 1
        
        if not var_count:
            return True
        
        # Pick variable with highest degree
        best_var = max(var_count, key=lambda v: var_count[v])
        
        # Try True first
        assignment[best_var] = True
        # Filter clauses
        sat_clauses = set()
        new_remaining = []
        for ci in remaining_clauses:
            clause = clauses[ci]
            satisfied = False
            for lit in clause:
                v = abs(lit) - 1
                val = assignment[v]
                if val is not None:
                    expected = (lit > 0)
                    if val == expected:
                        satisfied = True
                        break
            if satisfied:
                sat_clauses.add(ci)
            else:
                new_remaining.append(ci)
        
        if dpll_solve(assignment, new_remaining):
            return True
        
        # Backtrack
        assignment[best_var] = False
        new_remaining = []
        for ci in remaining_clauses:
            clause = clauses[ci]
            satisfied = False
            for lit in clause:
                v = abs(lit) - 1
                val = assignment[v]
                if val is not None:
                    expected = (lit > 0)
                    if val == expected:
                        satisfied = True
                        break
            if satisfied:
                pass
            else:
                new_remaining.append(ci)
        
        if dpll_solve(assignment, new_remaining):
            return True
        
        # Backtrack
        assignment[best_var] = None
        return False
    
    # Try DPLL first
    dpll_assignment = [None] * n
    all_clause_indices = list(range(m))
    if dpll_solve(dpll_assignment, all_clause_indices):
        # Convert None to False (shouldn't happen if all clauses satisfied)
        result = [bool(v) if v is not None else False for v in dpll_assignment]
        if check_satisfied(result):
            return result
    
    # Fallback to WalkSAT
    best_assignment = None
    best_unsat_count = float('inf')
    
    def compute_clause_sat_vec(a):
        """Vectorized clause satisfaction check."""
        if m == 0:
            return np.array([], dtype=bool), 0
        valid = clause_mat >= 0
        safe_mat = np.where(valid, clause_mat, 0)
        gathered = a[safe_mat]
        sig_pos = clause_sig == 1
        sig_neg = clause_sig == -1
        sat_elem = (sig_pos & gathered) | (sig_neg & ~gathered)
        sat_elem = sat_elem & valid
        clause_sat = sat_elem.any(axis=1)
        num_unsat = int(np.sum(~clause_sat))
        return clause_sat, num_unsat
    
    max_restarts = 3000
    base_steps = 600
    
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
        
        steps = base_steps + int(150 * (1.0 - min(num_unsat / max(m, 1), 1.0)))
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            if num_unsat == 0:
                return assignment
            
            # Pick random unsatisfied clause
            if num_unsat < m:
                unsat_mask = ~clause_sat
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
                
                gain = 0
                for (c_idx, s_in_c) in var_clause_info[v]:
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
                
                noise = rng.random() * 0.5
                score = -gain + noise
                
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var < 0:
                v = rng.randrange(n)
            else:
                v = best_var
            
            a[v] = not a[v]
            assignment[v] = not assignment[v]
            
            for (c_idx, s_in_c) in var_clause_info[v]:
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
