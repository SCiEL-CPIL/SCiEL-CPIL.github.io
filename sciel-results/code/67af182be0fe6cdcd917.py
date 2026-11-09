import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    # Deadline: 325 seconds to be safe against 330s kill
    start_time = time.time()
    deadline = start_time + 325.0
    
    m = len(clauses)
    
    # Pre-process clauses
    # clause_vars[i] = [var_idx_1, var_idx_2, var_idx_3]
    # clause_pols[i] = [pol_1, pol_2, pol_3]
    # var_to_clauses[v] = list of (clause_idx, pos_in_clause)
    
    clause_vars = []
    clause_pols = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        vars_in_c = []
        pols_in_c = []
        for j, l in enumerate(c):
            var_idx = abs(l) - 1
            pol = 1 if l > 0 else -1
            vars_in_c.append(var_idx)
            pols_in_c.append(pol)
            var_to_clauses[var_idx].append((i, j))
        clause_vars.append(vars_in_c)
        clause_pols.append(pols_in_c)
        
    # --- DPLL Solver with Unit Propagation ---
    
    def dpll(clauses, assignment, depth=0):
        """
        Solves the sub-problem defined by 'clauses' given current 'assignment'.
        assignment: list of 0, 1, or None (unassigned)
        Returns: assignment if satisfied, None if unsat.
        """
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            for i in range(len(clauses)):
                clause = clauses[i]
                if not clause:
                    return None # Empty clause -> UNSAT
                
                # Check for unit clause
                unassigned_count = 0
                unit_var = -1
                unit_pol = 0
                is_satisfied = False
                
                for v, p in clause:
                    if assignment[v] is None:
                        unassigned_count += 1
                        unit_var = v
                        unit_pol = p
                    else:
                        if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                            is_satisfied = True
                            break
                
                if is_satisfied:
                    continue # Clause is satisfied, ignore it for now (or mark as done)
                    
                if unassigned_count == 0:
                    return None # Clause not satisfied and no vars left -> UNSAT
                
                if unassigned_count == 1:
                    # Unit clause: force assignment
                    assignment[unit_var] = 1 if unit_pol == 1 else 0
                    changed = True
                    # We could also simplify other clauses here, but re-scanning is simpler for small n
                    # For efficiency, we can remove the variable from other clauses in a more advanced implementation,
                    # but for n=50, re-scanning is acceptable.
        
        # 2. Check if all clauses satisfied
        for clause in clauses:
            sat = False
            for v, p in clause:
                if assignment[v] is None:
                    continue
                if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                    sat = True
                    break
            if not sat:
                return None # Clause not satisfied and no unassigned vars to fix it? 
                # Actually, if unassigned vars exist, we proceed to branching.
                # If unassigned vars exist but clause is not sat, it's not UNSAT yet.
                # Wait, the logic above:
                # If unassigned_count > 0 and not satisfied, we need to branch.
                # The check "if not sat: return None" is only valid if unassigned_count == 0.
                # Let's re-evaluate the check.
        
        # Re-evaluate satisfaction status properly
        all_satisfied = True
        for clause in clauses:
            sat = False
            has_unassigned = False
            for v, p in clause:
                if assignment[v] is None:
                    has_unassigned = True
                else:
                    if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                        sat = True
                        break
            if not sat:
                if not has_unassigned:
                    return None # UNSAT
                else:
                    all_satisfied = False
                    # We need to branch on an unassigned variable in this clause or any clause
                    # Break to find a branch variable
                    break
        
        if all_satisfied:
            # Fill in any remaining unassigned variables arbitrarily (e.g., 0)
            for v in range(n):
                if assignment[v] is None:
                    assignment[v] = 0
            return assignment
            
        # 3. Branching
        # Choose a variable to branch on. Heuristic: pick a variable from the first unsatisfied clause.
        # Find first unsatisfied clause
        branch_clause = None
        for clause in clauses:
            sat = False
            for v, p in clause:
                if assignment[v] is None:
                    continue
                if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                    sat = True
                    break
            if not sat:
                branch_clause = clause
                break
                
        if branch_clause is None:
             # Should not happen if all_satisfied was false
             for v in range(n):
                if assignment[v] is None:
                    assignment[v] = 0
             return assignment

        # Pick first unassigned variable in branch_clause
        branch_var = -1
        for v, p in branch_clause:
            if assignment[v] is None:
                branch_var = v
                break
        
        if branch_var == -1:
            # Fallback
            for v in range(n):
                if assignment[v] is None:
                    branch_var = v
                    break
            
        if branch_var == -1:
            return None # No vars to branch? But not sat? Error.

        # Try True
        assignment_copy = assignment[:]
        assignment_copy[branch_var] = 1
        res = dpll(clauses, assignment_copy, depth + 1)
        if res is not None:
            return res
            
        # Try False
        assignment_copy[branch_var] = 0
        res = dpll(clauses, assignment_copy, depth + 1)
        if res is not None:
            return res
            
        return None

    # Initialize assignment for DPLL
    dpll_assignment = [None] * n
    
    # Run DPLL with a time limit? 
    # DPLL can be exponential. For n=50, it might hang on hard instances.
    # We should limit the depth or time for DPLL.
    # Let's run DPLL for a short burst or limited depth.
    # Actually, for planted SAT, DPLL with unit propagation is very strong.
    # But to be safe against exponential blowup, let's add a step limit or time check.
    
    # Simple time check inside DPLL? 
    # Since DPLL is recursive, we can't easily inject time checks without passing state.
    # Let's just run it and hope it finishes quickly for easy instances.
    # If it's hard, it will take too long.
    # Better: Run DPLL with a limited number of nodes.
    
    dpll_nodes = [0]
    def dpll_limited(clauses, assignment, depth=0):
        if time.time() > deadline - 5.0: # Leave 5s for WalkSAT
            return None
        dpll_nodes[0] += 1
        if dpll_nodes[0] > 10000: # Limit search space for DPLL
            return None
            
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            for i in range(len(clauses)):
                clause = clauses[i]
                # Check if clause is satisfied
                sat = False
                unassigned_count = 0
                unit_var = -1
                unit_pol = 0
                
                for v, p in clause:
                    if assignment[v] is None:
                        unassigned_count += 1
                        unit_var = v
                        unit_pol = p
                    else:
                        if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                            sat = True
                            break
                            
                if sat:
                    continue
                if unassigned_count == 0:
                    return None # UNSAT
                if unassigned_count == 1:
                    assignment[unit_var] = 1 if unit_pol == 1 else 0
                    changed = True
        
        # Check if all satisfied
        for clause in clauses:
            sat = False
            has_unassigned = False
            for v, p in clause:
                if assignment[v] is None:
                    has_unassigned = True
                else:
                    if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                        sat = True
                        break
            if not sat:
                if not has_unassigned:
                    return None
                break # Need to branch
        
        # If we get here, we didn't return None, so either all sat or need to branch.
        # Check if all sat
        all_sat = True
        for clause in clauses:
            sat = False
            for v, p in clause:
                if assignment[v] is None:
                    continue
                if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        
        if all_sat:
            for v in range(n):
                if assignment[v] is None:
                    assignment[v] = 0
            return assignment
        
        # Branch
        # Find first clause with unassigned var
        branch_var = -1
        for clause in clauses:
            sat = False
            for v, p in clause:
                if assignment[v] is not None:
                    if (assignment[v] == 1 and p == 1) or (assignment[v] == 0 and p == -1):
                        sat = True
                        break
            if not sat:
                for v, p in clause:
                    if assignment[v] is None:
                        branch_var = v
                        break
                if branch_var != -1:
                    break
        
        if branch_var == -1:
             # Fallback
             for v in range(n):
                 if assignment[v] is None:
                     branch_var = v
                     break
             if branch_var == -1:
                 return None

        # Try True
        assignment_copy = assignment[:]
        assignment_copy[branch_var] = 1
        res = dpll_limited(clauses, assignment_copy, depth + 1)
        if res is not None:
            return res
            
        # Try False
        assignment_copy[branch_var] = 0
        res = dpll_limited(clauses, assignment_copy, depth + 1)
        if res is not None:
            return res
            
        return None

    # Convert clauses to (v, p) format for DPLL
    dpll_clauses = []
    for i in range(m):
        dpll_clause = []
        for j in range(len(clause_vars[i])):
            dpll_clause.append((clause_vars[i][j], clause_pols[i][j]))
        dpll_clauses.append(dpll_clause)

    # Run DPLL
    solution = dpll_limited(dpll_clauses, dpll_assignment)
    if solution is not None:
        # Convert to boolean list
        return [bool(x) for x in solution]

    # If DPLL failed or timed out, use WalkSAT
    # We can use the partial assignment from DPLL as a starting point?
    # Or just start fresh? Starting fresh is safer if DPLL got stuck in a bad state.
    # But if DPLL assigned some variables correctly, it might help.
    # Let's just start fresh for simplicity and robustness, or use the last assignment if it's valid.
    # Actually, let's just start fresh random WalkSAT.

    rng = random.Random(0)
    
    # Pre-compute for WalkSAT
    # var_pos_in_clause[v][c_idx] = pos? No, var_to_clauses has (c_idx, pos).
    # We can build a lookup table: var_pos_map[v] = {c_idx: pos}
    var_pos_map = [{} for _ in range(n)]
    for v in range(n):
        for c_idx, pos in var_to_clauses[v]:
            var_pos_map[v][c_idx] = pos

    def init_state_walksat(assignment):
        unsat_counts = [0] * m
        num_unsat = 0
        for i in range(m):
            count = 0
            for j in range(len(clause_vars[i])):
                v = clause_vars[i][j]
                p = clause_pols[i][j]
                if (assignment[v] and p == -1) or (not assignment[v] and p == 1):
                    count += 1
            unsat_counts[i] = count
            if count == len(clause_vars[i]):
                num_unsat += 1
        return unsat_counts, num_unsat

    def flip_var_walksat(assignment, v, unsat_counts, num_unsat):
        """Flip variable v and update unsat_counts and num_unsat incrementally."""
        assignment[v] = not assignment[v]
        
        for c_idx, pos in var_to_clauses[v]:
            old_count = unsat_counts[c_idx]
            c_vars = clause_vars[c_idx]
            c_pols = clause_pols[c_idx]
            
            # Recalculate count for this clause
            new_count = 0
            for k in range(len(c_vars)):
                vv = c_vars[k]
                pp = c_pols[k]
                if (assignment[vv] and pp == -1) or (not assignment[vv] and pp == 1):
                    new_count += 1
            
            unsat_counts[c_idx] = new_count
            
            if old_count == len(c_vars) and new_count < len(c_vars):
                num_unsat -= 1
            elif old_count < len(c_vars) and new_count == len(c_vars):
                num_unsat += 1
                
        return num_unsat

    best_assignment = None
    best_num_unsat = float('inf')
    
    while time.time() < deadline:
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_counts, num_unsat = init_state_walksat(assignment)
        
        if num_unsat == 0:
            return assignment
            
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
            
        steps = 0
        max_steps = 100000
        noise = 0.05
        consecutive_stuck = 0
        
        while steps < max_steps:
            if time.time() >= deadline:
                break
                
            if num_unsat == 0:
                return assignment
                
            if num_unsat < best_num_unsat:
                best_num_unsat = num_unsat
                best_assignment = assignment[:]
                
            # Find unsat clauses
            unsat_indices = [i for i in range(m) if unsat_counts[i] == len(clause_vars[i])]
            if not unsat_indices:
                return assignment

            clause_idx = rng.choice(unsat_indices)
            
            # Heuristic: Minimize increase in num_unsat
            best_cost = float('inf')
            best_var_to_flip = -1
            
            for j in range(len(clause_vars[clause_idx])):
                v = clause_vars[clause_idx][j]
                
                # Calculate cost of flipping v
                # We need to simulate the flip to see change in num_unsat
                # To avoid modifying state, we can calculate delta.
                # But incremental update is fast. Let's just calculate delta analytically?
                # Or just pick the best var by looking at impact.
                
                # For each clause containing v:
                # If clause is currently unsat (count == len), flipping v (which is unsat in this clause)
                # will make it count-1. It stays unsat unless count was 1? No, count==len.
                # If len=3, count=3. Flip v -> count=2. Still unsat.
                # If clause is currently sat (count < len), flipping v (which is sat in this clause)
                # will make it count+1. It becomes unsat if count+1 == len.
                
                delta_num_unsat = 0
                for c_idx, pos in var_to_clauses[v]:
                    old_count = unsat_counts[c_idx]
                    c_len = len(clause_vars[c_idx])
                    
                    # Check if v is currently satisfying or unsatisfying c_idx
                    c_pol = clause_pols[c_idx][pos]
                    is_v_sat = (assignment[v] and c_pol == 1) or (not assignment[v] and c_pol == -1)
                    
                    if is_v_sat:
                        # v is currently helping. Flipping it hurts.
                        # New count = old_count + 1
                        if old_count + 1 == c_len:
                            # Becomes unsat
                            delta_num_unsat += 1
                    else:
                        # v is currently hurting. Flipping it helps.
                        # New count = old_count - 1
                        if old_count == 1:
                            # Becomes sat (count 0)
                            delta_num_unsat -= 1
                        # If old_count > 1, it was unsat, becomes less unsat, still unsat. No change to num_unsat.
                        # If old_count == 0, it was sat? No, if v is hurting, it can't be that count is 0 unless other vars compensate?
                        # Wait, if v is hurting (unsat literal), and count is 0, that's impossible.
                        # So if v is hurting, count >= 1.
                
                cost = num_unsat + delta_num_unsat
                if cost < best_cost:
                    best_cost = cost
                    best_var_to_flip = v
            
            # Apply best flip
            if best_var_to_flip != -1:
                v = best_var_to_flip
                num_unsat = flip_var_walksat(assignment, v, unsat_counts, num_unsat)
                
                if num_unsat < best_num_unsat:
                    consecutive_stuck = 0
                else:
                    consecutive_stuck += 1
                    
                # Adaptive noise
                if consecutive_stuck > 100:
                    noise = 0.2
                elif consecutive_stuck > 50:
                    noise = 0.1
                else:
                    noise = 0.05
                    
            else:
                # Should not happen
                break

            # Noise flip
            if rng.random() < noise:
                j = rng.randint(0, len(clause_vars[clause_idx]) - 1)
                v = clause_vars[clause_idx][j]
                num_unsat = flip_var_walksat(assignment, v, unsat_counts, num_unsat)
                
            steps += 1

    # Return best found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [False] * n
