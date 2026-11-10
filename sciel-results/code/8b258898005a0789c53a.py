"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Pre-compute variables in each clause for faster access
    clause_vars = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        clause_vars.append(vars_in_c)
        
    # Precompute variable to clause indices mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_to_clauses[v].append(i)

    # Precompute literal signs for faster satisfaction check
    # clause_lits[i] = list of (var_idx, sign_bool)
    clause_lits = []
    for c in clauses:
        lits = []
        for l in c:
            lits.append((abs(l) - 1, l > 0))
        clause_lits.append(lits)

    def is_clause_satisfied(ci, assignment):
        # Use precomputed literals
        for v, s in clause_lits[ci]:
            if assignment[v] == s:
                return True
        return False

    def count_unsat_clauses(assignment):
        count = 0
        for i in range(m):
            if not is_clause_satisfied(i, assignment):
                count += 1
        return count

    def get_unsat_indices(assignment):
        return [i for i in range(m) if not is_clause_satisfied(i, assignment)]

    def walksat_hybrid(assignment, max_steps, start_p=0.8, end_p=0.2):
        # Initialize clause status and unsat list
        clause_status = [False] * m
        unsat_clauses = []
        for i in range(m):
            if is_clause_satisfied(i, assignment):
                clause_status[i] = True
            else:
                unsat_clauses.append(i)
                
        if not unsat_clauses:
            return True, assignment
        
        # Track best unsat count for intensification
        min_unsat_seen = len(unsat_clauses)
        steps_since_improvement = 0
        last_improvement_step = 0
        
        for step in range(max_steps):
            if not unsat_clauses:
                return True, assignment
            
            # Adaptive probability p
            # Linearly interpolate p from start_p to end_p over max_steps
            progress = step / max_steps if max_steps > 0 else 0
            p = start_p + (end_p - start_p) * progress
            
            # Pick a random unsat clause
            ci = random.choice(unsat_clauses)
            clause = clause_lits[ci]
            vars_in_clause = clause_vars[ci]
            
            # Determine best flip using Min-Conflicts heuristic
            # We want to minimize the number of newly unsatisfied clauses
            min_conflicts = float('inf')
            best_vars = []
            
            for v in vars_in_clause:
                conflicts = 0
                for cj in var_to_clauses[v]:
                    if clause_status[cj]:
                        # Check if v is the ONLY satisfied literal in clause cj
                        # If so, flipping v will make cj unsatisfied
                        c_lits = clause_lits[cj]
                        sat_count = 0
                        is_v_satisfying = False
                        for vv, ss in c_lits:
                            if assignment[vv] == ss:
                                sat_count += 1
                                if vv == v:
                                    is_v_satisfying = True
                        if sat_count == 1 and is_v_satisfying:
                            conflicts += 1
                
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_vars = [v]
                elif conflicts == min_conflicts:
                    best_vars.append(v)
            
            # Choose variable to flip
            if random.random() < p:
                v_to_flip = random.choice(vars_in_clause)
            else:
                v_to_flip = random.choice(best_vars)
            
            # Perform flip
            old_val = assignment[v_to_flip]
            assignment[v_to_flip] = not old_val
            
            # Update clause statuses and unsat list
            # Only clauses containing v_to_flip can change
            for cj in var_to_clauses[v_to_flip]:
                was_sat = clause_status[cj]
                is_sat = is_clause_satisfied(cj, assignment)
                clause_status[cj] = is_sat
                
                if was_sat and not is_sat:
                    # Became unsat
                    unsat_clauses.append(cj)
                elif not was_sat and is_sat:
                    # Became sat
                    # Remove from unsat_clauses
                    try:
                        idx = unsat_clauses.index(cj)
                        unsat_clauses.pop(idx)
                    except ValueError:
                        pass
            
            # Check for improvement
            current_unsat = len(unsat_clauses)
            if current_unsat < min_unsat_seen:
                min_unsat_seen = current_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            # Intensification: If stuck for a while, try Min-Conflicts burst
            if steps_since_improvement > 50:
                # Min-Conflicts: Flip variable appearing in most unsat clauses
                var_count = [0] * n
                for uc_idx in unsat_clauses:
                    for v in clause_vars[uc_idx]:
                        var_count[v] += 1
                
                max_count = max(var_count)
                if max_count > 0:
                    candidates = [i for i in range(n) if var_count[i] == max_count]
                    v_mc = random.choice(candidates)
                    assignment[v_mc] = not assignment[v_mc]
                    
                    # Update statuses after Min-Conflicts flip
                    for cj in var_to_clauses[v_mc]:
                        was_sat = clause_status[cj]
                        is_sat = is_clause_satisfied(cj, assignment)
                        clause_status[cj] = is_sat
                        if was_sat and not is_sat:
                            unsat_clauses.append(cj)
                        elif not was_sat and is_sat:
                            try:
                                idx = unsat_clauses.index(cj)
                                unsat_clauses.pop(idx)
                            except ValueError:
                                pass
                    
                    # Reset improvement counter after jump
                    steps_since_improvement = 0
                    # Re-evaluate min_unsat_seen
                    current_unsat = len(unsat_clauses)
                    if current_unsat < min_unsat_seen:
                        min_unsat_seen = current_unsat

        return False, assignment

    # Main loop with restarts
    start_time = time.time()
    max_time = 290.0  # Safety margin
    
    restart = 0
    best_assignment = None
    best_unsat_count = float('inf')
    
    while time.time() - start_time < max_time:
        restart += 1
        # Exponential increase in steps per restart
        # Start with 500, then 1000, 2000, etc.
        steps = min(500 * (2 ** (restart - 1)), 50000)
        
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        solved, final_assignment = walksat_hybrid(assignment, steps, start_p=0.8, end_p=0.2)
        
        if solved:
            return final_assignment
        
        # Track best unsolved assignment
        unsat_count = count_unsat_clauses(final_assignment)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = final_assignment[:]

        # If we are very close to solving (e.g., < 3 unsat clauses), 
        # we might try a targeted search or just let the next restart handle it.
        # For N=50, if unsat is low, it's likely solvable with a few more flips.
        if best_unsat_count <= 2:
             # Try a short, high-greedy polish on the best assignment
             polish_assignment = best_assignment[:]
             # Use a very low p (greedy) for polish
             solved_polish, polished = walksat_hybrid(polish_assignment, 1000, start_p=0.1, end_p=0.0)
             if solved_polish:
                 return polished

    # If time runs out, return the best assignment found
    return best_assignment
# EVOLVE-BLOCK-END
