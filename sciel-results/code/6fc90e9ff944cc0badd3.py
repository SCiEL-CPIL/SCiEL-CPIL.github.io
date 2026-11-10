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

    def is_clause_satisfied(ci, assignment):
        c = clauses[ci]
        for l in c:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def walksat_efficient(assignment, max_steps, p=0.5):
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
        
        for _ in range(max_steps):
            if not unsat_clauses:
                return True, assignment
            
            # Pick a random unsat clause
            ci = random.choice(unsat_clauses)
            clause = clauses[ci]
            vars_in_clause = clause_vars[ci]
            
            # Determine best flip using Min-Conflicts heuristic
            # For each var in the clause, calculate how many OTHER satisfied clauses 
            # would become unsatisfied if we flip it.
            min_conflicts = float('inf')
            best_vars = []
            
            for v in vars_in_clause:
                conflicts = 0
                for cj in var_to_clauses[v]:
                    if clause_status[cj]:
                        # Check if v is the ONLY satisfied literal in clause cj
                        # If so, flipping v will make cj unsatisfied
                        c = clauses[cj]
                        sat_count = 0
                        for l in c:
                            if assignment[abs(l) - 1] == (l > 0):
                                sat_count += 1
                        if sat_count == 1:
                            # Find if v is the one
                            for l in c:
                                if abs(l) - 1 == v:
                                    if assignment[v] == (l > 0):
                                        conflicts += 1
                                    break
                
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
                    # Linear search is O(k) where k is number of unsat clauses.
                    # For n=50, m~200, this is acceptable.
                    try:
                        idx = unsat_clauses.index(cj)
                        unsat_clauses.pop(idx)
                    except ValueError:
                        pass
                # If was_sat and is_sat, or not was_sat and not is_sat, no change to list
                
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
        # Start with 1000, then 2000, 4000, etc.
        steps = min(1000 * (2 ** (restart - 1)), 100000)
        
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        solved, final_assignment = walksat_efficient(assignment, steps, p=0.5)
        
        if solved:
            return final_assignment
        
        # Track best unsolved assignment (fewest unsat clauses)
        # This is a simple heuristic to return a "best effort" if unsolved
        # But since we return the last assignment in the original code, 
        # and the score is fraction solved, returning unsolved is 0 for that instance.
        # However, keeping track might help if we had a partial score, but here it's binary.
        # So we just continue.
        
    # If time runs out, return the last assignment (unsolved)
    return assignment
# EVOLVE-BLOCK-END
