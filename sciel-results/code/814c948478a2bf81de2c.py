import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
        
    start_time = time.time()
    deadline = start_time + 325.0 # Safety margin
    
    # Pre-process clauses
    # clause_vars: list of (var_idx, is_positive)
    clause_vars = []
    # var_clauses: list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            v = abs(l) - 1
            pos = l > 0
            cv.append((v, pos))
            var_clauses[v].append((i, pos))
        clause_vars.append(cv)
        
    # WalkSAT parameters
    walk_prob = 0.5
    restart_limit = 1000 # Restart after this many steps without improvement
    max_steps_per_restart = 50000 # Hard cap per restart to prevent infinite loops in bad basins
    
    best_unsat = float('inf')
    best_assignment = None
    rng = random.Random(42)
    
    def initialize():
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat = [0] * m
        unsat_count = 0
        
        for i, cv in enumerate(clause_vars):
            sat_count = 0
            for v, pos in cv:
                if assignment[v] == pos:
                    sat_count += 1
            clause_sat[i] = sat_count
            if sat_count == 0:
                unsat_count += 1
        return assignment, clause_sat, unsat_count

    def get_unsat_clauses(clause_sat):
        # Return list of indices of unsatisfied clauses
        return [i for i, s in enumerate(clause_sat) if s == 0]

    def flip_variable(assignment, clause_sat, unsat_count, v):
        # Flip variable v
        assignment[v] = not assignment[v]
        new_unsat = unsat_count
        
        for ci, pos in var_clauses[v]:
            # Determine old satisfaction of this literal
            # If assignment[v] is now True, and pos is True, it is satisfied.
            # If assignment[v] is now True, and pos is False, it is unsatisfied.
            # We need to know if it was satisfied BEFORE the flip.
            
            # Current state of assignment[v] is the NEW value.
            # So OLD value was NOT assignment[v].
            
            # Let's determine if the literal was satisfied before.
            # Literal is (v, pos).
            # Satisfied if old_assignment[v] == pos.
            # old_assignment[v] = not assignment[v].
            
            was_satisfied = (not assignment[v]) == pos
            
            if was_satisfied:
                # It was satisfied, now it is not.
                # clause_sat[ci] decreases by 1
                old_sat = clause_sat[ci]
                new_sat = old_sat - 1
                clause_sat[ci] = new_sat
                
                # Check if it became unsatisfied
                if new_sat == 0 and old_sat > 0:
                    new_unsat += 1
            else:
                # It was not satisfied, now it is.
                # clause_sat[ci] increases by 1
                old_sat = clause_sat[ci]
                new_sat = old_sat + 1
                clause_sat[ci] = new_sat
                
                # Check if it became satisfied
                if old_sat == 0:
                    new_unsat -= 1
                    
        return new_unsat

    def evaluate_flip(assignment, clause_sat, v):
        # Calculate change in unsat count if we flip v
        # Returns the NEW unsat count
        # We simulate the flip logic without modifying state
        new_unsat = 0 # We only need the delta, but calculating absolute is safer if we pass current unsat? 
        # Actually, let's just return the delta.
        delta = 0
        for ci, pos in var_clauses[v]:
            was_satisfied = (not assignment[v]) == pos # Current assignment is OLD for this evaluation
            if was_satisfied:
                # Becomes unsatisfied
                if clause_sat[ci] == 1:
                    # Only one satisfied literal, so it becomes 0
                    delta += 1
            else:
                # Becomes satisfied
                if clause_sat[ci] == 0:
                    # Was 0, becomes 1
                    delta -= 1
        return delta

    while time.time() < deadline:
        assignment, clause_sat, unsat_count = initialize()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        if best_unsat == 0:
            break
            
        steps_since_improvement = 0
        local_unsat = unsat_count
        
        for _ in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            unsat_indices = get_unsat_clauses(clause_sat)
            if not unsat_indices:
                # Solved!
                best_unsat = 0
                best_assignment = assignment[:]
                break
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            cv = clause_vars[ci]
            
            # Decide which variable to flip
            if rng.random() < walk_prob:
                v, _ = rng.choice(cv)
            else:
                # Min-conflicts: pick v in cv that minimizes new unsat count
                best_v = None
                min_delta = float('inf')
                for v, _ in cv:
                    delta = evaluate_flip(assignment, clause_sat, v)
                    if delta < min_delta:
                        min_delta = delta
                        best_v = v
                v = best_v
            
            # Flip v
            unsat_count = flip_variable(assignment, clause_sat, unsat_count, v)
            
            if unsat_count < local_unsat:
                local_unsat = unsat_count
                steps_since_improvement = 0
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
                    if best_unsat == 0:
                        break
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > restart_limit:
                break
                
    if best_assignment is None:
        best_assignment = [False] * n
        
    return best_assignment
