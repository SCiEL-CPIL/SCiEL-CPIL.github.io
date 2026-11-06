"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        count = 0
        for cv in clause_vars:
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def is_satisfied(assignment, cv):
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def flip_var(assignment, var_idx):
        assignment[var_idx] = not assignment[var_idx]
    
    def walk_sat(assignment, rng, max_flips):
        """WalkSAT with adaptive noise."""
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return True
        
        # Track unsatisfied clauses
        unsat_clauses = []
        for i, cv in enumerate(clause_vars):
            if not is_satisfied(assignment, cv):
                unsat_clauses.append(i)
        
        flip_count = 0
        while flip_count < max_flips:
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # For each literal in the clause, compute the delta (change in unsat count) if we flip
            best_var = -1
            best_delta = float('inf')
            best_is_random = False
            
            # Adaptive noise: decrease over time
            noise = 0.5 * max(0, 1 - flip_count / max_flips)
            
            for var_idx, is_pos in cv:
                # If we flip this variable, what happens to unsat count?
                delta = 0
                for ci2, is_pos2 in var_clauses[var_idx]:
                    cv2 = clause_vars[ci2]
                    # Check if this clause is currently satisfied
                    currently_sat = False
                    for v2, p2 in cv2:
                        if assignment[v2] == p2:
                            currently_sat = True
                            break
                    
                    # After flip, will this clause still be satisfied?
                    # The flipped var's contribution changes
                    old_val = assignment[var_idx]
                    new_val = not old_val
                    
                    if var_idx in [v for v, _ in cv2]:
                        # Find the position of var_idx in cv2
                        for v2, p2 in cv2:
                            if v2 == var_idx:
                                # Old contribution
                                old_contrib = (old_val == p2)
                                # New contribution
                                new_contrib = (new_val == p2)
                                break
                        
                        if currently_sat:
                            # Was satisfied, check if still satisfied after flip
                            # If this var was the only one satisfying it, it becomes unsat
                            if old_contrib and not new_contrib:
                                # Check if any other var in the clause is satisfying
                                other_sat = False
                                for v2, p2 in cv2:
                                    if v2 != var_idx and assignment[v2] == p2:
                                        other_sat = True
                                        break
                                if not other_sat:
                                    delta += 1  # Becomes unsat
                        else:
                            # Was unsat, check if it becomes sat
                            if not old_contrib and new_contrib:
                                delta -= 1  # Becomes sat
                            
                    # Actually, let me simplify: just compute delta directly
                    # Reset and do it properly
                # This approach is getting complex, let me use a simpler delta computation
                
            # Simpler approach: for each var in the clause, compute delta
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = 0
                for ci2, is_pos2 in var_clauses[var_idx]:
                    cv2 = clause_vars[ci2]
                    # Is this clause currently satisfied?
                    cur_sat = False
                    for v2, p2 in cv2:
                        if assignment[v2] == p2:
                            cur_sat = True
                            break
                    
                    # After flipping var_idx, is it still satisfied?
                    # We need to check if any other literal in cv2 is satisfied
                    # OR if the flipped literal becomes satisfied
                    post_sat = False
                    for v2, p2 in cv2:
                        if v2 == var_idx:
                            # This is the flipped variable
                            if (not assignment[var_idx]) == p2:
                                post_sat = True
                                break
                        else:
                            if assignment[v2] == p2:
                                post_sat = True
                                break
                    
                    if cur_sat and not post_sat:
                        delta += 1
                    elif not cur_sat and post_sat:
                        delta -= 1
                
                # Apply noise
                delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                # Fallback: random flip
                best_var = rng.choice([v for v, _ in cv])[0]
            
            flip_var(assignment, best_var)
            flip_count += 1
            
            # Update unsat clauses
            # Recompute is expensive, but let's do it periodically
            if flip_count % 10 == 0:
                new_unsat = []
                for i, cv_i in enumerate(clause_vars):
                    if not is_satisfied(assignment, cv_i):
                        new_unsat.append(i)
                unsat_clauses = new_unsat
                if not unsat_clauses:
                    return True
        
        return len(unsat_clauses) == 0
    
    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Max flips per restart, increasing slightly
        max_flips = 5000 + restart * 500
        
        if walk_sat(assignment, rng, max_flips):
            return assignment
        
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
        # If we've done many restarts, increase max_flips
        if restart % 5 == 0:
            pass  # max_flips already increases with restart
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
