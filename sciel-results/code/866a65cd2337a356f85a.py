"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    deadline = time.time() + 290  # Safety margin before 300s kill

    # Precompute variable appearances for faster damage calculation
    # var_occurrences[var_idx] = list of (clause_idx, literal)
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for l in clause:
            var_idx = abs(l) - 1
            var_occurrences[var_idx].append((ci, l))

    def is_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def get_unsat_clauses(assignment):
        return [c for c in clauses if not is_satisfied(c, assignment)]

    def walksat_step(assignment, rng, unsat_clauses):
        """Pick an unsat clause and flip a variable using WalkSAT heuristic."""
        clause = rng.choice(unsat_clauses)
        
        best_literal = None
        best_score = -1
        
        for literal in clause:
            var_idx = abs(literal) - 1
            damage = 0
            
            # Count how many clauses would become unsatisfied if we flip this variable
            # A clause becomes unsatisfied if:
            # 1. It's currently satisfied ONLY by this variable (with its current value)
            # 2. After flipping, this variable no longer satisfies it
            # 3. No other literal in the clause satisfies it
            
            current_val = assignment[var_idx]
            this_satisfies = (current_val == (literal > 0))
            
            if not this_satisfies:
                # Flipping this variable would make it satisfy the clause, so no damage from this perspective
                # But we need to check if it was satisfying other clauses
                pass
            
            for ci, l in var_occurrences[var_idx]:
                if clauses[ci] is clause:
                    continue
                c = clauses[ci]
                # Check if this clause is currently satisfied by this variable
                var_satisfies_clause = (current_val == (l > 0))
                if var_satisfies_clause:
                    # Check if any other literal satisfies this clause
                    other_sat = False
                    for l2 in c:
                        if abs(l2) != var_idx + 1:
                            if assignment[abs(l2) - 1] == (l2 > 0):
                                other_sat = True
                                break
                    if not other_sat:
                        # This clause is satisfied ONLY by this variable
                        # Flipping will break it
                        damage += 1
            
            # We want to minimize damage, but also prefer literals that satisfy the chosen clause
            # Since all literals in the clause will satisfy it after flip, we just minimize damage
            score = -damage
            if score > best_score:
                best_score = score
                best_literal = literal
        
        if best_literal is None:
            best_literal = rng.choice(clause)
        
        assignment[abs(best_literal) - 1] = not assignment[abs(best_literal) - 1]
        return assignment

    def simulated_annealing_polish(assignment, rng, unsat_clauses):
        """Simulated annealing to polish the assignment, using soft-min surrogate."""
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment, unsat_clauses, True
        
        best_assignment = assignment[:]
        best_unsat = current_unsat
        best_unsat_clauses = unsat_clauses[:]
        
        # Temperature schedule
        start_temp = 2.0
        end_temp = 0.01
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Geometric cooling
            progress = step / max_steps
            temp = start_temp * (end_temp / start_temp) ** progress
            
            # Pick a random variable to flip
            var_idx = rng.randint(0, n - 1)
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Recalculate unsat clauses
            new_unsat = get_unsat_clauses(assignment)
            new_unsat_count = len(new_unsat)
            
            # Delta in unsat count (we want to minimize)
            delta = new_unsat_count - current_unsat
            
            if delta <= 0:
                # Accept improving or equal moves
                current_unsat = new_unsat_count
                unsat_clauses = new_unsat
                if new_unsat_count < best_unsat:
                    best_unsat = new_unsat_count
                    best_assignment = assignment[:]
                    best_unsat_clauses = new_unsat[:]
                    if best_unsat == 0:
                        return best_assignment, best_unsat_clauses, True
            else:
                # Accept worsening moves with probability exp(-delta/temp)
                if temp > 0:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        current_unsat = new_unsat_count
                        unsat_clauses = new_unsat
                    else:
                        # Reject: undo the flip
                        assignment[var_idx] = old_val
        
        return best_assignment, best_unsat_clauses, best_unsat == 0

    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 100
    steps_per_restart = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat = get_unsat_clauses(assignment)
        if not unsat:
            return assignment
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 10:
                break
            
            if not unsat:
                return assignment
            
            # Track best assignment
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment[:]
            
            # WalkSAT step
            assignment = walksat_step(assignment, rng, unsat)
            
            # Update unsat clauses
            unsat = get_unsat_clauses(assignment)
        
        if not unsat:
            return assignment
        
        # If we haven't solved it, try simulated annealing polish
        if time.time() > deadline - 15:
            break
        
        polished_assignment, polished_unsat, solved = simulated_annealing_polish(
            assignment, rng, unsat
        )
        
        if solved:
            return polished_assignment
        
        if len(polished_unsat) < best_unsat_count:
            best_unsat_count = len(polished_unsat)
            best_assignment = polished_assignment
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return current assignment
    return assignment
# EVOLVE-BLOCK-END
