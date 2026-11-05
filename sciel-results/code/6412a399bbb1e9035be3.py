"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Improved WalkSAT with deadline guard, smarter flip selection, and restarts.
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # Safety margin before 300s kill
    
    def count_sat(clause, assignment):
        """Count how many literals in clause are satisfied."""
        cnt = 0
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                cnt += 1
                if cnt >= 1:
                    break
        return cnt

    def is_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def get_unsat_clauses(assignment):
        return [c for c in clauses if not is_satisfied(c, assignment)]

    def walksat_step(assignment, rng, unsat_clauses):
        """Pick an unsat clause and flip a variable using WalkSAT heuristic."""
        # Pick a random unsatisfied clause
        clause = rng.choice(unsat_clauses)
        
        # Use WalkSAT flip rule: with prob p, pick a literal that maximizes
        # number of satisfied literals in other clauses (minimizes damage)
        # Simplified: try each literal in the clause, pick one that doesn't
        # break too many other clauses
        best_literal = None
        best_score = -1
        
        # Try a few random literals from the clause
        for literal in clause:
            var_idx = abs(literal) - 1
            # Simulate flipping and count how many other clauses become unsatisfied
            # This is expensive, so we use a simpler heuristic
            # Count how many clauses this variable appears in with the same polarity
            damage = 0
            for c in clauses:
                if c == clause:
                    continue
                for l in c:
                    if abs(l) == abs(literal):
                        # Check if this clause is currently satisfied by other literals
                        other_sat = False
                        for l2 in c:
                            if l2 != l and assignment[abs(l2) - 1] == (l2 > 0):
                                other_sat = True
                                break
                        if not other_sat:
                            # Flipping would break this clause
                            damage += 1
                        break
            
            score = -damage
            if score > best_score:
                best_score = score
                best_literal = literal
        
        if best_literal is None:
            best_literal = rng.choice(clause)
        
        assignment[abs(best_literal) - 1] = not assignment[abs(best_literal) - 1]
        return assignment

    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat = get_unsat_clauses(assignment)
        if not unsat:
            return assignment
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
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
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return current assignment
    return assignment
# EVOLVE-BLOCK-END
