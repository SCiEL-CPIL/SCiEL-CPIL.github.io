"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Improved WalkSAT with deadline guard, weighted literal selection, and restarts.
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # 30s safety margin before 310s kill
    
    def count_unsat(assignment, clauses):
        """Count unsatisfied clauses."""
        count = 0
        for clause in clauses:
            sat = False
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def is_satisfied(assignment, clause):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False
    
    def walk_sat_single(assignment, clauses, n, rng, max_flips, deadline):
        """Run WalkSAT from current assignment. Returns (assignment, unsat_count)."""
        for flip in range(max_flips):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat = [c for c in clauses if not is_satisfied(assignment, c)]
            if not unsat:
                return assignment, 0
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat)
            
            # Weighted literal selection: minimize unsatisfied literals in clause
            # Count how many literals would become satisfied if we flip each variable
            best_score = float('inf')
            candidates = []
            for l in clause:
                var = abs(l) - 1
                # If we flip this variable, how many literals in this clause become satisfied?
                # Actually, we want to minimize the number of UNSATISFIED literals after flip
                # Current unsatisfied count in this clause
                cur_unsat = 0
                for ll in clause:
                    if assignment[abs(ll) - 1] != (ll > 0):
                        cur_unsat += 1
                
                # If we flip variable var, check which literals in clause become satisfied
                # A literal l is satisfied if assignment[abs(l)-1] == (l>0)
                # After flip, assignment[var] becomes not assignment[var]
                # For literal l in clause:
                #   if abs(l)-1 == var: new_val = not assignment[var]
                #   else: same as before
                new_unsat = 0
                for ll in clause:
                    if abs(ll) - 1 == var:
                        new_val = not assignment[var]
                    else:
                        new_val = assignment[abs(ll) - 1]
                    if new_val != (ll > 0):
                        new_unsat += 1
                
                if new_unsat < best_score:
                    best_score = new_unsat
                    candidates = [l]
                elif new_unsat == best_score:
                    candidates.append(l)
            
            # With probability p, pick from best candidates; else pick randomly
            p = 0.55
            if rng.random() < p and candidates:
                literal = rng.choice(candidates)
            else:
                literal = rng.choice(clause)
            
            assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
        
        # Count final unsatisfied
        final_unsat = 0
        for c in clauses:
            if not is_satisfied(assignment, c):
                final_unsat += 1
        return assignment, final_unsat
    
    best_assignment = None
    best_unsat = float('inf')
    
    max_flips_per_run = 5000
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        rng = random.Random(restart * 1000 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        assignment, unsat_count = walk_sat_single(assignment, clauses, n, rng, max_flips_per_run, deadline)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
    
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
