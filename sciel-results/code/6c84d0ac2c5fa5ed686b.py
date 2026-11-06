"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute variable to clause index mapping for fast lookup
    # var_clauses[var] = list of (clause_index, literal_sign)
    # literal_sign is 1 if positive literal, -1 if negative literal in the clause
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for l in clause:
            var_idx = abs(l) - 1
            sign = 1 if l > 0 else -1
            var_clauses[var_idx].append((i, sign))
    
    def is_satisfied(assignment, clause):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for clause in clauses:
            if not is_satisfied(assignment, clause):
                count += 1
        return count

    def get_unsat_clauses(assignment):
        unsat = []
        for i, clause in enumerate(clauses):
            if not is_satisfied(assignment, clause):
                unsat.append(i)
        return unsat

    def flip_gain(assignment, var_idx):
        """Calculate the change in number of unsatisfied clauses if we flip var_idx.
        Returns (delta_unsat, var_idx). Negative delta is good (fewer unsat).
        """
        delta = 0
        for clause_idx, sign in var_clauses[var_idx]:
            clause = clauses[clause_idx]
            # Check if clause is currently unsatisfied
            currently_unsat = True
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    currently_unsat = False
                    break
            
            if currently_unsat:
                # If we flip var_idx, does it satisfy this clause?
                # The literal for var_idx in this clause is sign * (var_idx+1)
                # If sign is 1, literal is positive. If we flip to True, it satisfies.
                # If sign is -1, literal is negative. If we flip to False, it satisfies.
                new_val = not assignment[var_idx]
                if new_val == (sign == 1):
                    delta -= 1 # Clause becomes satisfied
            else:
                # Clause is currently satisfied.
                # If we flip var_idx, does it break the satisfaction?
                # Only if this variable was the *only* satisfying literal.
                # Check if any other literal satisfies it.
                other_sat = False
                for l in clause:
                    if abs(l) - 1 == var_idx:
                        continue
                    if assignment[abs(l) - 1] == (l > 0):
                        other_sat = True
                        break
                if not other_sat:
                    delta += 1 # Clause becomes unsatisfied
        return delta

    # WalkSAT parameters
    max_restarts = 50
    base_flips = 1000
    p_random = 0.45 # Probability of random flip
    
    for restart in range(max_restarts):
        rng = random.Random(restart * 1000 + 7)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial check
        if count_unsat(assignment) == 0:
            return assignment
        
        flips_limit = base_flips * (restart + 1)
        
        for _ in range(flips_limit):
            unsat_clauses = get_unsat_clauses(assignment)
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clauses[clause_idx]
            
            # Choose variable to flip
            if rng.random() < p_random:
                # Random flip
                l = rng.choice(clause)
                var_idx = abs(l) - 1
            else:
                # Min-conflicts heuristic: pick variable that minimizes delta_unsat
                best_var = -1
                best_delta = float('inf')
                for l in clause:
                    var_idx = abs(l) - 1
                    delta = flip_gain(assignment, var_idx)
                    if delta < best_delta:
                        best_delta = delta
                        best_var = var_idx
                var_idx = best_var
            
            assignment[var_idx] = not assignment[var_idx]
    
    # Fallback: return last assignment or random
    return assignment
# EVOLVE-BLOCK-END
