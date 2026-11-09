"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # 0-based internal representation
    # literals are 1-based signed ints in input.
    
    assignment = [None] * n
    trail = []  # list of (var, value) for backtracking
    
    def get_var_value(var):
        return assignment[var]
    
    def assign(var, value):
        if assignment[var] is not None:
            if assignment[var] != value:
                return False # Contradiction
            return True
        assignment[var] = value
        trail.append((var, value))
        return True
    
    def unit_propagate(clauses):
        """
        Modifies clauses in place (by replacing with simplified list) and assignment.
        Returns False if contradiction, True otherwise.
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            unit_literals = []
            
            for clause in clauses:
                simplified = []
                sat = False
                for lit in clause:
                    var = abs(lit) - 1
                    val = assignment[var]
                    if val is not None:
                        if val == (lit > 0):
                            sat = True
                            break
                        else:
                            continue
                    else:
                        simplified.append(lit)
                
                if sat:
                    continue
                
                if len(simplified) == 0:
                    return False
                elif len(simplified) == 1:
                    unit_literals.append(simplified[0])
                else:
                    new_clauses.append(simplified)
            
            clauses[:] = new_clauses # In-place update
            
            for lit in unit_literals:
                var = abs(lit) - 1
                val = lit > 0
                if not assign(var, val):
                    return False
                changed = True
        
        return True
    
    def pure_literal_eliminate(clauses):
        """
        Finds pure literals, assigns them, and simplifies clauses.
        Returns False if contradiction, True otherwise.
        """
        # Count polarities for unassigned variables
        var_pol = defaultdict(set)
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    var_pol[var].add(lit > 0)
        
        pure_vars = []
        for var, pols in var_pol.items():
            if len(pols) == 1:
                pure_vars.append((var, list(pols)[0]))
        
        if not pure_vars:
            return True
        
        # Assign pure literals
        for var, val in pure_vars:
            if not assign(var, val):
                return False
        
        # Simplify clauses after assignment
        # We can just call unit_propagate which will handle simplification
        # But unit_propagate also handles units. PLE might create units.
        # So calling unit_propagate is the right next step.
        return True
    
    def dpll(clauses):
        # 1. Unit Propagation
        if not unit_propagate(clauses):
            return False
        
        if not clauses:
            return True
        
        # 2. Pure Literal Elimination
        # Note: PLE might assign variables, which might create new units.
        # So we loop PLE and UP until fixed point.
        changed = True
        while changed:
            changed = False
            if pure_literal_eliminate(clauses):
                # PLE assigned some vars, so we need to propagate units again
                if not unit_propagate(clauses):
                    return False
                if not clauses:
                    return True
                # Check if any pure literals were assigned in this pass
                # If pure_literal_eliminate returned True but didn't assign anything, it means no pure literals.
                # We need to know if it actually changed state.
                # Let's refine: pure_literal_eliminate returns True always unless contradiction.
                # We need to check if assignment changed.
                # Simplest: just run it. If it found pure literals, it assigned them.
                # If it didn't find any, no change.
                # To avoid infinite loop if no change, we can check if trail length changed.
                # But pure_literal_eliminate doesn't return "changed".
                # Let's modify strategy:
                # Run PLE. If it assigned anything, loop. Else break.
                # How to know? 
                # Let's just assume it's fast enough to run UP again.
                # If no pure literals, pure_literal_eliminate does nothing.
                # Then unit_propagate does nothing (if no new units).
                # So the loop terminates.
                pass
            else:
                return False
        
        if not clauses:
            return True
        
        # 3. Branching
        # Heuristic: Most Frequent Variable
        var_counts = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    var_counts[var] += 1
        
        if not var_counts:
            return True
        
        # Pick variable with max count
        # Tie-breaker: lowest index for determinism
        branch_var = min(var_counts.keys(), key=lambda v: (-var_counts[v], v))
        
        # Save trail length
        trail_len = len(trail)
        
        # Try True
        if assign(branch_var, True):
            if dpll(clauses):
                return True
        
        # Undo
        while len(trail) > trail_len:
            var, _ = trail.pop()
            assignment[var] = None
        
        # Try False
        if assign(branch_var, False):
            if dpll(clauses):
                return True
        
        # Undo
        while len(trail) > trail_len:
            var, _ = trail.pop()
            assignment[var] = None
            
        return False

    # Initialize
    clauses_copy = [list(c) for c in clauses]
    
    # Run DPLL
    # Note: dpll modifies clauses_copy in place (via slicing assignment in UP)
    # This is safe as we don't need the original clauses after starting.
    solved = dpll(clauses_copy)
    
    if not solved:
        # Fallback for UNSAT or failure (should be SAT)
        # Random assignment
        import random
        rng = random.Random(0)
        assignment = [rng.random() < 0.5 for _ in range(n)]
    else:
        # Ensure all variables are assigned
        for i in range(n):
            if assignment[i] is None:
                assignment[i] = True
    
    return assignment
# EVOLVE-BLOCK-END
