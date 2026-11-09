"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocess: convert to 0-based indices
    # Represent literals as integers: var+1 for positive, -(var+1) for negative
    # This is standard SATLIB format.
    # However, the input is already signed 1-based literals.
    # We will keep them as is, but ensure we handle them correctly in DPLL.
    # Let's use a list of lists of integers.
    
    # To speed up, we can pre-process into a structure that allows fast unit propagation.
    # For n=50, a simple recursive DPLL with copying is likely fast enough if optimized.
    # But let's implement a robust DPLL.
    
    # We will represent the state as:
    # assignment: list of bools or None
    # clauses: list of lists of literals (integers)
    
    # Optimization: Use a trail for backtracking to avoid deep copying.
    # But for simplicity and correctness first, I'll use a clean recursive approach
    # with clause simplification.
    
    def dpll(assignment, clauses):
        """
        assignment: list of length n, elements are True, False, or None
        clauses: list of lists of integers (1-based signed literals)
        Returns True if satisfiable, updating assignment.
        """
        # 1. Check for empty clause (contradiction)
        # 2. Unit Propagation
        # 3. Pure Literal Elimination
        # 4. Branching
        
        # We'll do unit propagation first.
        # We need to repeatedly simplify clauses based on assignment.
        
        # To avoid infinite loops and ensure termination, we'll do a fixed-point iteration.
        
        changed = True
        while changed:
            changed = False
            
            # Simplify clauses:
            # - If a clause has a literal that is True in assignment, remove the clause.
            # - If a clause has a literal that is False in assignment, remove that literal.
            # - If a clause becomes empty, contradiction.
            # - If a clause becomes unit, assign the remaining literal.
            
            new_clauses = []
            unit_clauses = []
            
            for clause in clauses:
                simplified_clause = []
                clause_satisfied = False
                
                for lit in clause:
                    var = abs(lit) - 1
                    is_pos = lit > 0
                    val = assignment[var]
                    
                    if val is not None:
                        if val == is_pos:
                            clause_satisfied = True
                            break
                        else:
                            # Literal is false, skip it
                            continue
                    else:
                        simplified_clause.append(lit)
                
                if not clause_satisfied:
                    if len(simplified_clause) == 0:
                        # Contradiction
                        return False
                    elif len(simplified_clause) == 1:
                        unit_clauses.append(simplified_clause[0])
                    else:
                        new_clauses.append(simplified_clause)
            
            clauses = new_clauses
            
            # Process unit clauses
            for lit in unit_clauses:
                var = abs(lit) - 1
                is_pos = lit > 0
                if assignment[var] is None:
                    assignment[var] = is_pos
                    changed = True
                elif assignment[var] != is_pos:
                    # Contradiction
                    return False
            
            # If we made changes, loop again to propagate further
            # Note: We don't need to re-check pure literals inside the loop necessarily,
            # but unit propagation can create new unit clauses.
        
        # If no clauses left, satisfiable
        if not clauses:
            return True
        
        # Pure Literal Elimination
        # Find variables that appear in only one polarity among unassigned vars.
        var_polarities = defaultdict(set)
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    is_pos = lit > 0
                    var_polarities[var].add(is_pos)
        
        pure_vars = {}
        for var, polarities in var_polarities.items():
            if len(polarities) == 1:
                pure_vars[var] = list(polarities)[0]
        
        if pure_vars:
            # Assign pure literals
            for var, is_pos in pure_vars.items():
                assignment[var] = is_pos
            
            # Simplify clauses again
            new_clauses = []
            for clause in clauses:
                simplified_clause = []
                clause_satisfied = False
                for lit in clause:
                    var = abs(lit) - 1
                    is_pos = lit > 0
                    val = assignment[var]
                    if val is not None:
                        if val == is_pos:
                            clause_satisfied = True
                            break
                        else:
                            continue
                    else:
                        simplified_clause.append(lit)
                
                if not clause_satisfied:
                    if len(simplified_clause) == 0:
                        return False
                    new_clauses.append(simplified_clause)
            
            clauses = new_clauses
            
            # After pure literal elimination, we might have new unit clauses.
            # So we should loop back to unit propagation.
            # To keep it simple, let's just call dpll recursively? No, that's inefficient.
            # Let's just re-run the unit propagation loop.
            # Actually, the easiest way is to just return True if no clauses, else branch.
            # But we should propagate units again.
            # Let's restart the while loop? No, we can just call the unit propagation logic again.
            # For efficiency, I'll just let the next recursion handle it if we branch,
            # but if we don't branch, we need to be sure.
            # Actually, pure literal elimination can create unit clauses.
            # So we should go back to the top of the while loop.
            # To do this cleanly, I'll wrap the whole thing in a function that does UP.
            
            # Let's refactor: 
            # The current structure is a bit messy. 
            # Let's implement a helper `simplify` that returns (clauses, assignment, contradiction).
            # And `unit_propagate` that uses it.
            
            pass # Placeholder for logic restructure if needed. 
            # For now, let's assume UP is done. If pure literals created units, 
            # we might miss them. 
            # To be safe, let's just re-run the UP loop.
            
            changed = True
            while changed:
                changed = False
                new_clauses = []
                unit_clauses = []
                for clause in clauses:
                    simplified_clause = []
                    clause_satisfied = False
                    for lit in clause:
                        var = abs(lit) - 1
                        is_pos = lit > 0
                        val = assignment[var]
                        if val is not None:
                            if val == is_pos:
                                clause_satisfied = True
                                break
                            else:
                                continue
                        else:
                            simplified_clause.append(lit)
                    if not clause_satisfied:
                        if len(simplified_clause) == 0:
                            return False
                        elif len(simplified_clause) == 1:
                            unit_clauses.append(simplified_clause[0])
                        else:
                            new_clauses.append(simplified_clause)
                clauses = new_clauses
                for lit in unit_clauses:
                    var = abs(lit) - 1
                    is_pos = lit > 0
                    if assignment[var] is None:
                        assignment[var] = is_pos
                        changed = True
                    elif assignment[var] != is_pos:
                        return False
        
        if not clauses:
            return True
        
        # Branching Heuristic: Most Constrained Variable (appears in most clauses)
        var_counts = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    var_counts[var] += 1
        
        if not var_counts:
            return True
        
        # Pick variable with max count
        branch_var = max(var_counts, key=lambda v: var_counts[v])
        
        # Try True
        assignment[branch_var] = True
        if dpll(assignment, clauses):
            return True
        
        # Try False
        assignment[branch_var] = False
        if dpll(assignment, clauses):
            return True
        
        # Backtrack
        assignment[branch_var] = None
        return False
    
    # Initialize
    assignment = [None] * n
    # clauses are already in the correct format from instance["clauses"]
    # But we should make a copy to avoid modifying the original if needed (though solve is stateless per call)
    clauses_copy = [list(c) for c in clauses]
    
    satisfied = dpll(assignment, clauses_copy)
    
    if not satisfied:
        # Fallback (should not happen for planted SAT)
        import random
        rng = random.Random(0)
        assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Ensure all variables are assigned
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = True
    
    return assignment
# EVOLVE-BLOCK-END
