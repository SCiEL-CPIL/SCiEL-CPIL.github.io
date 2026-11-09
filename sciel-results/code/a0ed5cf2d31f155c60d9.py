"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocess: convert to 0-based indices
    # Each clause is a list of signed literals (1-based)
    # We'll work with a representation where:
    #   - Variables are 0..n-1
    #   - A literal is (var, is_positive)
    
    # Build clause list with 0-based variables
    clauses_0 = []
    for clause in clauses:
        clause_0 = []
        for lit in clause:
            var = abs(lit) - 1
            is_pos = lit > 0
            clause_0.append((var, is_pos))
        clauses_0.append(clause_0)
    
    num_clauses = len(clauses_0)
    
    # DPLL with unit propagation and pure literal elimination
    # assignment: dict var -> bool (True/False) or None if unassigned
    # We'll use a recursive approach with backtracking
    
    # For efficiency, we'll use a trail-based approach
    # But given the size (50 vars), a simpler recursive approach may suffice
    
    # Let's implement a clean DPLL:
    
    def dpll(assignment, clauses_list):
        """
        assignment: list of length n, each element is True, False, or None
        clauses_list: list of clauses, each clause is a list of (var, is_pos)
        Returns: True if satisfiable, and updates assignment if so
        """
        # Step 1: Check for empty clause (contradiction)
        for clause in clauses_list:
            if len(clause) == 0:
                return False
        
        # Step 2: Unit propagation
        # We need to repeatedly find unit clauses and propagate
        # We'll do this iteratively
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            for i, clause in enumerate(clauses_list):
                if len(clause) == 1:
                    var, is_pos = clause[0]
                    # Assign this variable
                    if assignment[var] is None:
                        assignment[var] = is_pos
                        changed = True
                        # Remove all clauses containing this variable's positive literal
                        # and simplify all clauses by removing the negated literal
                        new_clauses = []
                        for j, c in enumerate(clauses_list):
                            if j == i:
                                continue  # skip the unit clause itself
                            new_c = []
                            for (v, p) in c:
                                if v == var and p == is_pos:
                                    # This clause is satisfied
                                    break
                                elif v == var and not p:
                                    # This literal is false, remove it
                                    continue
                                else:
                                    new_c.append((v, p))
                            else:
                                # No satisfied literal found
                                new_clauses.append(new_c)
                            break
                        else:
                            new_clauses.append(new_c)
                        # Wait, the above logic is buggy. Let me redo.
                        # Actually, let me restart with a cleaner approach.
                        pass
                    elif assignment[var] == is_pos:
                        # Already satisfied, remove this clause
                        clauses_list.pop(i)
                        changed = True
                        break  # restart loop since indices changed
                    else:
                        # Contradiction
                        return False
            # If we found a unit clause and assigned a new variable, we need to simplify
            # This is getting complex. Let me use a different approach.
        
        # Let me restart with a cleaner implementation
        
        return True  # placeholder
    
    # Let me write a cleaner DPLL from scratch
    
    def simplify_clauses(clauses_list, assignment):
        """
        Given a list of clauses and a partial assignment, simplify the clauses.
        Returns (simplified_clauses, contradiction)
        where simplified_clauses has unit clauses removed and satisfied clauses removed.
        """
        new_clauses = []
        for clause in clauses_list:
            clause_simplified = []
            clause_satisfied = False
            for (var, is_pos) in clause:
                if assignment[var] is not None:
                    if assignment[var] == is_pos:
                        clause_satisfied = True
                        break
                    else:
                        # Literal is false, skip it
                        continue
                else:
                    clause_simplified.append((var, is_pos))
            if not clause_satisfied:
                new_clauses.append(clause_simplified)
        return new_clauses
    
    def unit_propagate(clauses_list, assignment):
        """
        Perform unit propagation. Returns (new_clauses, new_assignment, contradiction)
        """
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            for i, clause in enumerate(clauses_list):
                if len(clause) == 1:
                    var, is_pos = clause[0]
                    if assignment[var] is None:
                        assignment[var] = is_pos
                        changed = True
                        # Simplify all clauses
                        clauses_list, contradiction = simplify_and_check(clauses_list, assignment)
                        if contradiction:
                            return clauses_list, assignment, True
                        break
                    elif assignment[var] == is_pos:
                        # Remove this clause
                        clauses_list.pop(i)
                        changed = True
                        break
                    else:
                        # Contradiction
                        return clauses_list, assignment, True
            # After any change, we need to simplify all clauses
            if changed:
                clauses_list, contradiction = simplify_and_check(clauses_list, assignment)
                if contradiction:
                    return clauses_list, assignment, True
        return clauses_list, assignment, False
    
    def simplify_and_check(clauses_list, assignment):
        """Simplify clauses given current assignment, check for empty clause (contradiction)."""
        new_clauses = []
        for clause in clauses_list:
            clause_simplified = []
            clause_satisfied = False
            for (var, is_pos) in clause:
                if assignment[var] is not None:
                    if assignment[var] == is_pos:
                        clause_satisfied = True
                        break
                    else:
                        continue
                else:
                    clause_simplified.append((var, is_pos))
            if not clause_satisfied:
                if len(clause_simplified) == 0:
                    return new_clauses, True  # contradiction
                new_clauses.append(clause_simplified)
        return new_clauses, False
    
    def pure_literal_elimination(clauses_list, assignment):
        """
        Eliminate pure literals. Returns (new_clauses, new_assignment)
        """
        # Find all variables that appear in only one polarity
        var_polarities = defaultdict(set)
        for clause in clauses_list:
            for (var, is_pos) in clause:
                if assignment[var] is None:
                    var_polarities[var].add(is_pos)
        
        pure_vars = {}
        for var, polarities in var_polarities.items():
            if len(polarities) == 1:
                pure_vars[var] = list(polarities)[0]
        
        if not pure_vars:
            return clauses_list, assignment
        
        # Assign pure literals and simplify
        for var, is_pos in pure_vars.items():
            assignment[var] = is_pos
        
        new_clauses, contradiction = simplify_and_check(clauses_list, assignment)
        if contradiction:
            return new_clauses, assignment
        return new_clauses, assignment
    
    def dpll_recursive(clauses_list, assignment, depth=0):
        """
        DPLL recursive solver.
        Returns True if satisfiable.
        """
        if depth > 1000:
            return False
        
        # Check for empty clause
        for clause in clauses_list:
            if len(clause) == 0:
                return False
        
        # If all clauses satisfied, we're done
        # (already handled by simplify)
        
        # Unit propagation
        clauses_list, assignment, contradiction = unit_propagate(clauses_list, assignment)
        if contradiction:
            return False
        
        # Check again for empty clause after propagation
        for clause in clauses_list:
            if len(clause) == 0:
                return False
        
        # If no clauses left, satisfiable
        if not clauses_list:
            return True
        
        # Pure literal elimination
        clauses_list, assignment = pure_literal_elimination(clauses_list, assignment)
        
        # Check again
        for clause in clauses_list:
            if len(clause) == 0:
                return False
        
        if not clauses_list:
            return True
        
        # Pick a variable to branch on
        # Choose variable that appears most frequently (heuristic)
        var_counts = defaultdict(int)
        for clause in clauses_list:
            for (var, is_pos) in clause:
                var_counts[var] += 1
        
        if not var_counts:
            return True
        
        # Pick variable with highest count
        branch_var = max(var_counts, key=lambda v: var_counts[v])
        
        # Try True first
        assignment[branch_var] = True
        new_clauses, contradiction = simplify_and_check(clauses_list, assignment)
        if not contradiction:
            if dpll_recursive(new_clauses, assignment, depth + 1):
                return True
        assignment[branch_var] = False
        
        # Try False
        assignment[branch_var] = False
        new_clauses, contradiction = simplify_and_check(clauses_list, assignment)
        if not contradiction:
            if dpll_recursive(new_clauses, assignment, depth + 1):
                return True
        assignment[branch_var] = None
        
        return False
    
    # Initialize
    assignment = [None] * n
    
    # Run DPLL
    satisfied = dpll_recursive(clauses_0, assignment)
    
    if not satisfied:
        # Should not happen for planted SAT instances, but handle gracefully
        # Fall back to random assignment
        import random
        rng = random.Random(0)
        assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Ensure all variables are assigned
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = True
    
    return assignment
# EVOLVE-BLOCK-END
