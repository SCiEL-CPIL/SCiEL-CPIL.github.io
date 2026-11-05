"""Synthetic planted 3-SAT solver (50 variables) using DPLL with Unit Propagation and Pure Literal Elimination."""
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290
    
    # DPLL Solver
    # We represent the current assignment as a list of None (unassigned), True, or False
    # We need to track which variables are assigned to efficiently propagate.
    
    # Optimization: Pre-process clauses to remove duplicates and check for immediate contradictions
    # But for n=50, raw DPLL is fast enough.
    
    def dpll(clauses, assignment):
        # Check for deadline
        if time.time() > deadline:
            return None
        
        # 1. Check for empty clause (contradiction)
        # 2. Check if all clauses are satisfied
        # 3. Unit Propagation
        # 4. Pure Literal Elimination
        # 5. Branching
        
        # We will work on a copy of the clauses to avoid modifying the original
        # Actually, for efficiency, we can filter clauses based on assignment.
        
        # Filter clauses based on current assignment
        active_clauses = []
        for c in clauses:
            clause_state = 0 # 0: all unassigned, 1: some assigned but not sat, 2: satisfied
            for l in c:
                var = abs(l) - 1
                if assignment[var] is not None:
                    if assignment[var] == (l > 0):
                        clause_state = 2
                        break
                    else:
                        clause_state = 1 # This literal is false, but others might be true
                # If unassigned, do nothing
            if clause_state == 2:
                continue # Satisfied, remove from active list
            else:
                # Remove literals that are assigned to False
                new_c = [l for l in c if assignment[abs(l)-1] is None or assignment[abs(l)-1] == (l > 0)]
                # Wait, if assignment[var] is True and l is negative, literal is False.
                # If assignment[var] is False and l is positive, literal is False.
                # So we keep literals that are either unassigned or currently True.
                
                # Let's re-evaluate:
                # A literal l is satisfied if assignment[var] is True and l>0, or assignment[var] is False and l<0.
                # A literal l is falsified if assignment[var] is False and l>0, or assignment[var] is True and l<0.
                
                filtered_c = []
                sat = False
                for l in c:
                    var = abs(l) - 1
                    val = assignment[var]
                    if val is None:
                        filtered_c.append(l)
                    else:
                        if val == (l > 0):
                            sat = True
                            break
                        # else: literal is false, discard it
                if not sat:
                    active_clauses.append(filtered_c)
        
        # Check for contradiction (empty clause)
        if any(len(c) == 0 for c in active_clauses):
            return None
            
        # Check for success (no active clauses)
        if not active_clauses:
            return assignment[:]
            
        # Unit Propagation
        changed = True
        while changed:
            changed = False
            new_active_clauses = []
            for c in active_clauses:
                if len(c) == 1:
                    l = c[0]
                    var = abs(l) - 1
                    if assignment[var] is None:
                        assignment[var] = (l > 0)
                        changed = True
                    # If assignment[var] was already set, it should have been handled in filtering
                    # But if it was set to the opposite, we would have had a contradiction earlier.
                    # If it was set to the same, the clause would have been satisfied and removed.
                    # So if len(c)==1, the var is unassigned.
                else:
                    new_active_clauses.append(c)
            active_clauses = new_active_clauses
            
            # Re-filter after assignment change
            # This is expensive. Better to just loop again from the top of dpll?
            # Or simply, if we assigned a var, we need to update all clauses.
            # For n=50, re-filtering is fine.
            if changed:
                # Recursive call or loop?
                # Let's just return dpll(clauses, assignment) to restart the filtering/propagation
                return dpll(clauses, assignment)
                
        # Pure Literal Elimination
        # Find variables that appear only with one polarity in active_clauses
        # Count occurrences
        pos_count = [0] * n
        neg_count = [0] * n
        for c in active_clauses:
            for l in c:
                var = abs(l) - 1
                if l > 0:
                    pos_count[var] += 1
                else:
                    neg_count[var] += 1
                    
        pure_vars = []
        for var in range(n):
            if assignment[var] is None:
                if pos_count[var] > 0 and neg_count[var] == 0:
                    pure_vars.append((var, True))
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    pure_vars.append((var, False))
                    
        if pure_vars:
            for var, val in pure_vars:
                assignment[var] = val
            # Restart propagation
            return dpll(clauses, assignment)
            
        # Branching
        # Choose variable with highest degree in active_clauses
        degree = [0] * n
        for c in active_clauses:
            for l in c:
                var = abs(l) - 1
                degree[var] += 1
                
        # Find unassigned var with max degree
        best_var = -1
        max_deg = -1
        for var in range(n):
            if assignment[var] is None and degree[var] > max_deg:
                max_deg = degree[var]
                best_var = var
                
        if best_var == -1:
            # All assigned, but no active clauses? Then we are done.
            # This case is handled by "if not active_clauses" above.
            # If we get here, it means we have active clauses but no unassigned vars?
            # That implies a contradiction (empty clause) which should have been caught.
            return None
            
        # Branch: Try True first
        assignment[best_var] = True
        result = dpll(clauses, assignment)
        if result is not None:
            return result
            
        # Backtrack: Try False
        assignment[best_var] = False
        result = dpll(clauses, assignment)
        if result is not None:
            return result
            
        # Backtrack: Unassign
        assignment[best_var] = None
        return None
        
    # Initialize assignment with None
    assignment = [None] * n
    result = dpll(clauses, assignment)
    
    if result is not None:
        # Ensure all variables are assigned
        # DPLL should have assigned all, but just in case
        for i in range(n):
            if result[i] is None:
                result[i] = False # Arbitrary, but shouldn't happen if SAT
        return result
    else:
        # Fallback: Random assignment (should not happen for satisfiable instances)
        # Or return an empty list? No, must return list of booleans.
        # If DPLL fails, it's likely UNSAT or timeout.
        # Since task says "satisfiable", DPLL failure is unexpected.
        # Return a random assignment as a last resort to avoid crash, though score will be 0 if invalid.
        import random
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
