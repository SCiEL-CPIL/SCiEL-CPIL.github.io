"""
DPLL-based 3-SAT solver with unit propagation and pure literal elimination.
Structural change from local search to systematic search with constraint propagation.
"""
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: allow 290 seconds to be safe within the 300s limit
    deadline = time.time() + 290.0
    
    if n == 0:
        return []
        
    # State:
    # assignment: list of n elements, -1 (False), 0 (Unassigned), 1 (True)
    # trail: list of variables assigned in order, for backtracking
    
    assignment = [0] * n
    trail = []
    
    # We will use a recursive approach for clarity, but with a step counter/time check.
    # To ensure we don't exceed time, we check time at the start of each recursive call.
    
    # Helper to check if a clause is satisfied
    def is_sat(clause):
        for l in clause:
            if l > 0:
                if assignment[l-1] == 1:
                    return True
            else:
                if assignment[-l-1] == -1:
                    return True
        return False
        
    # Helper to check if a clause is falsified
    def is_falsified(clause):
        for l in clause:
            if l > 0:
                if assignment[l-1] == -1:
                    return False
            else:
                if assignment[-l-1] == 1:
                    return False
        return True
        
    # Helper to get unassigned literals in a clause
    def get_unassigned(clause):
        unassigned = []
        for l in clause:
            if l > 0:
                if assignment[l-1] == 0:
                    unassigned.append(l)
            else:
                if assignment[-l-1] == 0:
                    unassigned.append(l)
        return unassigned

    def dpll(depth):
        if time.time() > deadline:
            return False # Time out
            
        # 1. Check for satisfied/falsified clauses and collect units
        units = {}
        active_clauses = []
        
        for clause in clauses:
            if is_sat(clause):
                continue
                
            unassigned = get_unassigned(clause)
            
            if not unassigned:
                # Clause is falsified
                return False
                
            if len(unassigned) == 1:
                # Unit clause
                l = unassigned[0]
                val = 1 if l > 0 else -1
                var = abs(l)
                
                # Check for conflict
                if assignment[var-1] != 0:
                    if assignment[var-1] != val:
                        return False
                else:
                    if var in units:
                        if units[var] != val:
                            return False
                    units[var] = val
            else:
                active_clauses.append(clause)
                
        # Apply units
        for var, val in units.items():
            assignment[var-1] = val
            trail.append(var)
            
        # If no active clauses, we are done
        if not active_clauses:
            return True
            
        # 2. Pure Literal Elimination
        # Find variables that appear only with one sign in active_clauses
        pos_vars = set()
        neg_vars = set()
        
        for clause in active_clauses:
            for l in clause:
                if l > 0:
                    pos_vars.add(l)
                else:
                    neg_vars.add(-l)
                    
        pure_vars = {}
        for v in pos_vars:
            if v not in neg_vars and assignment[v-1] == 0:
                pure_vars[v] = 1
        for v in neg_vars:
            if v not in pos_vars and assignment[v-1] == 0:
                if v not in pure_vars:
                    pure_vars[v] = -1
        
        # Apply pure literals
        for var, val in pure_vars.items():
            assignment[var-1] = val
            trail.append(var)
            
        # Filter active_clauses again after pure literals
        new_active_clauses = []
        for clause in active_clauses:
            if is_sat(clause):
                continue
            unassigned = get_unassigned(clause)
            if not unassigned:
                return False
            if len(unassigned) > 1:
                new_active_clauses.append(clause)
            else:
                # New unit created by pure literal assignment
                # We need to handle this. 
                # Simplest: restart the unit propagation phase.
                # To avoid complex loops, we'll just let the recursive call handle it?
                # No, we are in the middle of the function.
                # Let's just add it to a list and if it's not empty, we should probably 
                # not proceed to variable selection but re-run propagation.
                # For simplicity and correctness, let's just return False if we create new units?
                # No, that's wrong.
                # Let's just pick a variable from the unit clause? 
                # Actually, if we have units, we should have caught them in the first pass.
                # Pure literals can create units.
                # Let's just pick one of the variables from the unit clause as the decision variable?
                # No, that breaks the logic.
                
                # Better approach: If new units are created, we should loop back to step 1.
                # Since we are in a function, we can't easily loop.
                # Let's just proceed to variable selection. If we pick a variable, 
                # the next recursive call will do unit propagation again.
                # This is less efficient but correct.
                new_active_clauses.append(clause)
                
        if not new_active_clauses:
            return True
            
        # 3. Variable Selection
        # Pick variable with highest frequency in new_active_clauses
        var_freq = {}
        for clause in new_active_clauses:
            for l in clause:
                v = abs(l)
                if assignment[v-1] == 0:
                    var_freq[v] = var_freq.get(v, 0) + 1
                    
        if not var_freq:
            return True # All satisfied
            
        best_var = max(var_freq, key=var_freq.get)
        
        # Branch 1: True
        assignment[best_var-1] = 1
        trail.append(best_var)
        
        if dpll(depth + 1):
            return True
            
        # Backtrack
        trail.pop()
        assignment[best_var-1] = 0
        
        # Branch 2: False
        assignment[best_var-1] = -1
        trail.append(best_var)
        
        if dpll(depth + 1):
            return True
            
        # Backtrack
        trail.pop()
        assignment[best_var-1] = 0
        
        return False

    # Start DPLL
    if dpll(0):
        # Convert assignment to list of booleans
        res = [False] * n
        for i in range(n):
            res[i] = (assignment[i] == 1)
        return res
    else:
        # Failed or timed out
        # Return a random assignment as fallback
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
