# EVOLVE-BLOCK-START
import sys
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to 0-based, internal representation: list of lists of ints
    # Positive literal: var+1, Negative literal: -(var+1) is standard, but let's use 0-based indices
    # Let's stick to 1-based signed literals for the solver logic to match standard DPLL implementations easily, 
    # but convert input to that format.
    # Input: signed 1-based literals.
    # Internal: list of lists of ints (1-based signed).
    
    # Preprocess: Remove duplicate literals in a clause, remove tautologies (x and -x in same clause)
    processed_clauses = []
    for c in clauses:
        # Convert to set to remove duplicates
        lit_set = set(c)
        # Check for tautology
        is_tautology = False
        for lit in lit_set:
            if -lit in lit_set:
                is_tautology = True
                break
        if not is_tautology:
            processed_clauses.append(list(lit_set))
            
    # Remove empty clauses? No, empty clause means UNSAT.
    # If any clause is empty, return None (or arbitrary assignment, but task implies satisfiable)
    for c in processed_clauses:
        if len(c) == 0:
            # Should not happen for satisfiable instances, but handle gracefully
            return [False] * n

    # DPLL Algorithm
    # State: assignment dict {var: bool}, clauses list
    # We will use a recursive function.
    
    # To avoid recursion limit issues (though 50 vars is small), we can use recursion.
    # Python recursion limit is usually 1000, which is fine for depth 50.
    
    # Optimization: Use a list of clauses.
    # Unit propagation:
    def unit_propagate(assignment, clauses):
        # Returns (success, new_assignment, new_clauses)
        # If UNSAT found, success=False
        # assignment: dict var -> bool
        # clauses: list of lists of ints (1-based signed)
        
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                # Simplify clause c with current assignment
                remaining = []
                unit_lit = None
                for lit in c:
                    var = abs(lit)
                    if var in assignment:
                        is_sat = (assignment[var] == (lit > 0))
                        if is_sat:
                            # Clause is satisfied, ignore it
                            remaining = None
                            break
                        else:
                            # Literal is false, remove it
                            continue
                    else:
                        remaining.append(lit)
                
                if remaining is None:
                    continue # Clause satisfied, discard
                
                if len(remaining) == 0:
                    return False, assignment, [] # UNSAT
                elif len(remaining) == 1:
                    unit_lit = remaining[0]
                    # Assign unit_lit
                    var = abs(unit_lit)
                    val = unit_lit > 0
                    assignment[var] = val
                    changed = True
                    # The clause becomes satisfied (or empty if it was just that lit? No, if len 1, assigning it satisfies it)
                    # Wait, if remaining is [lit], and we assign lit to True, the clause is satisfied.
                    # So we don't add it to new_clauses.
                else:
                    new_clauses.append(remaining)
            
            clauses = new_clauses
            
            # Pure Literal Elimination
            # Find variables that appear with only one sign in 'clauses'
            # And are not yet assigned.
            var_polarity = {} # var -> {polarity}
            for c in clauses:
                for lit in c:
                    var = abs(lit)
                    pol = lit > 0
                    if var not in var_polarity:
                        var_polarity[var] = set()
                    var_polarity[var].add(pol)
            
            pure_vars = []
            for var, pols in var_polarity.items():
                if var not in assignment:
                    if len(pols) == 1:
                        pure_vars.append((var, list(pols)[0]))
            
            if pure_vars:
                for var, val in pure_vars:
                    assignment[var] = val
                # Re-run propagation loop to simplify clauses
                changed = True
                # We need to simplify clauses again because we assigned new vars
                # The loop above will handle this in the next iteration if we set changed=True
                # But we need to ensure we re-process clauses.
                # Actually, the `while changed` loop will run again.
                # But we just added assignments. We need to simplify `clauses` with these new assignments.
                # The code above simplifies `clauses` into `new_clauses` based on `assignment`.
                # Since we updated `assignment`, the next iteration of `while changed` will simplify `clauses` (which is currently `new_clauses` from previous step? No, `clauses` is updated at end of while body? No, `clauses = new_clauses` is inside the for loop? No, it's after the for loop.
                # Wait, the structure is:
                # while changed:
                #   changed = False
                #   new_clauses = []
                #   for c in clauses: ...
                #   clauses = new_clauses
                #   # Pure literal check
                #   if pure_vars:
                #      update assignment
                #      changed = True
                # This looks correct. The next iteration will simplify the new clauses with the new assignments.
            
            if not changed:
                break
        
        return True, assignment, clauses

    def dpll(assignment, clauses):
        # Check for UNSAT
        if not clauses:
            return assignment
        
        # Unit Propagation
        success, assignment, clauses = unit_propagate(assignment, clauses)
        if not success:
            return None
        
        if not clauses:
            return assignment
        
        # Check if all assigned
        if len(assignment) == n:
            # Verify solution
            for c in clauses:
                sat = False
                for lit in c:
                    var = abs(lit)
                    if var in assignment and assignment[var] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    return None
            return assignment
        
        # Select variable
        # Heuristic: Variable appearing in most clauses (or with highest degree)
        # Count occurrences in remaining clauses
        var_count = {}
        for c in clauses:
            for lit in c:
                var = abs(lit)
                if var not in assignment:
                    var_count[var] = var_count.get(var, 0) + 1
        
        if not var_count:
            # All variables assigned?
            return assignment
        
        # Pick var with max count
        best_var = max(var_count, key=var_count.get)
        
        # Branch: True then False (or vice versa)
        # Try True
        new_assignment = assignment.copy()
        new_assignment[best_var] = True
        sol = dpll(new_assignment, clauses)
        if sol is not None:
            return sol
        
        # Try False
        new_assignment = assignment.copy()
        new_assignment[best_var] = False
        sol = dpll(new_assignment, clauses)
        if sol is not None:
            return sol
        
        return None

    # Run DPLL
    # Set recursion limit just in case, though n=50 is small
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    # Timeout guard? The evaluator has 300s. DPLL on 50 vars should be fast if satisfiable.
    # If it's hard, it might take long. But "planted" usually means easy for DPLL or random.
    
    solution = dpll({}, processed_clauses)
    
    if solution is None:
        # If DPLL fails (shouldn't for satisfiable, unless timeout or bug), 
        # we might need a fallback. But for this task, we assume satisfiable.
        # Return a random assignment as a last resort?
        # Or just return all False?
        # The score is fraction solved. If we return a non-satisfying assignment, it's 0 for that instance.
        # Better to return a random one? No, just return a default.
        return [False] * n
    
    # Convert dict solution to list of booleans
    result = [False] * n
    for i in range(1, n + 1):
        if i in solution:
            result[i-1] = solution[i]
        else:
            # Should not happen if DPLL completed
            result[i-1] = False
            
    return result

# EVOLVE-BLOCK-END
