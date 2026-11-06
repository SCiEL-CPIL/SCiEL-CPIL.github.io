import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0  # Leave some buffer
    
    # 1. Try DPLL
    result = dpll_solve(n, clauses, deadline)
    if result is not None:
        return result
    
    # 2. Fallback to WalkSAT
    return walksat_solve(n, clauses, deadline)

def dpll_solve(n, clauses, deadline):
    # Convert clauses to sets of literals for faster lookup? 
    # Or keep as lists. 
    # DPLL state: assignment dict {var: bool}, remaining clauses
    
    # Optimization: Pre-process clauses to remove tautologies? (Not needed for 3-SAT planted)
    
    def unit_propagate(assignment, clauses):
        # Returns (new_assignment, new_clauses) or (None, None) if conflict
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Check if clause is satisfied
                sat = False
                # Check if clause has unassigned literals
                unassigned_lits = []
                for lit in clause:
                    var = abs(lit) - 1
                    if var in assignment:
                        if assignment[var] == (lit > 0):
                            sat = True
                            break
                    else:
                        unassigned_lits.append(lit)
                
                if sat:
                    continue # Clause satisfied, drop it
                
                if len(unassigned_lits) == 0:
                    # Conflict
                    return None, None
                
                if len(unassigned_lits) == 1:
                    # Unit clause
                    lit = unassigned_lits[0]
                    var = abs(lit) - 1
                    val = lit > 0
                    if var in assignment:
                        if assignment[var] != val:
                            return None, None
                    else:
                        assignment[var] = val
                        changed = True
                        # Clause is now satisfied, drop it
                        continue
                
                new_clauses.append(unassigned_lits)
        
        return assignment, new_clauses

    def pure_literal_eliminate(assignment, clauses):
        # Find variables that appear with only one sign
        # This is O(m*n) which is fine for n=50
        var_signs = {} # var -> set of signs
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                if var not in var_signs:
                    var_signs[var] = set()
                var_signs[var].add(sign)
        
        changed = False
        new_clauses = []
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if var in assignment:
                    if assignment[var] == (lit > 0):
                        sat = True
                        break
            
            if sat:
                continue
            
            # Check if any literal in this clause is a pure literal
            # If so, we assign it and drop the clause
            # Actually, pure literal elimination assigns the variable.
            # We need to be careful: we assign ALL pure literals.
            
            # Let's just collect pure literals first
            pass
        
        # Better approach:
        pure_vars = {}
        for var, signs in var_signs.items():
            if len(signs) == 1:
                sign = list(signs)[0]
                pure_vars[var] = (sign > 0)
        
        if pure_vars:
            changed = True
            # Assign pure literals
            for var, val in pure_vars.items():
                if var not in assignment:
                    assignment[var] = val
            
            # Re-filter clauses
            new_clauses = []
            for clause in clauses:
                sat = False
                for lit in clause:
                    var = abs(lit) - 1
                    if var in assignment:
                        if assignment[var] == (lit > 0):
                            sat = True
                            break
                if not sat:
                    # Keep only unassigned literals
                    new_clause = []
                    for lit in clause:
                        var = abs(lit) - 1
                        if var not in assignment:
                            new_clause.append(lit)
                    if new_clause:
                        new_clauses.append(new_clause)
            return assignment, new_clauses, True
        
        return assignment, clauses, False

    def dpll(assignment, clauses):
        if time.time() > deadline:
            return None
        
        # Check if all variables assigned
        if len(assignment) == n:
            # Verify
            for clause in original_clauses: # Need original clauses for verification?
                # Actually, if we maintained clauses correctly, empty list means SAT
                pass
            # If we reach here with no conflicts, it's SAT
            # But we need to return the assignment
            return dict(assignment)
        
        # Unit Propagation
        assignment, clauses = unit_propagate(assignment, clauses)
        if clauses is None:
            return None
        
        # Pure Literal Elimination
        assignment, clauses, changed = pure_literal_eliminate(assignment, clauses)
        
        # If all clauses satisfied (empty list), SAT
        if not clauses:
            # Fill remaining vars arbitrarily
            for i in range(n):
                if i not in assignment:
                    assignment[i] = False
            return dict(assignment)
        
        # Branching
        # Choose variable with max occurrences
        var_count = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                var_count[var] = var_count.get(var, 0) + 1
        
        if not var_count:
            # No unassigned vars in clauses? Should have been caught by "not clauses"
            for i in range(n):
                if i not in assignment:
                    assignment[i] = False
            return dict(assignment)
        
        # Pick var with max count
        best_var = max(var_count, key=var_count.get)
        
        # Try True
        assignment[best_var] = True
        result = dpll(assignment, clauses)
        if result is not None:
            return result
        
        # Try False
        assignment[best_var] = False
        result = dpll(assignment, clauses)
        if result is not None:
            return result
        
        # Backtrack
        del assignment[best_var]
        return None

    # Initialize
    initial_assignment = {}
    initial_clauses = [list(c) for c in clauses]
    
    # Run DPLL
    result = dpll(initial_assignment, initial_clauses)
    
    if result:
        # Convert dict to list
        return [result.get(i, False) for i in range(n)]
    
    return None

def walksat_solve(n, clauses, deadline):
    # Optimized WalkSAT with Min-Conflicts
    best_assignment = None
    best_unsat = float('inf')
    
    # Precompute var_clauses
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            var_clauses[vi].append(ci)
    
    def count_unsat(assignment):
        count = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                idx = abs(lit) - 1
                if assignment[idx] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    for attempt in range(1000):
        if time.time() >= deadline:
            break
            
        rng = random.Random(attempt * 12345 + 999)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
            
        # WalkSAT loop
        for step in range(10000):
            if time.time() >= deadline:
                break
            
            # Find unsat clauses
            unsat_clauses = []
            for ci, clause in enumerate(clauses):
                sat = False
                for lit in clause:
                    idx = abs(lit) - 1
                    if assignment[idx] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            # Pick random unsat clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = clauses[ci]
            
            # Min-conflicts: pick var in clause that minimizes delta
            best_var = -1
            best_delta = float('inf')
            
            for lit in clause:
                var_idx = abs(lit) - 1
                # Calculate delta if we flip var_idx
                # Delta = (clauses becoming unsat) - (clauses becoming sat)
                # Since we are flipping a var in an unsat clause, it will become sat (delta -1 for that clause)
                # But other clauses might break.
                
                delta = 0
                for other_ci in var_clauses[var_idx]:
                    other_clause = clauses[other_ci]
                    # Check if other_clause is currently sat
                    other_sat = False
                    for olit in other_clause:
                        oidx = abs(olit) - 1
                        if assignment[oidx] == (olit > 0):
                            other_sat = True
                            break
                    
                    if other_sat:
                        # Check if it remains sat after flip
                        # It remains sat if it has another lit that is true, OR if the lit in this clause is not the only one making it true
                        # Actually, if it's currently sat, and we flip var_idx, it becomes unsat ONLY IF
                        # the lit in this clause was the ONLY one making it true.
                        
                        # Find the lit in other_clause corresponding to var_idx
                        # There is exactly one because it's 3-SAT and var appears once per clause? 
                        # In 3-SAT, a variable can appear at most once in a clause? No, usually distinct literals.
                        # Assume standard 3-CNF: distinct literals in a clause.
                        
                        # Let's check if other_clause has another true literal
                        has_other_true = False
                        for olit in other_clause:
                            oidx = abs(olit) - 1
                            if oidx == var_idx:
                                continue
                            if assignment[oidx] == (olit > 0):
                                has_other_true = True
                                break
                        
                        if not has_other_true:
                            delta += 1 # Becomes unsat
                    else:
                        # Currently unsat. Does it become sat?
                        # It becomes sat if the lit corresponding to var_idx becomes true.
                        # Since we are flipping var_idx, we need to check the sign.
                        # If the lit in other_clause matches the new value, it becomes sat.
                        # But we don't know the new value yet? 
                        # We are evaluating the flip.
                        # Let's determine the new value of var_idx.
                        # We are considering flipping var_idx.
                        # Current val = assignment[var_idx]. New val = not assignment[var_idx].
                        # The lit in other_clause is either var_idx or -var_idx.
                        
                        # Find the lit
                        lit_in_other = None
                        for olit in other_clause:
                            if abs(olit) - 1 == var_idx:
                                lit_in_other = olit
                                break
                        
                        if lit_in_other is not None:
                            new_val = not assignment[var_idx]
                            if new_val == (lit_in_other > 0):
                                delta -= 1 # Becomes sat
                            
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
                elif delta == best_delta and rng.random() < 0.5:
                    best_var = var_idx
            
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                # Update unsat count? 
                # For simplicity, we re-count every step or every few steps.
                # Re-counting is O(m*3). m is ~ 50*10=500. 1500 ops. 10000 steps = 15M ops. Fine.
                
                new_unsat = count_unsat(assignment)
                if new_unsat == 0:
                    return assignment
                if new_unsat < best_unsat:
                    best_unsat = new_unsat
                    best_assignment = assignment[:]
    
    if best_assignment:
        return best_assignment
    return [False] * n
