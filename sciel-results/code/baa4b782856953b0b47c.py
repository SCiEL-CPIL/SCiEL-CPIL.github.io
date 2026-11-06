"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert clauses to a list of lists of integers (signed literals)
    # We will use a recursive DPLL approach.
    # To avoid deep recursion issues in Python for large n (though 50 is small),
    # we use a standard recursive DPLL.
    
    # Pre-compute clause sets for faster lookup? 
    # For n=50, simple list operations are fine.
    
    def dpll(clauses, assignment):
        # assignment is a dict: {var_index: bool}
        # clauses is a list of lists of literals
        
        # 1. Check for empty clause (contradiction)
        for clause in clauses:
            if not clause:
                return None
            # Check for unit clause
            if len(clause) == 1:
                lit = clause[0]
                var = abs(lit) - 1
                val = lit > 0
                # If variable already assigned, check consistency
                if var in assignment:
                    if assignment[var] != val:
                        return None
                else:
                    # Unit propagation step (handled by recursion)
                    new_assignment = dict(assignment)
                    new_assignment[var] = val
                    # Remove all occurrences of this literal and its negation from all clauses
                    new_clauses = []
                    for c in clauses:
                        new_c = []
                        for l in c:
                            if abs(l) - 1 == var:
                                # If l matches the assignment, clause is satisfied, skip
                                if l == lit:
                                    continue # Clause satisfied, effectively empty in new context
                                else:
                                    # l is negation, remove it
                                    continue
                            else:
                                new_c.append(l)
                        if not new_c:
                            # Clause satisfied by this assignment
                            continue
                        new_clauses.append(new_c)
                    
                    # Recurse
                    result = dpll(new_clauses, new_assignment)
                    if result is not None:
                        return result
                    return None
        
        # 2. Check for pure literals
        # A variable is pure if it appears only with one sign in the remaining clauses.
        # If pure, assign it to satisfy those clauses.
        # We can do this iteratively or just pick one.
        # For efficiency, we'll identify one pure literal if any exists.
        var_occurrences = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if var not in var_occurrences:
                    var_occurrences[var] = set()
                var_occurrences[var].add(lit)
        
        for var, lits in var_occurrences.items():
            if var in assignment:
                continue
            # Check if all occurrences have the same sign
            if len(lits) == 1:
                lit = lits.pop()
                val = lit > 0
                new_assignment = dict(assignment)
                new_assignment[var] = val
                
                # Filter clauses
                new_clauses = []
                for c in clauses:
                    new_c = []
                    satisfied = False
                    for l in c:
                        if abs(l) - 1 == var:
                            if l == lit:
                                satisfied = True
                                break
                            else:
                                continue
                        else:
                            new_c.append(l)
                    if satisfied:
                        continue
                    new_clauses.append(new_c)
                
                result = dpll(new_clauses, new_assignment)
                if result is not None:
                    return result
                # If backtracking needed, we would try the other value, but for pure literals
                # in SAT instances, usually one branch works. However, strictly speaking,
                # we should try both if the first fails. But pure literal elimination is a sound
                # inference rule: if a variable is pure, there is a model with that variable set
                # to satisfy all its occurrences. So we don't need to backtrack on the *choice*
                # of value, only if the rest of the problem becomes unsat.
                # Wait, if the rest becomes unsat, then the original problem is unsat?
                # Yes, because any solution must satisfy the pure literal clauses, so the variable
                # MUST be set to that value. If that leads to contradiction, no solution exists.
                return None
        
        # 3. Variable selection heuristic
        # Pick a variable that appears in the most clauses (or most unassigned clauses).
        # Let's pick a variable from the first clause for simplicity, or use a heuristic.
        # Heuristic: Most frequent variable.
        var_counts = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if var not in assignment:
                    var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            # All variables assigned?
            # Check if all clauses are satisfied
            for clause in clauses:
                sat = False
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment.get(var) == (lit > 0):
                        sat = True
                        break
                if not sat:
                    return None
            return assignment
        
        var = max(var_counts, key=lambda v: var_counts[v])
        
        # Branch on True
        new_assignment_true = dict(assignment)
        new_assignment_true[var] = True
        new_clauses_true = []
        for c in clauses:
            new_c = []
            sat = False
            for l in c:
                if abs(l) - 1 == var:
                    if l > 0:
                        sat = True
                        break
                    else:
                        continue
                else:
                    new_c.append(l)
            if sat:
                continue
            new_clauses_true.append(new_c)
        
        result = dpll(new_clauses_true, new_assignment_true)
        if result is not None:
            return result
        
        # Branch on False
        new_assignment_false = dict(assignment)
        new_assignment_false[var] = False
        new_clauses_false = []
        for c in clauses:
            new_c = []
            sat = False
            for l in c:
                if abs(l) - 1 == var:
                    if l < 0:
                        sat = True
                        break
                    else:
                        continue
                else:
                    new_c.append(l)
            if sat:
                continue
            new_clauses_false.append(new_c)
        
        result = dpll(new_clauses_false, new_assignment_false)
        if result is not None:
            return result
        
        return None

    # Run DPLL
    solution = dpll(clauses, {})
    
    if solution is None:
        # Should not happen for satisfiable instances, but handle gracefully
        # Return random assignment as fallback
        import random
        return [random.random() < 0.5 for _ in range(n)]
    
    # Convert dict to list of booleans
    assignment_list = [solution.get(i, False) for i in range(n)]
    
    # Verify solution (optional safety check)
    for clause in clauses:
        sat = False
        for lit in clause:
            var = abs(lit) - 1
            if assignment_list[var] == (lit > 0):
                sat = True
                break
        if not sat:
            # This should not happen if DPLL is correct
            break
            
    return assignment_list
# EVOLVE-BLOCK-END
