import time
import random

def solve(instance):
    n = instance["n"]
    clauses_raw = instance["clauses"]
    
    # Parse literals
    # Format: list of lists of (var, sign)
    clauses = []
    for c in clauses_raw:
        pc = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pc.append((var, sign))
        clauses.append(pc)
        
    start_time = time.time()
    deadline = start_time + 290.0
    
    def simplify(assignment, clauses):
        """
        Applies unit propagation and pure literal elimination.
        Returns (new_assignment, new_clauses, success)
        """
        assignment = dict(assignment)
        # Filter out None clauses and copy remaining
        active_clauses = []
        for c in clauses:
            if c is None:
                continue
            active_clauses.append(list(c))
            
        changed = True
        while changed:
            changed = False
            
            # 1. Unit Propagation and Simplification
            for i in range(len(active_clauses)):
                if not active_clauses[i]:
                    return assignment, active_clauses, False # Conflict: empty clause
                
                new_clause = []
                satisfied = False
                for var, sign in active_clauses[i]:
                    if var in assignment:
                        if assignment[var] == (sign == 1):
                            satisfied = True
                            break
                    else:
                        new_clause.append((var, sign))
                
                if satisfied:
                    active_clauses[i] = None
                    changed = True
                else:
                    active_clauses[i] = new_clause
                    if len(new_clause) == 1:
                        var, sign = new_clause[0]
                        if var in assignment:
                            if assignment[var] != (sign == 1):
                                return assignment, active_clauses, False # Conflict
                        else:
                            assignment[var] = (sign == 1)
                            changed = True
            
            # 2. Pure Literal Elimination
            # Identify variables that appear with only one sign in remaining clauses
            var_signs = {}
            for c in active_clauses:
                if c is None:
                    continue
                for var, sign in c:
                    if var not in assignment:
                        if var not in var_signs:
                            var_signs[var] = set()
                        var_signs[var].add(sign)
            
            for var, signs in var_signs.items():
                if len(signs) == 1:
                    sign = list(signs)[0]
                    assignment[var] = (sign == 1)
                    changed = True
        
        return assignment, active_clauses, True

    def choose_variable(assignment, clauses):
        # Choose variable appearing in most active clauses
        counts = [0] * n
        for c in clauses:
            if c is None:
                continue
            for var, sign in c:
                if var not in assignment:
                    counts[var] += 1
        
        best_var = -1
        max_count = 0
        for var in range(n):
            if var not in assignment and counts[var] > max_count:
                max_count = counts[var]
                best_var = var
                
        return best_var

    def dpll(assignment, clauses):
        # Check time
        if time.time() > deadline:
            return None
            
        new_assignment, new_clauses, success = simplify(assignment, clauses)
        if not success:
            return None
            
        # Check if all variables are assigned
        # If so, and no conflict, it's a solution
        if len(new_assignment) == n:
            return new_assignment
            
        var = choose_variable(new_assignment, new_clauses)
        if var == -1:
            # All assigned but len < n? Impossible if n is correct.
            # Or if some vars don't appear in clauses.
            # If a var doesn't appear in any clause, its value doesn't matter.
            # We can assign it arbitrarily.
            for v in range(n):
                if v not in new_assignment:
                    new_assignment[v] = False
            return new_assignment
            
        # Branch: Try True
        sol = dpll({**new_assignment, var: True}, new_clauses)
        if sol is not None:
            return sol
            
        # Branch: Try False
        sol = dpll({**new_assignment, var: False}, new_clauses)
        if sol is not None:
            return sol
            
        return None

    # Initial DPLL call
    assignment = dpll({}, clauses)
    
    if assignment is not None:
        # Ensure all variables are in the assignment
        result = [assignment.get(var, False) for var in range(n)]
        return result
    else:
        # Fallback: Random assignment
        return [random.random() < 0.5 for _ in range(n)]
