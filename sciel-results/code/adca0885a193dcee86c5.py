import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to a list of lists of ints
    # Clauses are already lists of signed ints
    
    # We will use a recursive DPLL with unit propagation.
    # assignment[i] is None, True, or False.
    # We need to handle the state carefully.
    
    # To make it efficient, we can copy clauses or use a stack.
    # For n=50, copying is cheap.
    
    def unit_propagate(clauses, assignment):
        """
        Performs unit propagation.
        Returns (new_clauses, new_assignment, success)
        success is False if a conflict (empty clause) is found.
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Check if clause is satisfied
                sat = False
                unassigned_lits = []
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] is None:
                        unassigned_lits.append(lit)
                    else:
                        val = assignment[var]
                        is_true = (lit > 0)
                        if val == is_true:
                            sat = True
                            break
                        # else literal is false, continue
                
                if sat:
                    continue # Clause satisfied, remove it
                
                if len(unassigned_lits) == 0:
                    return None, None, False # Conflict: empty clause
                
                if len(unassigned_lits) == 1:
                    # Unit clause: must assign the remaining literal
                    lit = unassigned_lits[0]
                    var = abs(lit) - 1
                    assignment[var] = (lit > 0)
                    changed = True
                    # We don't add this clause to new_clauses because it's satisfied by the assignment
                    # But we must ensure we don't lose information. 
                    # Actually, once assigned, the clause is satisfied.
                else:
                    # Clause has >1 unassigned literals, keep it (possibly reduced)
                    # We can reduce the clause by removing literals that are assigned False
                    reduced_clause = []
                    for lit in unassigned_lits:
                        reduced_clause.append(lit)
                    # Literals assigned False are already excluded from unassigned_lits?
                    # No, unassigned_lits only contains unassigned ones.
                    # We should also remove literals that are assigned False from the clause if we were keeping the full clause.
                    # But we are building new_clauses from scratch.
                    # Let's rebuild the clause with only unassigned literals.
                    # Wait, if a literal is assigned False, it's not in unassigned_lits.
                    # So reduced_clause is correct.
                    new_clauses.append(reduced_clause)
            
            clauses = new_clauses
            
            if not clauses:
                return clauses, assignment, True # All clauses satisfied
        
        return clauses, assignment, True

    def pure_literal_eliminate(clauses, assignment):
        """
        Eliminates pure literals.
        Returns (new_clauses, new_assignment, success)
        """
        if not clauses:
            return clauses, assignment, True
            
        # Count occurrences of each literal
        # lit_count[lit] = count
        lit_count = {}
        for clause in clauses:
            for lit in clause:
                lit_count[lit] = lit_count.get(lit, 0) + 1
                # Also track the opposite literal to see if it's pure
                # If lit is in clause, -lit is the opposite.
                
        # Actually, easier: for each unassigned variable, check if only one polarity appears.
        pure_lits = []
        for var in range(n):
            if assignment[var] is not None:
                continue
            pos_lit = var + 1
            neg_lit = -(var + 1)
            pos_count = lit_count.get(pos_lit, 0)
            neg_count = lit_count.get(neg_lit, 0)
            
            if pos_count > 0 and neg_count == 0:
                pure_lits.append(pos_lit)
            elif neg_count > 0 and pos_count == 0:
                pure_lits.append(neg_lit)
        
        if not pure_lits:
            return clauses, assignment, True
            
        # Assign pure literals
        for lit in pure_lits:
            var = abs(lit) - 1
            assignment[var] = (lit > 0)
            
        # Re-run unit propagation to simplify
        return unit_propagate(clauses, assignment)

    def select_variable(clauses, assignment):
        """
        Selects a variable to branch on.
        Heuristic: Pick variable from the shortest clause.
        """
        if not clauses:
            return None
        
        # Find the clause with the fewest unassigned literals
        min_len = float('inf')
        best_clause = None
        for clause in clauses:
            unassigned_count = 0
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    unassigned_count += 1
            if unassigned_count < min_len:
                min_len = unassigned_count
                best_clause = clause
                if min_len == 1:
                    # Should have been caught by unit propagation, but just in case
                    break
        
        if best_clause is None:
            return None
            
        # Pick the first unassigned variable in the best clause
        for lit in best_clause:
            var = abs(lit) - 1
            if assignment[var] is None:
                return var
                
        # Fallback: any unassigned variable
        for var in range(n):
            if assignment[var] is None:
                return var
                
        return None

    def dpll(clauses, assignment):
        # Unit Propagation
        clauses, assignment, success = unit_propagate(clauses, assignment)
        if not success:
            return None
        
        # If no clauses left, satisfied
        if not clauses:
            return assignment
        
        # Pure Literal Elimination (optional, can be skipped for speed if DPLL is fast enough)
        # clauses, assignment, success = pure_literal_eliminate(clauses, assignment)
        # if not success:
        #     return None
        # if not clauses:
        #     return assignment

        # Select variable
        var = select_variable(clauses, assignment)
        if var is None:
            # No unassigned variables, but clauses remain?
            # This means all variables are assigned, but some clauses are not satisfied.
            # This is a conflict.
            return None
            
        # Try True
        assignment[var] = True
        # Copy clauses and assignment for recursion? 
        # Since we mutate, we should copy.
        new_assignment = assignment[:]
        result = dpll(clauses, new_assignment)
        if result is not None:
            return result
            
        # Try False
        assignment[var] = False
        new_assignment = assignment[:]
        result = dpll(clauses, new_assignment)
        if result is not None:
            return result
            
        # Backtrack
        assignment[var] = None
        return None

    # Initialize assignment
    assignment = [None] * n
    
    # Start DPLL
    solution = dpll(clauses, assignment)
    
    if solution is not None:
        return solution
    else:
        # If DPLL fails (should not happen for satisfiable instances within time), 
        # fallback to WalkSAT with restarts as a last resort?
        # Or just return a random assignment? 
        # The task says "planted satisfiable", so DPLL should find it.
        # If it times out or fails, we need a fallback.
        # Let's add a simple WalkSAT with restarts as a fallback if DPLL returns None.
        
        import random
        rng = random.Random(0)
        # WalkSAT with restarts
        for restart in range(10):
            w_assignment = [rng.random() < 0.5 for _ in range(n)]
            for i in range(10000):
                unsat = [c for c in clauses if not any(w_assignment[abs(l)-1] == (l>0) for l in c)]
                if not unsat:
                    return w_assignment
                clause = rng.choice(unsat)
                # Pick variable to flip
                # Heuristic: flip variable that appears in most unsat clauses?
                # Simple: random literal from clause
                lit = rng.choice(clause)
                w_assignment[abs(lit)-1] = not w_assignment[abs(lit)-1]
                
        # If all fails, return last assignment
        return w_assignment
