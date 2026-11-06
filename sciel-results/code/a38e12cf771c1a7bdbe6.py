"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert clauses to 0-based indices with sign encoded as +1/-1
    # Each literal becomes (index, sign) where sign is +1 if literal > 0, -1 if literal < 0
    def lit_to_idx_sign(l):
        return abs(l) - 1, (1 if l > 0 else -1)
    
    # Preprocess: remove duplicate literals within clauses, remove tautologies
    processed_clauses = []
    for clause in clauses:
        seen = set()
        new_clause = []
        is_tautology = False
        for l in clause:
            idx, sign = lit_to_idx_sign(l)
            if idx in seen:
                is_tautology = True
                break
            seen.add(idx)
            new_clause.append((idx, sign))
        if not is_tautology and new_clause:
            processed_clauses.append(new_clause)
    
    # If no clauses, any assignment works
    if not processed_clauses:
        return [False] * n
    
    # DPLL with unit propagation and pure literal elimination
    
    # Build initial watch lists and occurrence lists
    # We'll use a simpler approach: maintain current clauses and do unit propagation iteratively
    
    def dpll(current_clauses, assignment):
        """
        current_clauses: list of lists of (idx, sign)
        assignment: list of None/True/False for each variable
        Returns: assignment if satisfiable, None if unsatisfiable
        """
        # Unit propagation
        while True:
            # Find unit clauses
            unit_clauses = [c for c in current_clauses if len(c) == 1]
            if unit_clauses:
                # Propagate all units
                changed = True
                while changed:
                    changed = False
                    for c in current_clauses:
                        if len(c) == 1:
                            idx, sign = c[0]
                            val = (sign == 1)
                            if assignment[idx] is None:
                                assignment[idx] = val
                                changed = True
                            elif assignment[idx] != val:
                                return None  # Contradiction
                # Remove satisfied clauses and update unsatisfied ones
                new_clauses = []
                for c in current_clauses:
                    sat = False
                    new_c = []
                    for idx, sign in c:
                        if assignment[idx] is not None:
                            if assignment[idx] == (sign == 1):
                                sat = True
                                break
                        else:
                            new_c.append((idx, sign))
                    if not sat:
                        if new_c:
                            new_clauses.append(new_c)
                        # If new_c is empty, clause is falsified -> contradiction
                        elif not new_c:
                            return None
                current_clauses = new_clauses
            else:
                break
        
        # Check if all clauses are satisfied
        if not current_clauses:
            # Fill remaining variables with False
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment
        
        # Pure literal elimination
        # Find variables that only appear with one sign in remaining clauses
        pos_occurs = set()
        neg_occurs = set()
        for c in current_clauses:
            for idx, sign in c:
                if sign == 1:
                    pos_occurs.add(idx)
                else:
                    neg_occurs.add(idx)
        
        pure_vars = []
        for idx in pos_occurs:
            if idx not in neg_occurs and assignment[idx] is None:
                pure_vars.append((idx, True))
        for idx in neg_occurs:
            if idx not in pos_occurs and assignment[idx] is None:
                pure_vars.append((idx, False))
        
        if pure_vars:
            # Apply pure literal elimination
            for idx, val in pure_vars:
                assignment[idx] = val
            # Remove satisfied clauses and update
            new_clauses = []
            for c in current_clauses:
                sat = False
                new_c = []
                for idx, sign in c:
                    if assignment[idx] is not None:
                        if assignment[idx] == (sign == 1):
                            sat = True
                            break
                    else:
                        new_c.append((idx, sign))
                if not sat:
                    if new_c:
                        new_clauses.append(new_c)
                    elif not new_c:
                        return None
            current_clauses = new_clauses
            
            # Continue with unit propagation again
            return dpll(current_clauses, assignment)
        
        # Choose a branching variable
        # Pick a variable from the first clause
        branch_clause = current_clauses[0]
        if not branch_clause:
            return None
        
        idx, sign = branch_clause[0]
        
        # Try both assignments
        for val in [True, False]:
            new_assignment = assignment[:]
            new_assignment[idx] = val
            
            # Apply this assignment and propagate
            new_clauses = []
            consistent = True
            for c in current_clauses:
                sat = False
                new_c = []
                for i, s in c:
                    if new_assignment[i] is not None:
                        if new_assignment[i] == (s == 1):
                            sat = True
                            break
                    else:
                        new_c.append((i, s))
                if not sat:
                    if new_c:
                        new_clauses.append(new_c)
                    elif not new_c:
                        consistent = False
                        break
            if consistent:
                result = dpll(new_clauses, new_assignment)
                if result is not None:
                    return result
        
        return None
    
    initial_assignment = [None] * n
    result = dpll(processed_clauses, initial_assignment)
    
    if result is None:
        # Fallback: return all False (shouldn't happen for satisfiable instances)
        return [False] * n
    
    return result
# EVOLVE-BLOCK-END
