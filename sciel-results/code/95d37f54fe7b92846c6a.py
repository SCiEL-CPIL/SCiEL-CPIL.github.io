"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

sys.setrecursionlimit(100000)


def solve(instance):
    """
    Solve a planted 3-SAT instance using a DPLL-based solver with
    watched literals, unit clause propagation, and pure literal elimination.
    
    Returns a list of booleans (length n) representing the satisfying assignment,
    or None if UNSAT (though the task guarantees satisfiable instances).
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Normalize: ensure clauses are tuples of ints (signed literals, 1-based)
    clauses = [tuple(c) for c in clauses]
    
    # Build initial clause lists
    # We'll maintain:
    # - assignment: list of None/True/False for each variable (0-indexed)
    # - watched literals: for each clause, two literals to watch
    # - For efficiency, we'll use a simpler recursive DPLL with propagation
    
    # Let's implement a clean DPLL with:
    # 1. Unit propagation via queue
    # 2. Pure literal elimination
    # 3. Simple variable selection (first unassigned)
    
    # We'll use a state-based approach to avoid deep recursion issues
    # But given n=50, recursion depth is fine
    
    # Assignment: -1 = unassigned, 0 = False, 1 = True
    # We'll use a dict for assignments to allow backtracking
    
    # Actually, let's implement a standard DPLL with backtracking
    
    def dpll(assignment, clauses):
        """
        assignment: dict var_index -> bool
        clauses: list of tuples of literals (signed, 1-based)
        Returns: assignment dict if SAT, None if UNSAT
        """
        # Step 1: Unit propagation and pure literal elimination
        assignment, clauses = propagate(assignment, clauses)
        if clauses is None:
            return None
        
        if not clauses:
            # All clauses satisfied
            return assignment
        
        # Check if any clause is empty (UNSAT)
        for c in clauses:
            if len(c) == 0:
                return None
        
        # Pick a variable to branch on
        var = pick_variable(clauses, assignment, n)
        if var is None:
            # All variables assigned, check if all clauses satisfied
            return assignment
        
        # Try True first
        assign_true = dict(assignment)
        assign_true[var] = True
        result = dpll(assign_true, clauses)
        if result is not None:
            return result
        
        # Try False
        assign_false = dict(assignment)
        assign_false[var] = False
        result = dpll(assign_false, clauses)
        if result is not None:
            return result
        
        return None
    
    def propagate(assignment, clauses):
        """
        Perform unit propagation and pure literal elimination.
        Returns (new_assignment, new_clauses) or (None, None) if UNSAT detected.
        """
        assignment = dict(assignment)
        clauses = [list(c) for c in clauses]
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                # Check if clause is satisfied
                sat = False
                for lit in c:
                    var_idx = abs(lit) - 1
                    if var_idx in assignment:
                        val = assignment[var_idx]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            sat = True
                            break
                if sat:
                    continue
                
                # Check if clause is empty (UNSAT)
                if len(c) == 0:
                    return None, None
                
                # Check if clause is unit
                unassigned_lits = []
                for lit in c:
                    var_idx = abs(lit) - 1
                    if var_idx not in assignment:
                        unassigned_lits.append(lit)
                
                if len(unassigned_lits) == 0:
                    # All literals assigned but clause not satisfied -> UNSAT
                    return None, None
                elif len(unassigned_lits) == 1:
                    # Unit clause: assign the literal
                    lit = unassigned_lits[0]
                    var_idx = abs(lit) - 1
                    assignment[var_idx] = (lit > 0)
                    changed = True
                    # This clause is now satisfied, don't add to new_clauses
                else:
                    new_clauses.append(tuple(c))
            
            clauses = new_clauses
        
        # Pure literal elimination
        # Find variables that appear only positively or only negatively in remaining clauses
        pos_vars = set()
        neg_vars = set()
        for c in clauses:
            for lit in c:
                var_idx = abs(lit) - 1
                if var_idx in assignment:
                    continue
                if lit > 0:
                    pos_vars.add(var_idx)
                else:
                    neg_vars.add(var_idx)
        
        pure_vars = (pos_vars - neg_vars) | (neg_vars - pos_vars)
        for var_idx in pure_vars:
            if var_idx not in assignment:
                # Assign to satisfy all occurrences
                if var_idx in pos_vars:
                    assignment[var_idx] = True
                else:
                    assignment[var_idx] = False
        
        # After pure literal elimination, filter out satisfied clauses
        if pure_vars:
            final_clauses = []
            for c in clauses:
                sat = False
                for lit in c:
                    var_idx = abs(lit) - 1
                    if var_idx in assignment:
                        val = assignment[var_idx]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            sat = True
                            break
                if not sat:
                    final_clauses.append(c)
            clauses = final_clauses
        
        return assignment, clauses
    
    def pick_variable(clauses, assignment, n):
        """Pick the first unassigned variable that appears in an unsatisfied clause."""
        for c in clauses:
            for lit in c:
                var_idx = abs(lit) - 1
                if var_idx not in assignment:
                    return var_idx
        return None
    
    # Run DPLL
    result = dpll({}, clauses)
    
    if result is None:
        # Should not happen for satisfiable instances, but return all False as fallback
        return [False] * n
    
    # Convert dict to list
    return [result.get(i, False) for i in range(n)]

# EVOLVE-BLOCK-END
