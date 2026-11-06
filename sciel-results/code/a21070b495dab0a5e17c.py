"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def solve(instance):
    """
    Solve a 3-CNF SAT instance using DPLL with unit propagation and pure literal elimination.
    Returns a list of booleans representing the satisfying assignment.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 300s for safety
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Represent assignment as a dict: {var_index: bool}
    # var_index is 0-based internally
    assignment = {}
    
    # Convert clauses to use 0-based indices for faster access
    # Clause is a list of signed literals. Positive literal l -> var l-1, negative -> var l-1.
    # We store clauses as lists of (var, sign) where sign is +1 or -1.
    # Actually, keeping signed literals and using abs(l)-1 is fine, but let's optimize.
    # Let's pre-process clauses to remove tautologies and duplicate literals if any, though 3-SAT usually doesn't have them.
    
    def check_time():
        return time.time() > deadline

    def get_var(lit):
        return abs(lit) - 1

    def is_satisfied(clause, assign):
        for lit in clause:
            v = get_var(lit)
            if v in assign:
                val = assign[v]
                if (lit > 0 and val) or (lit < 0 and not val):
                    return True
        return False

    def simplify_clauses(clauses, assign):
        """
        Returns a list of simplified clauses.
        If a clause is satisfied, it is removed.
        If a literal is false, it is removed from the clause.
        If a clause becomes empty, return None (contradiction).
        """
        new_clauses = []
        for clause in clauses:
            new_clause = []
            satisfied = False
            for lit in clause:
                v = get_var(lit)
                if v in assign:
                    val = assign[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
                    # If false, skip this literal
                else:
                    new_clause.append(lit)
            
            if satisfied:
                continue
            
            if len(new_clause) == 0:
                return None # Contradiction
            
            new_clauses.append(new_clause)
        
        return new_clauses

    def find_pure_literal(clauses):
        """
        Find a literal that appears in only one polarity across all clauses.
        Returns (lit, var) or (None, None).
        """
        if not clauses:
            return None, None
        
        pos_vars = set()
        neg_vars = set()
        
        for clause in clauses:
            for lit in clause:
                v = get_var(lit)
                if lit > 0:
                    pos_vars.add(v)
                else:
                    neg_vars.add(v)
        
        # A variable is pure if it appears only in pos or only in neg
        # We want to assign the literal that is pure.
        # If v is in pos_vars but not neg_vars, then literal +v is pure.
        # If v is in neg_vars but not pos_vars, then literal -v is pure.
        
        # Optimization: We can just pick the first one we find.
        # To be deterministic, we can sort or just iterate.
        # Since n=50, this is fast.
        
        # Let's find a pure literal.
        for v in range(n):
            in_pos = v in pos_vars
            in_neg = v in neg_vars
            
            if in_pos and not in_neg:
                return v + 1, v # Literal +v
            if in_neg and not in_pos:
                return -(v + 1), v # Literal -v
                
        return None, None

    def unit_propagate(clauses, assign):
        """
        Perform unit propagation.
        Returns (success, new_clauses, new_assign).
        success is False if contradiction is found.
        """
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            unit_clauses = [c for c in clauses if len(c) == 1]
            
            if not unit_clauses:
                break
            
            # Process all unit clauses
            # Note: In a single pass, multiple unit clauses might constrain the same variable or different ones.
            # If two unit clauses constrain the same variable to opposite values, it's a contradiction.
            # We can process them one by one.
            
            for clause in unit_clauses:
                if len(clause) != 1:
                    continue # Might have been modified? No, we iterate a snapshot? 
                             # Better to re-scan or handle carefully.
                             # Let's just pick one unit clause, assign, simplify, and restart the loop.
                             # This is simpler and safer.
                lit = clause[0]
                v = get_var(lit)
                
                if v in assign:
                    # Check if assignment conflicts
                    val = assign[v]
                    if (lit > 0 and not val) or (lit < 0 and val):
                        return False, None, None
                    # If it matches, it's already satisfied, so this clause should have been removed in simplify?
                    # Wait, simplify removes satisfied clauses. So if a unit clause exists, the var is NOT assigned.
                    # So this case (v in assign) shouldn't happen for a unit clause in a simplified set.
                    # But if we add assignments one by one, we should simplify after each.
                    pass
                
                # Assign
                assign[v] = (lit > 0)
                changed = True
                
                # Simplify clauses
                clauses = simplify_clauses(clauses, assign)
                if clauses is None:
                    return False, None, None
                
                break # Restart loop to check for new unit clauses
            
            if check_time():
                return True, clauses, assign # Return current state if time runs out (best effort)
        
        return True, clauses, assign

    def dpll(clauses, assign):
        """
        Recursive DPLL.
        Returns a dict of assignments or None if unsat.
        """
        if check_time():
            return assign # Return partial assignment if time runs out
        
        # Base case: no clauses
        if not clauses:
            return assign
        
        # Step 1: Unit Propagation
        success, clauses, assign = unit_propagate(clauses, assign)
        if not success:
            return None
        if check_time():
            return assign
        
        # Base case: no clauses after propagation
        if not clauses:
            return assign
        
        # Step 2: Pure Literal Elimination
        lit, v = find_pure_literal(clauses)
        if lit is not None:
            assign[v] = (lit > 0)
            clauses = simplify_clauses(clauses, assign)
            if clauses is None:
                return None
            if check_time():
                return assign
            return dpll(clauses, assign)
        
        # Step 3: Variable Selection
        # Heuristic: Choose variable appearing in the most clauses (or shortest clause containing it).
        # Let's pick the variable from the shortest clause (Most Constrained).
        # Find the shortest clause
        min_len = min(len(c) for c in clauses)
        shortest_clause = None
        for c in clauses:
            if len(c) == min_len:
                shortest_clause = c
                break
        
        # Pick the first variable in the shortest clause that is not assigned
        branch_var = None
        branch_lit = None
        for lit in shortest_clause:
            v = get_var(lit)
            if v not in assign:
                branch_var = v
                branch_lit = lit
                break
        
        if branch_var is None:
            # This should not happen if clauses are simplified correctly
            return None

        # Branch 1: Try branch_lit = True
        assign_copy = assign.copy()
        assign_copy[branch_var] = (branch_lit > 0)
        clauses_1 = simplify_clauses(clauses, assign_copy)
        if clauses_1 is not None:
            result = dpll(clauses_1, assign_copy)
            if result is not None:
                return result
        
        # Branch 2: Try branch_lit = False
        assign_copy = assign.copy()
        assign_copy[branch_var] = not (branch_lit > 0)
        clauses_2 = simplify_clauses(clauses, assign_copy)
        if clauses_2 is not None:
            result = dpll(clauses_2, assign_copy)
            if result is not None:
                return result
        
        return None

    # Start solving
    # Initial simplification
    current_clauses = simplify_clauses(clauses, {})
    if current_clauses is None:
        # Unsat? But task says satisfiable. Return empty or random?
        # Task guarantees satisfiable.
        return [False] * n
        
    result_assign = dpll(current_clauses, {})
    
    # If result is None or partial, we need to fill in unassigned variables.
    # Since the instance is satisfiable, if DPLL returned None, it might be due to time limit or bug.
    # If it returned a partial assignment, we can just assign False to the rest, 
    # but that might not satisfy the clauses. 
    # However, DPLL with full propagation should find a full assignment if it finishes.
    # If it hits the deadline, it returns the current `assign` dict which might be partial.
    # In that case, we can just fill the rest with False and hope for the best, or run a quick local search?
    # Given the small size (50 vars), DPLL should finish instantly.
    
    final_assign = [False] * n
    if result_assign:
        for v, val in result_assign.items():
            if 0 <= v < n:
                final_assign[v] = val
    else:
        # Fallback: Random assignment
        import random
        rng = random.Random(0)
        final_assign = [rng.random() < 0.5 for _ in range(n)]
        
    return final_assign
# EVOLVE-BLOCK-END
