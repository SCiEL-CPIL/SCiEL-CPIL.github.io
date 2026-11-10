import time
import sys

sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin

    # Pre-process clauses to 0-based indices
    # Store as list of lists of (var, sign) where sign is 1 for pos, -1 for neg
    # Or better, store as lists of integers: var+1 for pos, -(var+1) for neg? 
    # Let's stick to (var_idx, is_positive) for clarity in DPLL logic, 
    # or use a standard representation: list of ints where >0 is var, <0 is not var.
    # Let's use 1-based signed literals in the input, convert to 0-based signed for internal use.
    # Internal representation: list of lists of ints. 
    # val > 0 means var (val-1) is True.
    # val < 0 means var (-val-1) is False.
    
    internal_clauses = []
    for c in clauses:
        ic = []
        for lit in c:
            if lit > 0:
                ic.append(lit) # Keep 1-based for now? No, let's convert to 0-based signed.
                # 0-based signed: if lit > 0, var = lit-1, sign = +1. Value = lit-1 + 1? 
                # Standard SAT format: var 1..n. 
                # Let's keep 1-based signed literals internally to avoid off-by-one errors in logic.
                pass
            else:
                ic.append(lit)
        internal_clauses.append(ic)
        
    # Actually, let's just keep the input literals as is (1-based signed).
    # Assignment: dict var -> bool, or list of bools (size n).
    # Let's use a list `assignment` of size n, initialized to None (unassigned).
    
    assignment = [None] * n
    
    # Helper to get value of a literal
    def get_val(lit):
        var = abs(lit) - 1
        if assignment[var] is None:
            return None
        if lit > 0:
            return assignment[var]
        else:
            return not assignment[var]
            
    def is_satisfied(clause):
        for lit in clause:
            if get_val(lit) == True:
                return True
        return False
        
    def is_falsified(clause):
        for lit in clause:
            if get_val(lit) == False:
                return True
        return False

    def unit_propagate(clauses):
        """
        Applies unit propagation. 
        Returns (success, new_clauses, assignments_made)
        If success is False, a conflict (empty clause) was found.
        """
        changed = True
        current_clauses = [c[:] for c in clauses] # Shallow copy of lists
        
        while changed:
            changed = False
            new_clauses = []
            for c in current_clauses:
                # Check for unit or empty
                unassigned_lits = []
                is_sat = False
                is_fal = False
                for lit in c:
                    v = get_val(lit)
                    if v == True:
                        is_sat = True
                        break
                    elif v == False:
                        is_fal = True
                        break
                    else:
                        unassigned_lits.append(lit)
                        
                if is_sat:
                    continue # Clause is satisfied, discard it
                elif is_fal:
                    # Check if all assigned and none satisfied? 
                    # Actually, if we hit is_fal, we must check if there are unassigned.
                    # If we break on is_fal, we didn't check the rest. 
                    # Let's re-evaluate properly.
                    pass
                
                # Re-do logic cleanly
                unassigned = []
                sat = False
                for lit in c:
                    v = get_val(lit)
                    if v == True:
                        sat = True
                        break
                    elif v is None:
                        unassigned.append(lit)
                    
                if sat:
                    continue
                    
                if len(unassigned) == 0:
                    return False, None, None # Conflict
                    
                if len(unassigned) == 1:
                    # Unit clause
                    lit = unassigned[0]
                    var = abs(lit) - 1
                    val = lit > 0
                    if assignment[var] is not None:
                        if assignment[var] != val:
                            return False, None, None # Conflict
                    else:
                        assignment[var] = val
                        changed = True
                    # Clause becomes satisfied, so we don't add it to new_clauses
                    # But wait, if we assign, the clause is satisfied.
                    # However, other clauses might be affected.
                    # We don't add the satisfied clause.
                else:
                    new_clauses.append(unassigned)
                    
            current_clauses = new_clauses
            
        return True, current_clauses, None

    def pure_literal_elimination(clauses):
        """
        Identifies and assigns pure literals.
        Returns (success, new_clauses)
        """
        # Count occurrences of var as pos and neg
        pos_count = [0] * n
        neg_count = [0] * n
        
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                if lit > 0:
                    pos_count[var] += 1
                else:
                    neg_count[var] += 1
                    
        changed = False
        for var in range(n):
            if assignment[var] is None:
                if pos_count[var] == 0:
                    assignment[var] = False
                    changed = True
                elif neg_count[var] == 0:
                    assignment[var] = True
                    changed = True
                    
        if not changed:
            return True, clauses
            
        # If assignments were made, we need to filter clauses
        new_clauses = []
        for c in clauses:
            unassigned = []
            sat = False
            for lit in c:
                v = get_val(lit)
                if v == True:
                    sat = True
                    break
                elif v is None:
                    unassigned.append(lit)
                    
            if not sat:
                if len(unassigned) == 0:
                    return False, None # Conflict
                new_clauses.append(unassigned)
                
        return True, new_clauses

    def dpll(clauses):
        # Check deadline
        if time.time() > deadline:
            return None
            
        # 1. Check for empty clause (failure)
        if any(len(c) == 0 for c in clauses):
            return False
            
        # 2. Check for tautology/empty set (success)
        if not clauses:
            return True
            
        # 3. Unit Propagation
        success, clauses, _ = unit_propagate(clauses)
        if not success:
            return False
        if not clauses:
            return True
            
        # 4. Pure Literal Elimination
        success, clauses = pure_literal_elimination(clauses)
        if not success:
            return False
        if not clauses:
            return True
            
        # 5. Select a variable
        # Heuristic: Variable with highest frequency in remaining clauses
        var_freq = [0] * n
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                var_freq[var] += 1
                
        # Pick var with max freq, breaking ties arbitrarily (or by index)
        max_freq = max(var_freq)
        if max_freq == 0:
            # No variables? Should be handled by empty clause check
            return True
            
        var_to_split = None
        for v in range(n):
            if var_freq[v] == max_freq:
                var_to_split = v
                break
                
        if var_to_split is None:
            # Fallback to first unassigned
            for v in range(n):
                if assignment[v] is None:
                    var_to_split = v
                    break
            
        if var_to_split is None:
            return True # All assigned
            
        # 6. Recurse
        # Try True
        assignment[var_to_split] = True
        if dpll(clauses):
            return True
            
        # Try False
        assignment[var_to_split] = False
        if dpll(clauses):
            return True
            
        # Backtrack
        assignment[var_to_split] = None
        return False

    # Run DPLL
    result = dpll(internal_clauses)
    
    if result:
        # Convert assignment list (True/False/None) to list of booleans
        # If any are None, they are irrelevant, but we must return a list of booleans.
        # We can set them to False arbitrarily.
        final_assignment = [bool(assignment[i]) if assignment[i] is not None else False for i in range(n)]
        return final_assignment
    else:
        # Should not happen for satisfiable instances, but return a default just in case
        return [False] * n
