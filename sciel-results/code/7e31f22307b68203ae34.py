import time
import random

# EVOLVE-BLOCK-START
def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 290 seconds to be safe
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess: Convert 1-based signed literals to 0-based signed integers
    # literal > 0 means variable (literal-1) is True
    # literal < 0 means variable (-literal-1) is False
    # We represent literals as:
    #   variable i (0-based) positive -> i
    #   variable i (0-based) negative -> -i-1
    # But standard DPLL often uses:
    #   variable i positive -> i
    #   variable i negative -> -i
    # Let's stick to a simple internal representation:
    #   lit = var_index if positive, -var_index-1 if negative? 
    # Actually, let's use 0-based indices where:
    #   +v is represented as v
    #   -v is represented as -v-1
    # This avoids 0 ambiguity for the sign check easily? 
    # No, let's use:
    #   var i is True:  i
    #   var i is False: -i-1
    # This way, abs(lit) > i? No.
    # Let's use:
    #   Positive literal of var i: i
    #   Negative literal of var i: -i-1
    # Check:
    #   var 0 pos: 0
    #   var 0 neg: -1
    #   var 1 pos: 1
    #   var 1 neg: -2
    #   
    # Helper to get var index:
    #   if lit >= 0: var = lit
    #   else: var = -lit - 1
    # Helper to check if lit is satisfied by assignment (list of bools):
    #   if lit >= 0: assignment[lit] == True
    #   else: assignment[-lit-1] == False

    def to_internal(lit):
        if lit > 0:
            return lit - 1
        else:
            return -lit - 1 # This is getting confusing. Let's use a simpler mapping.
            
    # Simpler mapping:
    # Internal literal:
    #   Positive var i: i
    #   Negative var i: -i - 1
    # Let's re-verify:
    #   Input lit: 1..n or -1..-n
    #   Input lit 1 (var 0 pos) -> Internal 0
    #   Input lit -1 (var 0 neg) -> Internal -1 - 1 = -2? No.
    
    # Let's use:
    #   Internal lit:
    #   Positive var i: i
    #   Negative var i: -i - 1
    #   Input lit 1 -> var 0 pos -> Internal 0
    #   Input lit -1 -> var 0 neg -> Internal -0 - 1 = -1
    #   Input lit 2 -> var 1 pos -> Internal 1
    #   Input lit -2 -> var 1 neg -> Internal -1 - 1 = -2
    #   
    #   Var from Internal:
    #   if lit >= 0: var = lit
    #   else: var = -lit - 1
    #   
    #   Is satisfied:
    #   if lit >= 0: assignment[lit]
    #   else: not assignment[-lit - 1]

    internal_clauses = []
    for c in clauses:
        ic = []
        for l in c:
            if l > 0:
                ic.append(l - 1)
            else:
                ic.append(-l - 1)
        internal_clauses.append(ic)

    # DPLL Solver
    # State:
    #   assignment: list of bools or None (unassigned)
    #   clauses: list of lists of internal literals (only unassigned literals)
    
    def dpll(assignment, clauses):
        # Check for empty clause (unsat)
        if any(len(c) == 0 for c in clauses):
            return False, None
        
        # Unit Propagation and Pure Literal Elimination
        # We need to be careful with performance. n=50 is small.
        
        while True:
            # Check time
            if time.time() > deadline:
                return None, None # Timeout indicator
            
            # 1. Unit Propagation
            unit_clause_found = False
            new_clauses = []
            for c in clauses:
                if len(c) == 1:
                    lit = c[0]
                    var = lit if lit >= 0 else -lit - 1
                    # Determine value
                    if lit >= 0:
                        val = True
                    else:
                        val = False
                    
                    # If already assigned, check consistency
                    if assignment[var] is not None:
                        if assignment[var] != val:
                            return False, None
                        # Clause is satisfied, remove it
                        continue
                    else:
                        assignment[var] = val
                        unit_clause_found = True
                        # Propagate: remove satisfied clauses, remove negated lit from others
                        # We'll rebuild clauses in the next iteration or handle here
                        # For efficiency with n=50, rebuilding is fine.
                        pass
                else:
                    new_clauses.append(c)
            
            if unit_clause_found:
                # Rebuild clauses based on new assignments
                filtered_clauses = []
                for c in new_clauses:
                    new_c = []
                    satisfied = False
                    for lit in c:
                        var = lit if lit >= 0 else -lit - 1
                        if assignment[var] is not None:
                            if lit >= 0:
                                if assignment[var]:
                                    satisfied = True
                                    break
                            else:
                                if not assignment[var]:
                                    satisfied = True
                                    break
                            # If assigned and doesn't satisfy, remove lit
                        else:
                            new_c.append(lit)
                    if not satisfied:
                        filtered_clauses.append(new_c)
                clauses = filtered_clauses
                continue
            
            # 2. Pure Literal Elimination
            # Find literals that appear in only one sign
            pos_counts = [0] * n
            neg_counts = [0] * n
            for c in clauses:
                for lit in c:
                    var = lit if lit >= 0 else -lit - 1
                    if lit >= 0:
                        pos_counts[var] += 1
                    else:
                        neg_counts[var] += 1
            
            pure_found = False
            for var in range(n):
                if assignment[var] is not None:
                    continue
                if pos_counts[var] > 0 and neg_counts[var] == 0:
                    assignment[var] = True
                    pure_found = True
                elif neg_counts[var] > 0 and pos_counts[var] == 0:
                    assignment[var] = False
                    pure_found = True
            
            if pure_found:
                # Rebuild clauses
                filtered_clauses = []
                for c in clauses:
                    new_c = []
                    satisfied = False
                    for lit in c:
                        var = lit if lit >= 0 else -lit - 1
                        if assignment[var] is not None:
                            if lit >= 0:
                                if assignment[var]:
                                    satisfied = True
                                    break
                            else:
                                if not assignment[var]:
                                    satisfied = True
                                    break
                        else:
                            new_c.append(lit)
                    if not satisfied:
                        filtered_clauses.append(new_c)
                clauses = filtered_clauses
                continue
            
            # 3. Check if all assigned
            if all(a is not None for a in assignment):
                return True, assignment[:]
            
            # 4. Branch
            # Heuristic: Pick variable with highest frequency
            var_freq = [0] * n
            for c in clauses:
                for lit in c:
                    var = lit if lit >= 0 else -lit - 1
                    if assignment[var] is None:
                        var_freq[var] += 1
            
            # Find max freq unassigned var
            best_var = -1
            max_freq = -1
            for var in range(n):
                if assignment[var] is None and var_freq[var] > max_freq:
                    max_freq = var_freq[var]
                    best_var = var
            
            if best_var == -1:
                # All assigned?
                return True, assignment[:]
            
            # Try True
            assignment[best_var] = True
            # Propagate and recurse
            # We need to make a copy of clauses or rely on the unit prop loop to modify?
            # The unit prop loop above modifies `clauses` in place? No, it creates `filtered_clauses`.
            # But it doesn't return to the caller with the modified state if it didn't find units/pures?
            # Actually, the loop above only runs if we enter the `while True`.
            # If no units/pures, we break out? No, the `continue` is inside `if unit_clause_found` or `if pure_found`.
            # If neither, we fall through to branching.
            
            # To branch, we need to pass the current state.
            # The current `clauses` list is valid for the current assignment.
            
            # Copy assignment
            assign_copy = assignment[:]
            
            # We need to propagate the new assignment.
            # Instead of complex state passing, let's just call a helper that does unit prop from scratch?
            # Or, since n is small, we can just filter clauses here.
            
            def propagate(assign, cls):
                # Returns (is_sat, new_assign, new_clauses) or (False, None, None) for UNSAT
                # This is a simplified propagation
                changed = True
                while changed:
                    changed = False
                    # Unit
                    new_cls = []
                    for c in cls:
                        if len(c) == 1:
                            lit = c[0]
                            var = lit if lit >= 0 else -lit - 1
                            val = True if lit >= 0 else False
                            if assign[var] is not None:
                                if assign[var] != val:
                                    return False, None, None
                                continue # Satisfied
                            else:
                                assign[var] = val
                                changed = True
                                continue
                        new_cls.append(c)
                    cls = new_cls
                    
                    # Filter satisfied/removed lits
                    filtered_cls = []
                    for c in cls:
                        new_c = []
                        sat = False
                        for lit in c:
                            var = lit if lit >= 0 else -lit - 1
                            if assign[var] is not None:
                                if lit >= 0:
                                    if assign[var]: sat = True; break
                                else:
                                    if not assign[var]: sat = True; break
                            else:
                                new_c.append(lit)
                        if not sat:
                            filtered_cls.append(new_c)
                    cls = filtered_cls
                    
                    if any(len(c)==0 for c in cls):
                        return False, None, None
                        
            # Try True
            ok, res_assign, _ = propagate(assign_copy, clauses)
            if ok:
                # Check if solved
                if all(a is not None for a in res_assign):
                    return True, res_assign
                # Recurse
                res, sol = dpll(res_assign, []) # Wait, dpll expects clauses.
                # My dpll signature is dpll(assignment, clauses).
                # But the clauses passed to dpll should be the ones AFTER propagation.
                # The propagate function returns new_clauses. I discarded it.
                
            # Let's refactor to be cleaner.
            
            return False, None # Fallback, shouldn't happen if logic is right
            
    # The above DPLL implementation is getting messy with state management.
    # Let's write a cleaner, recursive DPLL.
    
    def dpll_clean(assignment, clauses):
        if time.time() > deadline:
            return None, None
            
        # 1. Unit Propagation
        while True:
            unit_found = False
            new_clauses = []
            for c in clauses:
                if len(c) == 1:
                    lit = c[0]
                    var = lit if lit >= 0 else -lit - 1
                    val = True if lit >= 0 else False
                    if assignment[var] is not None:
                        if assignment[var] != val:
                            return False, None
                        # Satisfied, drop clause
                    else:
                        assignment[var] = val
                        unit_found = True
                else:
                    new_clauses.append(c)
            
            if unit_found:
                # Filter clauses
                filtered = []
                for c in new_clauses:
                    nc = []
                    sat = False
                    for lit in c:
                        var = lit if lit >= 0 else -lit - 1
                        if assignment[var] is not None:
                            if lit >= 0:
                                if assignment[var]: sat = True; break
                            else:
                                if not assignment[var]: sat = True; break
                        else:
                            nc.append(lit)
                    if not sat:
                        filtered.append(nc)
                clauses = filtered
                continue
            else:
                break
        
        # Check for empty clause
        if any(len(c) == 0 for c in clauses):
            return False, None
            
        # Check if all assigned
        if all(a is not None for a in assignment):
            return True, assignment[:]
            
        # 2. Pure Literal Elimination
        pos_counts = [0] * n
        neg_counts = [0] * n
        for c in clauses:
            for lit in c:
                var = lit if lit >= 0 else -lit - 1
                if lit >= 0: pos_counts[var] += 1
                else: neg_counts[var] += 1
                
        pure_found = False
        for var in range(n):
            if assignment[var] is None:
                if pos_counts[var] > 0 and neg_counts[var] == 0:
                    assignment[var] = True
                    pure_found = True
                elif neg_counts[var] > 0 and pos_counts[var] == 0:
                    assignment[var] = False
                    pure_found = True
                    
        if pure_found:
            # Filter clauses again
            filtered = []
            for c in clauses:
                nc = []
                sat = False
                for lit in c:
                    var = lit if lit >= 0 else -lit - 1
                    if assignment[var] is not None:
                        if lit >= 0:
                            if assignment[var]: sat = True; break
                        else:
                            if not assignment[var]: sat = True; break
                    else:
                        nc.append(lit)
                if not sat:
                    filtered.append(nc)
            clauses = filtered
            # Loop back to unit propagation
            return dpll_clean(assignment, clauses)
            
        # 3. Branch
        # Heuristic: Most frequent variable
        var_freq = [0] * n
        for c in clauses:
            for lit in c:
                var = lit if lit >= 0 else -lit - 1
                if assignment[var] is None:
                    var_freq[var] += 1
                    
        best_var = -1
        max_freq = -1
        for var in range(n):
            if assignment[var] is None and var_freq[var] > max_freq:
                max_freq = var_freq[var]
                best_var = var
                
        if best_var == -1:
            return True, assignment[:]
            
        # Try True
        assignment[best_var] = True
        # We need to propagate this assignment to get the new clause set
        # To avoid copying huge structures, we can pass the current clauses and let the recursive call do propagation.
        # But dpll_clean expects the clauses to be consistent with the assignment?
        # No, dpll_clean starts with unit propagation. So we can just pass the current clauses.
        # However, the current clauses still contain the variable `best_var`.
        # The unit propagation in the next call will handle it.
        
        ok, sol = dpll_clean(assignment, clauses)
        if ok is not None:
            if ok:
                return True, sol
            # If UNSAT, backtrack
            assignment[best_var] = False
            ok, sol = dpll_clean(assignment, clauses)
            if ok:
                return True, sol
            else:
                return False, None
        else:
            # Timeout
            return None, None

    # Run DPLL
    initial_assignment = [None] * n
    ok, sol = dpll_clean(initial_assignment, internal_clauses)
    
    if ok is True and sol is not None:
        return sol
    elif ok is False:
        # Should not happen for satisfiable instances
        return [False] * n
    else:
        # Timeout
        # Fall back to random or WalkSAT
        rng = random.Random(0)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        def satisfied(clause):
            return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
        for _ in range(20000):
            unsat = [c for c in clauses if not satisfied(c)]
            if not unsat:
                break
            clause = rng.choice(unsat)
            literal = rng.choice(clause)
            assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
        return assignment
# EVOLVE-BLOCK-END
