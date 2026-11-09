# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to 0-based literals
    # lit > 0 means var (lit-1) is True
    # lit < 0 means var (-lit-1) is False
    # We represent literals as signed 1-based in the input, 
    # but internally we can use 0-based signed or just keep 1-based.
    # Let's keep 1-based signed for simplicity in logic, but use 0-based for array indexing.
    
    # Preprocess clauses: remove tautologies and duplicates?
    # For 3-SAT, tautologies are unlikely in planted instances, but let's be safe.
    # Also, if a clause is empty, UNSAT.
    
    processed_clauses = []
    for c in clauses:
        # Check for tautology
        has_pos = False
        has_neg = False
        for lit in c:
            if lit > 0:
                has_pos = True
            else:
                has_neg = True
        if has_pos and has_neg:
            # Check if same variable
            vars_in_clause = set(abs(l) for l in c)
            if len(vars_in_clause) < len(c):
                continue # Tautology
        processed_clauses.append(c)
        
    if not processed_clauses:
        # All clauses were tautologies, any assignment works
        return [False] * n

    # DPLL Solver
    # assignment: dict var_index -> bool
    # clauses: list of lists of signed 1-based literals
    
    def unit_propagate(clauses, assignment):
        """
        Perform unit propagation.
        Returns (new_clauses, new_assignment, satisfied_count, unsat_flag)
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                # Filter assigned literals
                unassigned_lits = []
                is_satisfied = False
                is_unsat = False
                for lit in c:
                    var = abs(lit) - 1
                    if var in assignment:
                        val = assignment[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            is_satisfied = True
                            break
                        else:
                            is_unsat = True
                            break
                    else:
                        unassigned_lits.append(lit)
                
                if is_unsat:
                    return None, None, False # UNSAT
                if is_satisfied:
                    continue
                if len(unassigned_lits) == 1:
                    # Unit clause
                    lit = unassigned_lits[0]
                    var = abs(lit) - 1
                    val = lit > 0
                    assignment[var] = val
                    changed = True
                else:
                    new_clauses.append(unassigned_lits)
            
            clauses = new_clauses
        
        # Check for empty clauses
        for c in clauses:
            if len(c) == 0:
                return None, None, False # UNSAT
        
        return clauses, assignment, True

    def pure_literal_elimination(clauses, assignment):
        """
        Identify pure literals.
        Returns list of (var, value) to assign, or empty if none.
        """
        pos_vars = set()
        neg_vars = set()
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                if lit > 0:
                    pos_vars.add(var)
                else:
                    neg_vars.add(var)
        
        pure = []
        for var in pos_vars:
            if var not in neg_vars and var not in assignment:
                pure.append((var, True))
        for var in neg_vars:
            if var not in pos_vars and var not in assignment:
                pure.append((var, False))
        return pure

    def select_variable(clauses, assignment):
        """
        Heuristic: pick variable in most clauses, or highest degree.
        """
        var_counts = {}
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                if var not in assignment:
                    var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            return None
        
        # Pick variable with max count
        max_count = max(var_counts.values())
        candidates = [v for v, c in var_counts.items() if c == max_count]
        return random.choice(candidates)

    def dpll(clauses, assignment, depth=0):
        # 1. Check for UNSAT (empty clause)
        for c in clauses:
            if len(c) == 0:
                return None
        
        # 2. Unit Propagation
        res = unit_propagate(clauses, assignment)
        if res[2] is False:
            return None
        clauses, assignment, _ = res
        
        # 3. Check for SAT (no clauses left)
        if not clauses:
            return assignment
        
        # 4. Pure Literal Elimination
        pure = pure_literal_elimination(clauses, assignment)
        if pure:
            for var, val in pure:
                assignment[var] = val
            
            # Filter clauses with these assignments
            new_clauses = []
            for c in clauses:
                unassigned_lits = []
                is_satisfied = False
                for lit in c:
                    var = abs(lit) - 1
                    if var in assignment:
                        val = assignment[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            is_satisfied = True
                            break
                    else:
                        unassigned_lits.append(lit)
                if not is_satisfied:
                    new_clauses.append(unassigned_lits)
            
            # Recurse
            return dpll(new_clauses, assignment, depth + 1)
        
        # 5. Select a variable
        var = select_variable(clauses, assignment)
        if var is None:
            # All variables assigned, but clauses remain? Should have been caught by unit prop or empty check
            # If we have clauses but no unassigned vars, they must be empty (UNSAT) or satisfied.
            # But unit_propagate handles empty. If clauses exist and no unassigned vars, 
            # it means we have a non-empty clause with only assigned literals that didn't satisfy?
            # That would mean it's UNSAT.
            return None

        # 6. Branch
        # Try True
        assignment[var] = True
        new_clauses = []
        for c in clauses:
            unassigned_lits = []
            is_satisfied = False
            for lit in c:
                v = abs(lit) - 1
                if v == var:
                    if lit > 0: # True satisfies positive
                        is_satisfied = True
                        break
                    else: # True makes negative false, so remove
                        continue
                elif v in assignment:
                    val = assignment[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        is_satisfied = True
                        break
                else:
                    unassigned_lits.append(lit)
            
            if not is_satisfied:
                if len(unassigned_lits) == 0:
                    return None # UNSAT in this branch
                new_clauses.append(unassigned_lits)
        
        res = dpll(new_clauses, assignment, depth + 1)
        if res is not None:
            return res
        
        # Backtrack
        del assignment[var]
        
        # Try False
        assignment[var] = False
        new_clauses = []
        for c in clauses:
            unassigned_lits = []
            is_satisfied = False
            for lit in c:
                v = abs(lit) - 1
                if v == var:
                    if lit < 0: # False satisfies negative
                        is_satisfied = True
                        break
                    else: # False makes positive false, so remove
                        continue
                elif v in assignment:
                    val = assignment[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        is_satisfied = True
                        break
                else:
                    unassigned_lits.append(lit)
            
            if not is_satisfied:
                if len(unassigned_lits) == 0:
                    return None # UNSAT in this branch
                new_clauses.append(unassigned_lits)
        
        res = dpll(new_clauses, assignment, depth + 1)
        if res is not None:
            return res
        
        del assignment[var]
        return None

    # Initialize
    assignment = {}
    
    # Run DPLL
    # To prevent infinite loops or extreme backtracking on hard instances (though planted are usually easy),
    # we could add a node limit, but for 50 vars, DPLL is robust.
    # We'll use a fixed seed for reproducibility if needed, but random helps escape bad branches.
    rng = random.Random(42)
    
    # The function above uses global random. Let's inject rng.
    # Actually, modifying the inner functions to use a specific rng is cleaner.
    # Let's rewrite the inner logic to accept rng.
    
    def unit_propagate_rg(clauses, assignment, rng):
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                unassigned_lits = []
                is_satisfied = False
                is_unsat = False
                for lit in c:
                    var = abs(lit) - 1
                    if var in assignment:
                        val = assignment[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            is_satisfied = True
                            break
                        else:
                            is_unsat = True
                            break
                    else:
                        unassigned_lits.append(lit)
                
                if is_unsat:
                    return None, None, False
                if is_satisfied:
                    continue
                if len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    var = abs(lit) - 1
                    val = lit > 0
                    assignment[var] = val
                    changed = True
                else:
                    new_clauses.append(unassigned_lits)
            
            clauses = new_clauses
        
        for c in clauses:
            if len(c) == 0:
                return None, None, False
        
        return clauses, assignment, True

    def pure_literal_elimination_rg(clauses, assignment, rng):
        pos_vars = set()
        neg_vars = set()
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                if lit > 0:
                    pos_vars.add(var)
                else:
                    neg_vars.add(var)
        
        pure = []
        for var in pos_vars:
            if var not in neg_vars and var not in assignment:
                pure.append((var, True))
        for var in neg_vars:
            if var not in pos_vars and var not in assignment:
                pure.append((var, False))
        return pure

    def select_variable_rg(clauses, assignment, rng):
        var_counts = {}
        for c in clauses:
            for lit in c:
                var = abs(lit) - 1
                if var not in assignment:
                    var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            return None
        
        max_count = max(var_counts.values())
        candidates = [v for v, c in var_counts.items() if c == max_count]
        return rng.choice(candidates)

    def dpll_rg(clauses, assignment, rng, depth=0):
        # Check for UNSAT
        for c in clauses:
            if len(c) == 0:
                return None
        
        # Unit Propagation
        res = unit_propagate_rg(clauses, assignment, rng)
        if res[2] is False:
            return None
        clauses, assignment, _ = res
        
        # Check for SAT
        if not clauses:
            return assignment
        
        # Pure Literal Elimination
        pure = pure_literal_elimination_rg(clauses, assignment, rng)
        if pure:
            for var, val in pure:
                assignment[var] = val
            
            new_clauses = []
            for c in clauses:
                unassigned_lits = []
                is_satisfied = False
                for lit in c:
                    var = abs(lit) - 1
                    if var in assignment:
                        val = assignment[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            is_satisfied = True
                            break
                    else:
                        unassigned_lits.append(lit)
                if not is_satisfied:
                    new_clauses.append(unassigned_lits)
            
            return dpll_rg(new_clauses, assignment, rng, depth + 1)
        
        # Select Variable
        var = select_variable_rg(clauses, assignment, rng)
        if var is None:
            return None

        # Branch True
        assignment[var] = True
        new_clauses = []
        valid = True
        for c in clauses:
            unassigned_lits = []
            is_satisfied = False
            for lit in c:
                v = abs(lit) - 1
                if v == var:
                    if lit > 0:
                        is_satisfied = True
                        break
                    else:
                        continue
                elif v in assignment:
                    val = assignment[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        is_satisfied = True
                        break
                else:
                    unassigned_lits.append(lit)
            
            if not is_satisfied:
                if len(unassigned_lits) == 0:
                    valid = False
                    break
                new_clauses.append(unassigned_lits)
        
        if valid:
            res = dpll_rg(new_clauses, assignment, rng, depth + 1)
            if res is not None:
                return res
        
        del assignment[var]
        
        # Branch False
        assignment[var] = False
        new_clauses = []
        valid = True
        for c in clauses:
            unassigned_lits = []
            is_satisfied = False
            for lit in c:
                v = abs(lit) - 1
                if v == var:
                    if lit < 0:
                        is_satisfied = True
                        break
                    else:
                        continue
                elif v in assignment:
                    val = assignment[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        is_satisfied = True
                        break
                else:
                    unassigned_lits.append(lit)
            
            if not is_satisfied:
                if len(unassigned_lits) == 0:
                    valid = False
                    break
                new_clauses.append(unassigned_lits)
        
        if valid:
            res = dpll_rg(new_clauses, assignment, rng, depth + 1)
            if res is not None:
                return res
        
        del assignment[var]
        return None

    # Start solving
    solution = dpll_rg(processed_clauses, {}, rng)
    
    if solution is None:
        # Fallback: if DPLL fails (should be rare for planted SAT), return random
        # Or try WalkSAT as backup?
        # For now, return all False or random
        return [False] * n
    
    # Convert dict to list
    result = [solution.get(i, False) for i in range(n)]
    return result
# EVOLVE-BLOCK-END
