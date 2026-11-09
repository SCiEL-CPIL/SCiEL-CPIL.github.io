import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to 0-based
    cnf = []
    for c in clauses:
        cnf.append([l - 1 if l > 0 else -l - 1 for l in c])
        
    deadline = time.time() + 290
    
    def dpll(assignment, clauses):
        if time.time() > deadline:
            return None
            
        new_clauses = []
        has_unit = False
        unit_literals = []
        
        for clause in clauses:
            unsat_lits = []
            sat = False
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    val = assignment[var]
                    is_true = (lit > 0 and val) or (lit < 0 and not val)
                    if is_true:
                        sat = True
                        break
                else:
                    unsat_lits.append(lit)
            
            if sat:
                continue
            if not unsat_lits:
                return None
            if len(unsat_lits) == 1:
                has_unit = True
                unit_literals.append(unsat_lits[0])
            else:
                new_clauses.append(unsat_lits)
                
        while has_unit:
            has_unit = False
            for lit in unit_literals:
                var = abs(lit)
                val = lit > 0
                if var in assignment:
                    if assignment[var] != val:
                        return None
                else:
                    assignment[var] = val
            
            temp_clauses = []
            new_units = []
            for clause in new_clauses:
                unsat_lits = []
                sat = False
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        val = assignment[var]
                        is_true = (lit > 0 and val) or (lit < 0 and not val)
                        if is_true:
                            sat = True
                            break
                    else:
                        unsat_lits.append(lit)
                        
                if sat:
                    continue
                if not unsat_lits:
                    return None
                if len(unsat_lits) == 1:
                    new_units.append(unsat_lits[0])
                else:
                    temp_clauses.append(unsat_lits)
                    
            new_clauses = temp_clauses
            if new_units:
                has_unit = True
                unit_literals = new_units
            else:
                break
                
        # Pure Literal Elimination
        var_polarity = {}
        for clause in new_clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    if var not in var_polarity:
                        var_polarity[var] = set()
                    var_polarity[var].add(lit > 0)
                    
        pure_vars = []
        for var, pols in var_polarity.items():
            if len(pols) == 1:
                pure_vars.append((var, list(pols)[0]))
                
        if pure_vars:
            for var, val in pure_vars:
                assignment[var] = val
                
            filtered_clauses = []
            for clause in new_clauses:
                is_sat = False
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        v = assignment[var]
                        if (lit > 0 and v) or (lit < 0 and not v):
                            is_sat = True
                            break
                if is_sat:
                    continue
                new_clause = []
                for lit in clause:
                    var = abs(lit)
                    if var not in assignment:
                        new_clause.append(lit)
                if not new_clause:
                    return None
                filtered_clauses.append(new_clause)
            new_clauses = filtered_clauses
            
        if not new_clauses:
            for i in range(n):
                if i not in assignment:
                    assignment[i] = False
            return assignment
            
        var_freq = {}
        for clause in new_clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    var_freq[var] = var_freq.get(var, 0) + 1
                    
        if not var_freq:
            for i in range(n):
                if i not in assignment:
                    assignment[i] = False
            return assignment
            
        branch_var = max(var_freq, key=var_freq.get)
        
        assignment[branch_var] = True
        result = dpll(dict(assignment), new_clauses[:])
        if result is not None:
            return result
            
        assignment[branch_var] = False
        result = dpll(dict(assignment), new_clauses[:])
        if result is not None:
            return result
            
        return None

    assignment = {}
    result = dpll(assignment, cnf)
    
    if result is not None:
        return [result.get(i, False) for i in range(n)]
        
    # Fallback: WalkSAT
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Fast WalkSAT
    clause_data = []
    for c in cnf:
        cd = []
        for l in c:
            var = abs(l)
            pos = l > 0
            cd.append((var, pos))
        clause_data.append(cd)
        
    def count_satisfied(assignment):
        count = 0
        for cd in clause_data:
            sat = False
            for var, pos in cd:
                if assignment[var] == pos:
                    sat = True
                    break
            if sat:
                count += 1
        return count
                
    cur_count = count_satisfied(assignment)
    max_iters = 10000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if cur_count == len(cnf):
            break
            
        unsat_indices = []
        for i, cd in enumerate(clause_data):
            sat = False
            for var, pos in cd:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                unsat_indices.append(i)
                
        if not unsat_indices:
            break
            
        clause_idx = rng.choice(unsat_indices)
        cd = clause_data[clause_idx]
        
        # Pick random literal
        var, pos = rng.choice(cd)
        assignment[var] = pos
        
        new_count = count_satisfied(assignment)
        if new_count > cur_count:
            cur_count = new_count
        else:
            # Random restart or continue? Just continue for simplicity in fallback
            pass
            
    return assignment
