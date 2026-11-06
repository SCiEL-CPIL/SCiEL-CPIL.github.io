import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 330s limit, use 325s to be safe
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Convert clauses to sets of literals for efficient manipulation
    # Literals are 1-based signed integers
    cnf = [set(c) for c in clauses]
    
    if not cnf:
        return [False] * n

    def dpll(cnf, assignment):
        """
        cnf: list of sets of literals
        assignment: dict {var: bool}
        Returns: assignment dict if satisfied, None if unsat or timeout
        """
        # Check time
        if time.time() > deadline:
            return None

        # 1. Process clauses: remove satisfied, check for conflicts
        new_cnf = []
        for clause in cnf:
            # Check if clause is satisfied
            is_satisfied = False
            remaining = set()
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                        is_satisfied = True
                        break
                else:
                    remaining.add(lit)
            
            if is_satisfied:
                continue # Clause is satisfied, drop it
                
            if not remaining:
                # Clause is not satisfied and has no unassigned literals
                # This is a conflict
                return None
                
            new_cnf.append(remaining)
        
        if not new_cnf:
            return assignment

        # 2. Unit Propagation
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                return None
            
            unit_clauses = []
            other_clauses = []
            
            for clause in new_cnf:
                if len(clause) == 1:
                    unit_clauses.append(clause)
                else:
                    other_clauses.append(clause)
            
            if not unit_clauses:
                break
                
            new_cnf = other_clauses
            
            for clause in unit_clauses:
                lit = list(clause)[0]
                var = abs(lit)
                val = (lit > 0)
                
                if var in assignment:
                    if assignment[var] != val:
                        return None
                    else:
                        continue
                        
                assignment[var] = val
                changed = True
                
                # Apply this assignment to remaining clauses
                filtered_cnf = []
                for c in new_cnf:
                    satisfied = False
                    new_c = set()
                    for l in c:
                        v = abs(l)
                        if v == var:
                            if (l > 0 and val) or (l < 0 and not val):
                                satisfied = True
                                break
                            else:
                                continue
                        else:
                            new_c.add(l)
                            
                    if satisfied:
                        continue
                    elif not new_c:
                        return None
                    else:
                        filtered_cnf.append(new_c)
                new_cnf = filtered_cnf
                
                if not new_cnf:
                    return assignment

        # 3. Pure Literal Elimination
        if time.time() > deadline:
            return None
            
        # Collect variables
        pos_vars = set()
        neg_vars = set()
        
        for clause in new_cnf:
            for lit in clause:
                if lit > 0:
                    pos_vars.add(lit)
                else:
                    neg_vars.add(-lit)
                    
        pure_vars = pos_vars - neg_vars
        pure_vars_neg = neg_vars - pos_vars
        
        if pure_vars or pure_vars_neg:
            # Assign all pure literals
            to_assign = []
            for var in pure_vars:
                if var not in assignment:
                    to_assign.append((var, True))
            for var in pure_vars_neg:
                if var not in assignment:
                    to_assign.append((var, False))
                        
            if to_assign:
                for var, val in to_assign:
                    assignment[var] = val
                    
                # Filter clauses
                filtered_cnf = []
                for c in new_cnf:
                    satisfied = False
                    new_c = set()
                    for l in c:
                        v = abs(l)
                        if v in assignment:
                            if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                                satisfied = True
                                break
                            else:
                                continue
                        else:
                            new_c.add(l)
                            
                    if satisfied:
                        continue
                    elif not new_c:
                        return None
                    else:
                        filtered_cnf.append(new_c)
                new_cnf = filtered_cnf
                
                if not new_cnf:
                    return assignment

        # 4. Branching
        if not new_cnf:
            return assignment
            
        # Select a variable to branch on
        # Heuristic: Pick variable from the shortest clause
        min_len = len(new_cnf[0])
        target_clause = new_cnf[0]
        for c in new_cnf:
            if len(c) < min_len:
                min_len = len(c)
                target_clause = c
                
        # Pick a variable from target_clause
        var = list(target_clause)[0]
        
        # Branch 1: var = True
        res = dpll(new_cnf, {**assignment, var: True})
        if res is not None:
            return res
            
        # Branch 2: var = False
        res = dpll(new_cnf, {**assignment, var: False})
        if res is not None:
            return res
            
        return None

    # Run DPLL
    initial_assignment = {}
    result = dpll(cnf, initial_assignment)
    
    if result is not None:
        final_assignment = [False] * n
        for var, val in result.items():
            final_assignment[var - 1] = val
        return final_assignment
    else:
        # Fallback: Random assignment
        return [random.random() < 0.5 for _ in range(n)]
