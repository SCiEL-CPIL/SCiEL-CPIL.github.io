"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    def lit_to_internal(l):
        return abs(l) - 1, (1 if l > 0 else -1)

    processed_clauses = []
    for clause in clauses:
        seen_vars = set()
        internal_clause = []
        is_tautology = False
        for l in clause:
            v, s = lit_to_internal(l)
            if v in seen_vars:
                is_tautology = True
                break
            seen_vars.add(v)
            internal_clause.append((v, s))
        
        if not is_tautology and internal_clause:
            processed_clauses.append(internal_clause)
            
    if not processed_clauses:
        return [False] * n

    def dpll(clauses, assignment):
        # Unit Propagation
        while True:
            unit_clause_found = False
            new_clauses = []
            for c in clauses:
                sat = False
                unassigned = []
                for v, s in c:
                    if assignment[v] is not None:
                        if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                            sat = True
                            break
                    else:
                        unassigned.append((v, s))
                
                if sat:
                    continue
                
                if not unassigned:
                    return None
                
                if len(unassigned) == 1:
                    v, s = unassigned[0]
                    val = (s == 1)
                    if assignment[v] is not None:
                        if assignment[v] != val:
                            return None
                    else:
                        assignment[v] = val
                        unit_clause_found = True
                
                new_clauses.append(unassigned)
            
            if not unit_clause_found:
                break
            
            clauses = new_clauses
        
        if not clauses:
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = False
            return assignment
        
        # Pure Literal Elimination
        pos_vars = set()
        neg_vars = set()
        for c in clauses:
            for v, s in c:
                if s == 1:
                    pos_vars.add(v)
                else:
                    neg_vars.add(v)
        
        pure_found = False
        for v in pos_vars:
            if v not in neg_vars and assignment[v] is None:
                assignment[v] = True
                pure_found = True
        for v in neg_vars:
            if v not in pos_vars and assignment[v] is None:
                assignment[v] = False
                pure_found = True
        
        if pure_found:
            new_clauses = []
            for c in clauses:
                sat = False
                unassigned = []
                for v, s in c:
                    if assignment[v] is not None:
                        if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                            sat = True
                            break
                    else:
                        unassigned.append((v, s))
                if sat:
                    continue
                if not unassigned:
                    return None
                new_clauses.append(unassigned)
            clauses = new_clauses
            return dpll(clauses, assignment)
        
        # Branching
        shortest_clause = min(clauses, key=len)
        if not shortest_clause:
            return None
            
        var_counts = {}
        for c in clauses:
            for v, s in c:
                var_counts[v] = var_counts.get(v, 0) + 1
        
        branch_var = max(shortest_clause, key=lambda x: var_counts.get(x[0], 0))[0]
        
        # Copy state for first branch
        assign_1 = assignment[:]
        assign_1[branch_var] = True
        clauses_1 = [c[:] for c in clauses]
        
        result = dpll(clauses_1, assign_1)
        if result is not None:
            return result
        
        # Copy state for second branch
        assign_2 = assignment[:]
        assign_2[branch_var] = False
        clauses_2 = [c[:] for c in clauses]
        
        result = dpll(clauses_2, assign_2)
        if result is not None:
            return result
        
        return None

    initial_assignment = [None] * n
    result = dpll(processed_clauses, initial_assignment)
    
    if result is None:
        return [False] * n
    return result
# EVOLVE-BLOCK-END
