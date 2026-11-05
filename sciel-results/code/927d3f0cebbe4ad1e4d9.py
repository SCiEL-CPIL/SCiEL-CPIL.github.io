"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def _check_unsat(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring using DPLL with unit propagation.
    """
    clauses = []
    for edge in edges:
        clauses.append([(v, 1) for v in edge])
        clauses.append([(v, -1) for v in edge])
    
    def dpll(clauses, assignment):
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                satisfied = False
                remaining = []
                for (var, sign) in clause:
                    if var in assignment:
                        val = assignment[var]
                        if (val and sign == 1) or (not val and sign == -1):
                            satisfied = True
                            break
                    else:
                        remaining.append((var, sign))
                if satisfied:
                    continue
                if not remaining:
                    return None
                if len(remaining) == 1:
                    var, sign = remaining[0]
                    if var in assignment:
                        val = assignment[var]
                        if (val and sign == 1) or (not val and sign == -1):
                            continue
                        else:
                            return None
                    assignment[var] = (sign == 1)
                    changed = True
                else:
                    new_clauses.append(remaining)
            clauses = new_clauses
        
        if not clauses:
            return assignment
        
        var_count = {}
        for clause in clauses:
            for (var, sign) in clause:
                var_count[var] = var_count.get(var, 0) + 1
        if not var_count:
            return assignment
        
        var = max(var_count, key=lambda v: var_count[v])
        
        assignment[var] = True
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        assignment[var] = False
        result = dpll([c[:] for c in clauses], dict(assignment))
        if result is not None:
            return result
        del assignment[var]
        
        return None
    
    result = dpll(clauses, {})
    return result is None

def construct():
    best_result = None
    best_edges_count = float('inf')
    
    random.seed(42)
    
    for n in [9, 10, 11]:
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        # Start with complete graph for n=9, dense random for others
        if n == 9:
            initial_edges = all_edges
        else:
            num_edges = int(len(all_edges) * 0.5)
            random.shuffle(all_edges)
            initial_edges = all_edges[:num_edges]
            
        if not _check_unsat(n, initial_edges):
            if n <= 10:
                initial_edges = all_edges
                if not _check_unsat(n, initial_edges):
                    continue
            else:
                continue
                
        edges_set = set(tuple(e) for e in initial_edges)
        current_edges = set(edges_set)
        
        edges_to_try = list(edges_set)
        random.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if e not in current_edges:
                continue
            
            test_edges = current_edges - {e}
            test_list = [list(x) for x in test_edges]
            
            if _check_unsat(n, test_list):
                current_edges = test_edges
                
        if len(current_edges) < best_edges_count:
            best_edges_count = len(current_edges)
            best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
            
        if best_edges_count <= 51:
            break
            
    if best_result is not None:
        return best_result
        
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
