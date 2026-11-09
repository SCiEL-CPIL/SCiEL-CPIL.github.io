"""Find a small non-2-colorable 5-uniform hypergraph using SAT-based search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    def has_2_coloring(n, edges):
        """Check if hypergraph HAS a 2-coloring using DPLL SAT solver."""
        m = len(edges)
        if m == 0:
            return True
        
        edges_t = [tuple(e) for e in edges]
        
        # Build CNF: for each edge, two clauses
        # Clause 1: at least one vertex is color 1 (not all 0)
        # Clause 2: at least one vertex is color 0 (not all 1)
        clauses = []
        for edge in edges_t:
            clauses.append([(v, 1) for v in edge])  # Some vertex is 1
            clauses.append([(v, 0) for v in edge])  # Some vertex is 0
        
        # DPLL with unit propagation
        def dpll(assignments, clause_list):
            # assignments: dict vertex -> 0 or 1
            # Simplify and check for contradictions
            simplified = []
            for clause in clause_list:
                new_clause = []
                satisfied = False
                for (v, val) in clause:
                    if v in assignments:
                        if assignments[v] == val:
                            satisfied = True
                            break
                    else:
                        new_clause.append((v, val))
                if not satisfied:
                    if not new_clause:
                        return False  # Contradiction
                    simplified.append(new_clause)
            
            if not simplified:
                return True
            
            # Unit propagation
            changed = True
            while changed:
                changed = False
                new_simplified = []
                for clause in simplified:
                    if len(clause) == 1:
                        v, val = clause[0]
                        if v in assignments:
                            if assignments[v] != val:
                                return False
                        else:
                            assignments[v] = val
                            changed = True
                    else:
                        new_simplified.append(clause)
                simplified = new_simplified
            
            if not simplified:
                return True
            
            # Choose a variable from the first clause
            v = simplified[0][0][0]
            if v in assignments:
                # Should have been handled by propagation
                return dpll(assignments, simplified)
            
            # Try v = 0
            assignments[v] = 0
            if dpll(assignments, simplified):
                return True
            del assignments[v]
            
            # Try v = 1
            assignments[v] = 1
            if dpll(assignments, simplified):
                return True
            del assignments[v]
            
            return False
        
        return dpll({}, clauses)
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # Try to find smaller constructions
    for n in range(9, 14):
        if time.time() > deadline - 30:
            break
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Try random subsets
        for target_size in range(30, min(60, len(all_edges))):
            if time.time() > deadline - 10:
                break
            
            for trial in range(20):
                if time.time() > deadline - 5:
                    break
                
                sample = random.sample(all_edges, target_size)
                
                if not has_2_coloring(n, sample):
                    if target_size < best_count:
                        best_count = target_size
                        best_n = n
                        best_edges = [list(e) for e in sample]
                    break
        
        # Local search: try to remove edges
        if best_n == n and best_edges is not None:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(best_edges)):
                    if time.time() > deadline - 3:
                        break
                    test = best_edges[:i] + best_edges[i+1:]
                    if not has_2_coloring(n, test):
                        best_edges = test
                        best_count = len(best_edges)
                        improved = True
                        break
    
    return {'n': best_n, 'edges': best_edges}

