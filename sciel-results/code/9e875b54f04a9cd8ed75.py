"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges, timeout=5):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    start_time = time.time()
    edges_set = [tuple(sorted(e)) for e in edges]
    
    def backtrack(assign):
        if time.time() - start_time > timeout:
            return None  # timeout, assume colorable
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for e in edges_set:
                colors = [assign.get(v) for v in e]
                num_0 = sum(1 for c in colors if c == 0)
                num_1 = sum(1 for c in colors if c == 1)
                num_un = sum(1 for c in colors if c is None)
                
                if num_0 >= 5 or num_1 >= 5:
                    return False
                if num_un == 0:
                    continue
                if num_0 == 4 and num_un == 1:
                    for v in e:
                        if v not in assign:
                            assign[v] = 1
                            changed = True
                elif num_1 == 4 and num_un == 1:
                    for v in e:
                        if v not in assign:
                            assign[v] = 0
                            changed = True
        
        unassigned = [v for v in range(n) if v not in assign]
        if not unassigned:
            for e in edges_set:
                colors = [assign[v] for v in e]
                if len(set(colors)) < 2:
                    return False
            return True
        
        v = unassigned[0]
        assign[v] = 0
        result = backtrack(assign)
        if result is not None:
            return result
        del assign[v]
        
        assign[v] = 1
        result = backtrack(assign)
        if result is not None:
            return result
        del assign[v]
        
        return None
    
    result = backtrack({})
    if result is None and time.time() - start_time <= timeout:
        return True
    return False

def construct():
    deadline = time.time() + 130  # safety margin
    
    # Start with complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = list(combinations(range(n), 5))
    current_edges = set(all_edges)
    
    # Verify it's non-2-colorable
    if not _is_not_2_colorable(n, [list(e) for e in current_edges], timeout=10):
        return {'n': n, 'edges': [list(e) for e in current_edges]}
    
    # Greedy edge removal: try removing edges one by one
    random.seed(42)
    edge_list = list(current_edges)
    random.shuffle(edge_list)
    
    for edge in edge_list:
        if time.time() > deadline:
            break
        if len(current_edges) <= 51:
            break
        
        # Try removing this edge
        test_edges = current_edges - {edge}
        if _is_not_2_colorable(n, [list(e) for e in test_edges], timeout=3):
            current_edges = test_edges
    
    # Try more rounds with different orderings
    for round_num in range(5):
        if time.time() > deadline:
            break
        if len(current_edges) <= 51:
            break
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        for edge in edge_list:
            if time.time() > deadline:
                break
            if len(current_edges) <= 51:
                break
            test_edges = current_edges - {edge}
            if _is_not_2_colorable(n, [list(e) for e in test_edges], timeout=3):
                current_edges = test_edges
    
    return {'n': n, 'edges': [list(e) for e in current_edges]}
# EVOLVE-BLOCK-END
