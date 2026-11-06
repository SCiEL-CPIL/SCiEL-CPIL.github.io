"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges, timeout=10):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    start_time = time.time()
    edges_set = [tuple(sorted(e)) for e in edges]
    
    # For each edge, at least one vertex must be in each color
    # A proper 2-coloring means every edge has both colors present
    # So NOT 2-colorable means every 2-coloring has some edge with all same color
    
    # Backtracking: assign colors 0/1 to vertices, check all edges have both colors
    def backtrack(assign, edge_constraints):
        if time.time() - start_time > timeout:
            return None  # timeout, assume colorable
        
        # Unit propagation: if an edge has 4 vertices same color, 5th must be opposite
        changed = True
        while changed:
            changed = False
            for ei, e in enumerate(edges_set):
                colors = [assign.get(v) for v in e]
                num_0 = sum(1 for c in colors if c == 0)
                num_1 = sum(1 for c in colors if c == 1)
                num_un = sum(1 for c in colors if c is None)
                
                if num_0 >= 5:
                    return False  # all same color, edge violated
                if num_1 >= 5:
                    return False
                if num_un == 0:
                    # Edge is satisfied (has both colors)
                    continue
                if num_0 == 4 and num_un == 1:
                    # 5th must be 1
                    for v in e:
                        if assign.get(v) is None:
                            if v not in assign:
                                assign[v] = 1
                                changed = True
                elif num_1 == 4 and num_un == 1:
                    # 5th must be 0
                    for v in e:
                        if assign.get(v) is None:
                            if v not in assign:
                                assign[v] = 0
                                changed = True
        
        # Check if all vertices assigned
        unassigned = [v for v in range(n) if v not in assign]
        if not unassigned:
            # Verify all edges have both colors
            for e in edges_set:
                colors = [assign[v] for v in e]
                if len(set(colors)) < 2:
                    return False
            return True  # valid coloring found
        
        # Pick next vertex
        v = unassigned[0]
        
        # Try color 0
        assign[v] = 0
        result = backtrack(assign, edge_constraints)
        if result is not None:
            return result
        del assign[v]
        
        # Try color 1
        assign[v] = 1
        result = backtrack(assign, edge_constraints)
        if result is not None:
            return result
        del assign[v]
        
        return None  # no valid coloring from this state
    
    result = backtrack({}, None)
    if result is None and time.time() - start_time <= timeout:
        return True  # no coloring found, so not 2-colorable
    return False  # found coloring or timeout

def _find_small_non_2colorable(max_edges=60, n_start=8, n_end=15, timeout_per=5):
    """Search for small non-2-colorable 5-uniform hypergraphs."""
    best = None
    start_time = time.time()
    
    for n in range(n_start, n_end + 1):
        if time.time() - start_time > 90:
            break
        max_possible = len(list(combinations(range(n), 5)))
        for e in range(29, min(max_edges + 1, max_possible + 1)):
            if time.time() - start_time > 90:
                break
            # Try random subsets
            all_edges = list(combinations(range(n), 5))
            attempts = 0
            while attempts < 50 and time.time() - start_time < 90:
                random.shuffle(all_edges)
                subset = all_edges[:e]
                if _is_not_2_colorable(n, [list(x) for x in subset], timeout_per):
                    if best is None or e < len(best['edges']):
                        best = {'n': n, 'edges': [list(x) for x in subset]}
                    break
                attempts += 1
        if best is not None and len(best['edges']) <= 35:
            break
    
    return best

def construct():
    # First, try to find a small non-2-colorable hypergraph
    result = _find_small_non_2colorable(max_edges=60, n_start=8, n_end=14, timeout_per=3)
    
    if result is not None:
        return result
    
    # Fallback: complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
