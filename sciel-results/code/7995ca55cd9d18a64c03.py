import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges, timeout=0.5):
    start_time = time.time()
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
    edge_tuples = [tuple(e) for e in edges]
    
    def propagate(coloring):
        changed = True
        while changed:
            changed = False
            for edge in edge_tuples:
                c0 = 0
                c1 = 0
                unknown = -1
                for v in edge:
                    if coloring[v] == -1:
                        unknown = v
                    elif coloring[v] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                if c0 == 4 and unknown != -1:
                    if coloring[unknown] == -1:
                        coloring[unknown] = 1
                        changed = True
                    elif coloring[unknown] == 0:
                        return None
                elif c1 == 4 and unknown != -1:
                    if coloring[unknown] == -1:
                        coloring[unknown] = 0
                        changed = True
                    elif coloring[unknown] == 1:
                        return None
                if c0 == 5 or c1 == 5:
                    return None
        return coloring

    def backtrack(coloring):
        if time.time() - start_time > timeout:
            return None
        coloring = propagate(coloring)
        if coloring is None:
            return None
        unassigned = [v for v in range(n) if coloring[v] == -1]
        if not unassigned:
            return coloring
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        coloring[v] = 0
        result = backtrack(coloring)
        if result is not None:
            return result
        coloring[v] = 1
        result = backtrack(coloring)
        if result is not None:
            return result
        coloring[v] = -1
        return None

    coloring = [-1] * n
    result = backtrack(coloring)
    return result is not None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(best_n), 5)]
    best_m = len(best_edges)
    
    # Verify fallback
    if is_2_colorable_fast(best_n, best_edges, timeout=5):
        return {'n': best_n, 'edges': best_edges}
        
    random.seed(42)
    
    # Try different starting hypergraphs for local search
    starting_configs = [
        (8, [list(e) for e in combinations(range(8), 5)]), # 56 edges
        (9, [list(e) for e in combinations(range(9), 5)]), # 126 edges
        (10, None), # Random 80 edges
        (11, None), # Random 90 edges
    ]
    
    for n_start, edges_start in starting_configs:
        if time.time() > deadline:
            break
        
        if edges_start is None:
            # Generate random dense hypergraph
            all_edges = list(combinations(range(n_start), 5))
            random.shuffle(all_edges)
            # Pick a high density
            num_edges = int(len(all_edges) * 0.6)
            edges_start = [list(e) for e in all_edges[:num_edges]]
            
        # Check if start is non-2-colorable
        if is_2_colorable_fast(n_start, edges_start, timeout=2):
            continue # Skip if 2-colorable
            
        current_n = n_start
        current_edges = [list(e) for e in edges_start]
        
        # Local search: remove edges
        # We run this for a fixed number of steps or until time is up
        steps_limit = 200
        for step in range(steps_limit):
            if time.time() > deadline:
                break
                
            if len(current_edges) <= best_m:
                # If we are already better or equal, we can stop this branch 
                # unless we want to try to go even lower? 
                # If len == best_m, we can't improve edge count.
                # If len < best_m, we have a new best.
                if len(current_edges) < best_m:
                    best_m = len(current_edges)
                    best_n = current_n
                    best_edges = [list(e) for e in current_edges]
                # If we reached a minimal state (can't remove any), we stop
                # But we don't know if it's minimal without trying.
                # So we continue trying to remove.
                pass
            
            if len(current_edges) == 0:
                break
                
            idx = random.randint(0, len(current_edges) - 1)
            removed_edge = current_edges.pop(idx)
            
            if not is_2_colorable_fast(current_n, current_edges, timeout=0.05):
                # Keep removed
                if len(current_edges) < best_m:
                    best_m = len(current_edges)
                    best_n = current_n
                    best_edges = [list(e) for e in current_edges]
            else:
                current_edges.append(removed_edge)
                
    return {'n': best_n, 'edges': best_edges}
