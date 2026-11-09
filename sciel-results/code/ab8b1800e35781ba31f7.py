import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    if not edges:
        return True
        
    # Precompute adjacency
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    # State
    colors = [-1] * n
    # For each edge, count 0s and 1s
    count_0 = [0] * len(edges)
    count_1 = [0] * len(edges)
    
    def propagate(queue):
        while queue:
            v = queue.pop()
            c = colors[v]
            if c == -1:
                continue
            for e_idx in vertex_edges[v]:
                if c == 0:
                    count_0[e_idx] += 1
                else:
                    count_1[e_idx] += 1
                    
                # Check edge e_idx
                if count_0[e_idx] > 0 and count_1[e_idx] > 0:
                    # Satisfied
                    continue
                elif count_0[e_idx] > 0:
                    # Has 0s, no 1s.
                    # Need at least one 1 in unassigned vertices of this edge.
                    # If only 1 unassigned, it must be 1.
                    unassigned = [u for u in edges[e_idx] if colors[u] == -1]
                    if len(unassigned) == 0:
                        return False
                    if len(unassigned) == 1:
                        u = unassigned[0]
                        if colors[u] == -1:
                            colors[u] = 1
                            queue.append(u)
                        elif colors[u] == 0:
                            return False
                elif count_1[e_idx] > 0:
                    # Has 1s, no 0s.
                    unassigned = [u for u in edges[e_idx] if colors[u] == -1]
                    if len(unassigned) == 0:
                        return False
                    if len(unassigned) == 1:
                        u = unassigned[0]
                        if colors[u] == -1:
                            colors[u] = 0
                            queue.append(u)
                        elif colors[u] == 1:
                            return False
        return True

    def backtrack():
        # Find unassigned vertex
        # Heuristic: pick vertex in edge with smallest number of unassigned vertices?
        # Or just first unassigned?
        # Let's pick vertex with highest degree in unassigned edges?
        
        # Simple: first unassigned
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
                
        if v == -1:
            # All assigned. Check if all edges are satisfied.
            for i in range(len(edges)):
                if not (count_0[i] > 0 and count_1[i] > 0):
                    return False
            return True
            
        # Try 0
        colors[v] = 0
        if propagate([v]):
            if backtrack():
                return True
        # Try 1
        colors[v] = 1
        if propagate([v]):
            if backtrack():
                return True
        colors[v] = -1
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_edges = None
    best_n = None
    best_m = float('inf')
    
    # Use a fixed seed for reproducibility in evaluation? No, random is fine.
    
    for n in [15, 16, 17, 14]:
        if time.time() > deadline:
            break
            
        max_m = len(list(combinations(range(n), 5)))
        
        # Try several random instances
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Start with a dense graph
            m = max_m // 2 
            # Randomize m slightly
            m = random.randint(max_m // 3, max_m)
            
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            edges = [list(e) for e in all_edges[:m]]
            
            if not _is_2_colorable(n, edges):
                # Non-2-colorable. Try to reduce.
                current_m = m
                # Local search: remove edges
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    indices = list(range(len(edges)))
                    random.shuffle(indices)
                    for i in indices:
                        if time.time() > deadline:
                            break
                        trial = edges[:i] + edges[i+1:]
                        if not _is_2_colorable(n, trial):
                            edges = trial
                            current_m -= 1
                            improved = True
                            break
                            
                if current_m < best_m:
                    best_m = current_m
                    best_edges = edges
                    best_n = n
    
    if best_edges is None:
        from itertools import combinations
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
