import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 125
    
    def solve(n, seed):
        rng = random.Random(seed)
        
        # adj[i][j] = True if edge exists
        adj = np.zeros((n, n), dtype=bool)
        # common[i][j] = number of common neighbors of i and j (for i != j)
        common = np.zeros((n, n), dtype=np.int32)
        
        def add_edge(u, v):
            """Add edge (u,v) and update common counts. Returns False if it would create C4."""
            if u == v or adj[u][v]:
                return False
            # Adding (u,v): for any w, if w is neighbor of both u and v, common[u][v] increases
            # But more importantly, adding (u,v) affects common counts for pairs involving u or v
            # Actually, common[i][j] changes when we add edge (u,v):
            #   common[u][j] += adj[v][j] for all j != u,v (because v becomes a common neighbor of u and j)
            #   common[v][j] += adj[u][j] for all j != u,v
            #   common[u][v] = sum of adj[u][k] & adj[v][k] for k != u,v
            #   For other pairs (i,j) not involving u or v: no change
            # Wait, that's not right either. Let me think again.
            # common[i][j] = |N(i) ∩ N(j)|
            # If I add edge (u,v), then:
            #   v enters N(u), u enters N(v)
            #   For pair (u, j) where j != u,v: common[u][j] increases by 1 if v ∈ N(j), i.e., adj[v][j]
            #   For pair (v, j) where j != u,v: common[v][j] increases by 1 if u ∈ N(j), i.e., adj[u][j]
            #   For pair (u, v): common[u][v] = |N(u) ∩ N(v)| (after adding, but u∉N(u), v∉N(v))
            #   For pairs not involving u or v: unchanged
            #   For pair (i, j) where i,j != u,v: unchanged
            # So:
            #   common[u][v] = common[v][u] = sum(adj[u][k] & adj[v][k] for k in range(n) if k != u and k != v)
            #   For j in range(n) if j != u and j != v:
            #       common[u][j] += adj[v][j]
            #       common[v][j] += adj[u][j]
            #       common[j][u] += adj[v][j]
            #       common[j][v] += adj[u][j]
            
            # Check if adding creates C4: common[u][v] would be the new common neighbor count
            # We need common[u][v] <= 1 after adding
            new_common_uv = 0
            for k in range(n):
                if k != u and k != v and adj[u][k] and adj[v][k]:
                    new_common_uv += 1
                    if new_common_uv > 1:
                        return False
            
            # Also need to check that adding doesn't make any common[u][j] > 1 or common[v][j] > 1
            # Actually, we need ALL pairs to have common <= 1
            # So check: for all j != u,v, common[u][j] + adj[v][j] <= 1 and common[v][j] + adj[u][j] <= 1
            for j in range(n):
                if j == u or j == v:
                    continue
                if common[u][j] + adj[v][j] > 1:
                    return False
                if common[v][j] + adj[u][j] > 1:
                    return False
            
            # Safe to add
            adj[u][v] = adj[v][u] = True
            common[u][v] = common[v][u] = new_common_uv
            for j in range(n):
                if j == u or j == v:
                    continue
                common[u][j] += adj[v][j]
                common[j][u] += adj[v][j]
                common[v][j] += adj[u][j]
                common[j][v] += adj[u][j]
            return True
        
        def remove_edge(u, v):
            """Remove edge (u,v) and update common counts."""
            if not adj[u][v]:
                return
            adj[u][v] = adj[v][u] = False
            for j in range(n):
                if j == u or j == v:
                    continue
                common[u][j] -= adj[v][j]
                common[j][u] -= adj[v][j]
                common[v][j] -= adj[u][j]
                common[j][v] -= adj[u][j]
            # Recompute common[u][v]
            common[u][v] = common[v][u] = 0
            for k in range(n):
                if k != u and k != v and adj[u][k] and adj[v][k]:
                    common[u][v] += 1
        
        # Initialize: start with a random graph, but ensure C4-free from start
        # Strategy: add edges one by one, only if it doesn't create a C4
        # This gives us a maximal C4-free graph
        vertices = list(range(n))
        rng.shuffle(vertices)
        # Order edges randomly
        edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                edge_list.append((i, j))
        rng.shuffle(edge_list)
        
        for (u, v) in edge_list:
            if time.time() > deadline:
                return None
            add_edge(u, v)
        
        # Now we have a maximal C4-free graph. Check min degree.
        degs = [int(sum(adj[i])) for i in range(n)]
        md = min(degs)
        if md >= n - 39:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # Try local search: remove some edges to allow adding others to boost degree
        # Actually, the graph is already maximal C4-free, so we can't add any more edges.
        # We need a different approach.
        
        # Alternative: try different random orderings
        return None
    
    best_result = None
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline - 3:
            break
        
        for seed in range(200):
            if time.time() > deadline - 3:
                break
            result = solve(n, seed)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        best_result = {'n': 39, 'edges': []}
    
    return best_result
