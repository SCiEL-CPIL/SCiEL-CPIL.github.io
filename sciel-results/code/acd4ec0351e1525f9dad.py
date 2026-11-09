import random
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edge_order = P['edge_order']
    
    # 1. Precompute edge indices
    # idx[u][v] = index (1-based) of edge (u,v) in edge_order
    idx = [[0] * n for _ in range(n)]
    for i, (u, v) in enumerate(edge_order):
        idx[u][v] = i + 1
        idx[v][u] = i + 1

    def get_cost(cycle):
        # cycle is a list of n vertices
        max_idx = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if idx[u][v] > max_idx:
                max_idx = idx[u][v]
        return max_idx

    def greedy_cycle():
        # Try to build a cycle by adding edges in order of index
        parent = list(range(n))
        rank = [0] * n
        deg = [0] * n
        edges = []
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        for i, (u, v) in enumerate(edge_order):
            if len(edges) == n:
                break
            if deg[u] >= 2 or deg[v] >= 2:
                continue
            
            pu, pv = find(u), find(v)
            if pu == pv:
                # Adding this edge creates a cycle
                # Check if it's the final edge (closes the big cycle)
                # This happens if the component size is n
                # We don't track size explicitly, but if we've added n-1 edges 
                # and this is the nth, and it connects the two ends?
                # Actually, simpler: if we have n-1 edges, and this edge connects 
                # two vertices in the same component, it must be the closing edge 
                # IF the component is the whole graph.
                # But we don't know if it's the whole graph without size tracking.
                # Let's add size tracking.
                pass
            else:
                # Safe to add, no cycle created
                union(u, v)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        
        # The simple greedy above fails to close the cycle because it rejects 
        # the edge that creates the cycle. 
        # We need to allow the final edge if it closes the loop on all n vertices.
        
        # Revised Greedy:
        parent = list(range(n))
        size = [1] * n
        deg = [0] * n
        edges = []
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if size[px] < size[py]:
                px, py = py, px
            parent[py] = px
            size[px] += size[py]
            return True

        for i, (u, v) in enumerate(edge_order):
            if len(edges) == n:
                break
            if deg[u] >= 2 or deg[v] >= 2:
                continue
            
            pu, pv = find(u), find(v)
            if pu == pv:
                # Cycle created
                if size[pu] == n and len(edges) == n - 1:
                    # This is the closing edge
                    edges.append((u, v))
                    deg[u] += 1
                    deg[v] += 1
                    break
                else:
                    # Sub-cycle, reject
                    continue
            else:
                union(u, v)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        
        if len(edges) != n:
            return None
        
        # Reconstruct cycle from edges
        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # All vertices should have degree 2
        # Find a starting node and traverse
        start = 0
        cycle = [start]
        prev = -1
        curr = start
        for _ in range(n):
            nexts = adj[curr]
            # Pick the one that is not prev
            if nexts[0] == prev:
                nxt = nexts[1]
            else:
                nxt = nexts[0]
            cycle.append(nxt)
            prev = curr
            curr = nxt
            
        # Check if it's a valid cycle (first == last)
        if cycle[0] != cycle[-1]:
            return None
        
        # Remove the last element which is a duplicate of the first
        return cycle[:-1]

    def two_opt(cycle):
        # cycle is a list of n vertices (no closing edge repeated)
        current_cost = get_cost(cycle)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() - start_time > 55:
                    break
                u = cycle[i]
                v = cycle[(i + 1) % n]
                for j in range(i + 2, n):
                    if time.time() - start_time > 55:
                        break
                    x = cycle[j]
                    y = cycle[(j + 1) % n]
                    
                    # Try reversing segment i+1 to j
                    # New edges: (u, x) and (v, y)
                    # Old edges: (u, v) and (x, y)
                    new_idx1 = idx[u][x]
                    new_idx2 = idx[v][y]
                    
                    # We only care if the max decreases
                    # The new max will be max(old_max_minus_removed, new_idx1, new_idx2)
                    # But calculating old_max_minus_removed is expensive.
                    # Just calculate new cost if potential is good.
                    # A quick check: if max(new_idx1, new_idx2) >= current_cost, 
                    # it might still be better if it removes a very high edge.
                    # But if max(new_idx1, new_idx2) > current_cost, it's definitely worse.
                    if max(new_idx1, new_idx2) > current_cost:
                        continue
                    
                    # Perform the move
                    new_cycle = cycle[:i+1] + cycle[j+1:i:-1] + cycle[j+1:]
                    # Wait, reversing segment i+1..j
                    # cycle: ... u, v, ..., x, y ...
                    # Reverse v..x
                    # segment = cycle[i+1 : j+1]
                    # reversed_segment = segment[::-1]
                    # new_cycle = cycle[:i+1] + reversed_segment + cycle[j+1:]
                    
                    seg = cycle[i+1 : j+1]
                    seg_rev = seg[::-1]
                    new_cycle = cycle[:i+1] + seg_rev + cycle[j+1:]
                    
                    new_cost = get_cost(new_cycle)
                    if new_cost < current_cost:
                        cycle = new_cycle
                        current_cost = new_cost
                        improved = True
                        # Restart inner loop or just continue? 
                        # Usually restart or break to re-evaluate.
                        # Breaking inner and outer to restart full scan is safer.
                        break
                if improved:
                    break
        return cycle, current_cost

    start_time = time.time()
    best_cycle = None
    best_cost = float('inf')
    
    # 1. Try Greedy
    g_cycle = greedy_cycle()
    if g_cycle:
        g_cost = get_cost(g_cycle)
        best_cycle = g_cycle
        best_cost = g_cost
    
    # 2. Try Random Starts if Greedy failed or to improve
    # We can run 2-opt on the greedy result
    if best_cycle:
        polished, cost = two_opt(best_cycle)
        if cost < best_cost:
            best_cycle = polished
            best_cost = cost
    
    # 3. If we have time, try a few random restarts
    # n=24, 2-opt is fast.
    # Let's do a few random permutations.
    rng = random.Random(42)
    
    max_restarts = 10
    for _ in range(max_restarts):
        if time.time() - start_time > 50:
            break
            
        rand_cycle = list(range(n))
        rng.shuffle(rand_cycle)
        
        # Simple 2-opt on random
        polished, cost = two_opt(rand_cycle)
        
        if cost < best_cost:
            best_cost = cost
            best_cycle = polished
            
    # Final check
    if best_cycle is None:
        # Fallback to identity
        return list(range(n))
        
    return best_cycle

# EVOLVE-BLOCK-END
