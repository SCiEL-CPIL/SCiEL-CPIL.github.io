import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_valid(coloring):
    """Check if coloring is valid: adjacent differ, and no bicolored cycles."""
    n = P['n']
    edges = P['edges']
    
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
            
    colors_used = set(coloring)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            parent = list(range(n))
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            has_cycle = False
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if union(u, v):
                        has_cycle = True
                        break
            if has_cycle:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute edge indices for fast access
    edge_u = [e[0] for e in edges]
    edge_v = [e[1] for e in edges]
    edge_idx_by_node = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        edge_idx_by_node[u].append(i)
        edge_idx_by_node[v].append(i)

    def solve(max_colors):
        coloring = [-1] * n
        # Union-find structures for each color pair
        # We only need to track pairs where both colors are used
        # To simplify, we can use a dictionary mapping (c1, c2) -> UnionFind
        # But creating/destroying UF structures is expensive.
        # Alternative: Since max_colors is small (try 2, 3, 4...), we can pre-allocate
        # or use a simpler cycle detection for the incremental step.
        
        # Incremental check: 
        # When we assign color c to node v, we add edges (v, u) for all neighbors u.
        # These edges are added to the subgraph of pair (c, color[u]) if color[u] is assigned.
        # We need to check if adding these edges creates a cycle in that specific pair's subgraph.
        
        # For efficiency, let's use a different approach for the search itself:
        # Backtracking with forward checking on the bicolored cycle constraint is hard to do incrementally 
        # without maintaining state. 
        # However, we can check validity of the *partial* assignment's implications.
        
        # Let's stick to a robust backtracking but with a very fast local check.
        # The key insight from the failure of the previous program is likely that it didn't find a solution 
        # with few colors quickly, or the search space was too large.
        
        # Let's try a randomized restart strategy with a fixed number of steps per restart.
        
        pass

    # Strategy: 
    # 1. Try to find a coloring with k colors for k=2,3,4...
    # 2. Use randomized greedy with local search (Simulated Annealing or Simple Hill Climbing) 
    #    to minimize the number of bicolored cycles, then try to eliminate them.
    #    
    # Actually, a simpler and often effective approach for small graphs is 
    # **Randomized Backtracking with Restart**.
    
    def random_backtrack(max_colors, time_limit):
        coloring = [-1] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Order vertices by degree descending to constrain early
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        # Map original index to order index and vice versa
        pos_in_order = [0]*n
        for i, v in enumerate(order):
            pos_in_order[v] = i
            
        def check_partial_validity(v, c):
            # Check if assigning c to v violates constraints with already colored neighbors
            # 1. Adjacency: no neighbor has color c
            for u in adj[v]:
                if coloring[u] == c:
                    return False
            
            # 2. Bicolored cycle check: 
            # This is the hard part. We need to know if adding v with color c 
            # creates a cycle in any (c, cu) pair where cu is a neighbor's color.
            # We can do a local BFS/DFS in the subgraph induced by {v} U {neighbors with color cu}
            # to see if there's a path between two neighbors of v that both have color cu?
            # No, a cycle in the induced subgraph of {c, cu} involving v would mean 
            # there is a path between two neighbors of v (both color cu) in the subgraph 
            # induced by {c, cu} that does not use v.
            # Wait, the induced subgraph includes ALL vertices with color c or cu.
            # So if we assign v=c, we add v to the {c, cu} subgraph.
            # A cycle is formed if v connects two vertices in the same component of the 
            # existing {c, cu} subgraph (excluding v).
            
            # Let's precompute components for each color pair? No, it changes.
            # Let's just run a quick union-find or BFS for the specific pairs involved.
            
            involved_colors = set()
            for u in adj[v]:
                if coloring[u] != -1:
                    involved_colors.add(coloring[u])
                    
            for cu in involved_colors:
                # Check pair (c, cu)
                # Build the subgraph induced by vertices with color c or cu
                # Check if adding v creates a cycle.
                # This is equivalent to: are any two neighbors of v (with color cu) 
                # connected in the subgraph induced by {c, cu} excluding v?
                # Or more generally, does the subgraph induced by {c, cu} containing v 
                # have a cycle?
                
                # Efficient check:
                # Take all neighbors of v with color cu. Let this set be N_cu.
                # If |N_cu| >= 2, check if any two are connected in the {c, cu} subgraph 
                # without using v.
                # Actually, simpler: 
                # Run BFS from one neighbor in N_cu, restricted to nodes with color c or cu, 
                # and see if we reach another neighbor in N_cu.
                
                neighbors_with_cu = [u for u in adj[v] if coloring[u] == cu]
                if len(neighbors_with_cu) < 2:
                    continue
                
                # BFS from neighbors_with_cu[0]
                start = neighbors_with_cu[0]
                visited = {start}
                queue = [start]
                found_cycle = False
                while queue:
                    curr = queue.pop(0)
                    if curr == v:
                        continue # v is not yet fully integrated, we are checking connectivity of its neighbors
                    for nb in adj[curr]:
                        if coloring[nb] in (c, cu):
                            if nb in visited:
                                # If we hit another neighbor of v with color cu, we have a cycle
                                # (v - start - ... - nb - v)
                                if nb in neighbors_with_cu and nb != start:
                                    found_cycle = True
                                    break
                            else:
                                visited.add(nb)
                                queue.append(nb)
                    if found_cycle:
                        break
                if found_cycle:
                    return False
            return True

        def backtrack(idx):
            if time.time() > time_limit:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            # Try colors in random order to diversify
            colors = list(range(max_colors))
            import random
            random.shuffle(colors)
            
            for c in colors:
                if check_partial_validity(v, c):
                    coloring[v] = c
                    if backtrack(idx + 1):
                        return True
                    coloring[v] = -1
            return False

        return backtrack(0)

    # Try increasing number of colors
    for k in range(2, 15):
        if time.time() > deadline - 5:
            break
        # Try a few random restarts for this k
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Seed random for reproducibility if needed, but random is fine
            import random
            random.seed(attempt * 1000 + k)
            if random_backtrack(k, deadline - 5):
                found = True
                break
        if found:
            # Verify full validity just in case
            if _check_valid(coloring):
                return [int(c) for c in coloring]

    # Fallback: Greedy with many colors
    coloring = [-1] * n
    for v in range(n):
        used = set(coloring[u] for u in adj[v] if coloring[u] != -1)
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    
    if _check_valid(coloring):
        return [int(c) for c in coloring]
        
    return [int(i) for i in range(n)]

if __name__ == '__main__':
    # Quick self-test
    res = construct()
    print(f"Colors used: {len(set(res))}")
    print(f"Valid: {_check_valid(res)}")
