import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices to potentially improve search
    # Simple order 0..n-1
    order = list(range(n))
    
    def is_valid_coloring(colors):
        # Check proper coloring
        for u in range(n):
            for v in adj[u]:
                if colors[u] == colors[v]:
                    return False
        # Check no bicolored cycles
        # For each pair of colors, check if induced subgraph is a forest
        # We can just check if there are any cycles in the whole graph that are bicolored?
        # No, we check for each pair (c1, c2) if G[V_c1 U V_c2] has a cycle.
        # Efficient way:
        # For each cycle in the graph, check if it uses <= 2 colors.
        # But finding all cycles is hard.
        # Alternative: For each pair (c1, c2), build subgraph and check for cycles.
        # Since n=50, this is fast.
        colors_used = set(colors)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                # Build subgraph induced by c1 and c2
                # Vertices: [v for v in range(n) if colors[v] in (c1, c2)]
                # Edges: (u, v) in P['edges'] if colors[u] in (c1, c2) and colors[v] in (c1, c2)
                # Check if this subgraph has a cycle.
                # Use Union-Find.
                uf = list(range(n))
                def find(x):
                    while uf[x] != x:
                        uf[x] = uf[uf[x]]
                        x = uf[x]
                    return x
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False # Cycle
                    uf[rx] = ry
                    return True
                
                for u, v in P['edges']:
                    if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                        if not union(u, v):
                            return False
        return True

    def backtrack(idx, colors, k, start_time):
        if time.time() - start_time > 50:
            return None
        if idx == n:
            if is_valid_coloring(colors):
                return colors[:]
            return None
        
        v = order[idx]
        # Try colors
        for c in range(k):
            # Check proper coloring with neighbors
            valid = True
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[v] = c
            # Incremental check? 
            # For simplicity, we can just check at the end, but that's slow.
            # Let's do a quick check for cycles involving v.
            # If adding v creates a cycle in any pair (c, c'), fail.
            # We can check this incrementally.
            # For each c' != c:
            #   Check if any two neighbors of v with colors in {c, c'} are connected in the subgraph induced by {c, c'} \ {v}.
            # This is the BFS check described earlier.
            
            # Let's implement the incremental check.
            incremental_ok = True
            for c_prime in range(k):
                if c_prime == c:
                    continue
                # Get neighbors of v with colors c or c_prime
                neigh = [u for u in adj[v] if colors[u] in (c, c_prime)]
                if len(neigh) >= 2:
                    # Check if any two are connected in the subgraph induced by {c, c_prime} excluding v
                    # We do BFS from neigh[0] to see if we reach any other in neigh
                    # Subgraph vertices: {u | colors[u] in (c, c_prime) and u != v}
                    # Edges: (u, w) in P['edges'] if u, w in subgraph vertices
                    visited = set()
                    queue = [neigh[0]]
                    visited.add(neigh[0])
                    found_cycle = False
                    while queue:
                        curr = queue.pop(0)
                        if curr in neigh and curr != neigh[0]:
                            found_cycle = True
                            break
                        for w in adj[curr]:
                            if w == v:
                                continue
                            if colors[w] in (c, c_prime) and w not in visited:
                                visited.add(w)
                                queue.append(w)
                    if found_cycle:
                        incremental_ok = False
                        break
            
            if incremental_ok:
                res = backtrack(idx + 1, colors, k, start_time)
                if res is not None:
                    return res
            colors[v] = -1
        return None

    start_time = time.time()
    
    # Try k=3
    colors = [-1] * n
    res = backtrack(0, colors, 3, start_time)
    if res is not None:
        return res
        
    # Try k=4
    colors = [-1] * n
    res = backtrack(0, colors, 4, start_time)
    if res is not None:
        return res

    # Fallback: Greedy or random
    # Just return a valid coloring with more colors if needed
    # For safety, return a simple valid coloring
    # e.g. proper coloring with 4 colors (greedy)
    # But we need to ensure no bicolored cycles.
    # If backtracking failed, it might be due to time limit.
    # We should return the best we have.
    # Since we don't have a partial result, we can try a heuristic.
    # But 50 vertices is small. Backtracking should work.
    
    # If still None, return a dummy valid coloring (e.g. all distinct)
    return list(range(n))
