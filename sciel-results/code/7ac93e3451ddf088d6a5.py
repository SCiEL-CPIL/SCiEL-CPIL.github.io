"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _has_bicolored_cycle(coloring, n, edges):
    """Check if any pair of colors induces a cycle using union-find."""
    # Group vertices by color
    from collections import defaultdict
    color_groups = defaultdict(list)
    for v in range(n):
        color_groups[coloring[v]].append(v)
    
    # For each pair of colors, check if induced subgraph has a cycle
    colors_used = list(color_groups.keys())
    for i in range(len(colors_used)):
        for j in range(i+1, len(colors_used)):
            ci, cj = colors_used[i], colors_used[j]
            # Build induced subgraph on vertices with color ci or cj
            vertices_in_pair = set(color_groups[ci]) | set(color_groups[cj])
            # Count edges in induced subgraph
            edge_count = 0
            for u, v in edges:
                if u in vertices_in_pair and v in vertices_in_pair:
                    edge_count += 1
            # A forest on k vertices has at most k-1 edges (if connected) or fewer
            # Actually, a forest on k vertices has at most k - c edges where c is number of components
            # But simpler: if edge_count >= len(vertices_in_pair), there's a cycle
            # More precisely: a graph with V vertices and E edges has a cycle iff E >= V (for the induced subgraph)
            # Wait, that's not quite right. A forest has E <= V - 1 if connected, but in general E <= V - c where c >= 1
            # So E < V is necessary but not sufficient for being a forest (disconnected forests have E <= V - c)
            # Actually: a graph is a forest iff for every connected component, E_i <= V_i - 1
            # Equivalently: E <= V - c where c is number of components
            # Simplest check: E >= V means there's definitely a cycle (since even with 1 component, max edges in forest is V-1)
            # But E = V - 1 could still be a cycle if disconnected... no, if E = V-1 and it's a forest it must be connected (tree)
            # Actually the cleanest: use union-find to detect cycle
            if edge_count >= len(vertices_in_pair):
                # Definitely has a cycle
                return True
            # If edge_count < len(vertices_in_pair), it could still be a forest or have a cycle in a component
            # Let's do proper union-find for safety
            if edge_count > 0:
                # Union-find on the induced subgraph
                parent = {v: v for v in vertices_in_pair}
                rank = {v: 0 for v in vertices_in_pair}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False  # cycle detected
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                
                has_cycle = False
                for u, v in edges:
                    if u in vertices_in_pair and v in vertices_in_pair:
                        if not union(u, v):
                            has_cycle = True
                            break
                if has_cycle:
                    return True
    return False

def _validate(coloring, n, edges):
    """Check all constraints: adjacent vertices differ, no bicolored cycles."""
    # Check proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycles
    if _has_bicolored_cycle(coloring, n, edges):
        return False
    return True

def construct():
    deadline = time.time() + 80  # Leave 10s safety margin
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Start with a valid coloring using many colors (identity)
    best_coloring = list(range(n))
    best_count = n
    
    # Try to reduce colors using local search / backtracking
    # Strategy: try k colors for k = 3, 4, 5, ...
    # Use randomized backtracking with time limit
    
    def try_k_colors(k, time_limit):
        """Try to find valid coloring with exactly k colors."""
        start_time = time.time()
        coloring = [-1] * n
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        pos_in_order = [0] * n
        for i, v in enumerate(order):
            pos_in_order[v] = i
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(k):
                # Check if c conflicts with already-colored neighbors
                valid = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                coloring[v] = c
                # Quick check: for each pair of colors used so far involving c,
                # check if adding v creates a bicolored cycle
                # This is expensive, so only do it periodically or at the end
                if backtrack(idx + 1):
                    return True
                coloring[v] = -1
            return False
        
        # Use randomized restarts
        random.seed(42)
        for attempt in range(200):
            # Shuffle color order for diversity
            color_perm = list(range(k))
            random.shuffle(color_perm)
            # Temporarily modify the backtrack to use color_perm
            # Simpler: just try with different random seeds by shuffling vertex order slightly
            if attempt > 0:
                # Slightly perturb the vertex ordering
                for _ in range(3):
                    i = random.randint(0, n-2)
                    if random.random() < 0.3:
                        order[i], order[i+1] = order[i+1], order[i]
            
            if backtrack(0):
                # Verify
                if _validate(coloring, n, edges):
                    return coloring
        return None
    
    # Try increasing k
    for k in range(3, 10):
        if time.time() > deadline:
            break
        time_limit = min(15.0, max(1.0, (deadline - time.time()) / (10 - k)))
        result = try_k_colors(k, time_limit)
        if result is not None:
            best_coloring = result
            best_count = k
            break
    
    # If we didn't find a good one, try local search from identity
    if best_count == n and time.time() < deadline:
        # Greedy: try to recolor vertices to reduce color count
        coloring = list(range(n))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(n):
                # Try to assign a lower color to v
                current = coloring[v]
                for c in range(current):
                    # Check if c is valid
                    valid = True
                    for nb in adj[v]:
                        if coloring[nb] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check bicolored cycle constraint
                    old_color = coloring[v]
                    coloring[v] = c
                    if _validate(coloring, n, edges):
                        improved = True
                        break
                    else:
                        coloring[v] = old_color
        if _validate(coloring, n, edges):
            distinct = len(set(coloring))
            if distinct < best_count:
                best_count = distinct
                best_coloring = coloring
    
    # Final validation
    if not _validate(best_coloring, n, edges):
        # Fallback to identity
        best_coloring = list(range(n))
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
