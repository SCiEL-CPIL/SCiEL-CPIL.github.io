"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, store its neighbors
    neighbors = {i: adj[i] for i in range(n)}
    
    def check_all_forests(colors, k):
        """Check if all pairs of colors induce forests. Returns (is_valid, violating_pairs)."""
        violating = []
        for i in range(k):
            for j in range(i+1, k):
                # Union-Find for the subgraph induced by colors i and j
                parent = list(range(n))
                rank = [0] * n
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                
                for u, v in edges:
                    if (colors[u] == i and colors[v] == j) or (colors[u] == j and colors[v] == i):
                        if not union(u, v):
                            violating.append((i, j))
                            break
        return (len(violating) == 0, violating)
    
    def count_violations(colors, k):
        """Count the number of violating color pairs."""
        count = 0
        for i in range(k):
            for j in range(i+1, k):
                parent = list(range(n))
                rank = [0] * n
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                
                has_cycle = False
                for u, v in edges:
                    if (colors[u] == i and colors[v] == j) or (colors[u] == j and colors[v] == i):
                        if not union(u, v):
                            has_cycle = True
                            break
                if has_cycle:
                    count += 1
        return count
    
    def is_proper(colors, k):
        """Check if the coloring is proper (adjacent vertices have different colors)."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        return True
    
    # Try to find a valid coloring with k colors using backtracking
    def backtrack_k(k, time_limit=2.0):
        bt_start = time.time()
        bt_deadline = bt_start + time_limit
        
        colors = [-1] * n
        # Order by degree descending for better pruning
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        # For incremental checking, we'll track which color pairs are "active" 
        # (i.e., have at least one edge in the current partial assignment)
        
        def is_valid_partial(idx):
            """Check if the partial assignment up to idx is valid (no bicolored cycles yet)."""
            if time.time() > bt_deadline:
                return False, "timeout"
            
            # Check all pairs of colors used so far
            used_colors = set(colors[:idx])
            for i in used_colors:
                for j in used_colors:
                    if i >= j:
                        continue
                    # Check if subgraph induced by colors i and j (among assigned vertices) is a forest
                    parent = list(range(n))
                    rank = [0] * n
                    
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    
                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx == ry:
                            return False
                        if rank[rx] < rank[ry]:
                            rx, ry = ry, rx
                        parent[ry] = rx
                        if rank[rx] == rank[ry]:
                            rank[rx] += 1
                        return True
                    
                    for u, v in edges:
                        if u < idx and v < idx:
                            if (colors[u] == i and colors[v] == j) or (colors[u] == j and colors[v] == i):
                                if not union(u, v):
                                    return False, "cycle"
            return True, "ok"
        
        def bt(idx):
            if time.time() > bt_deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            
            for c in range(k):
                # Proper coloring check
                valid = True
                for nb in neighbors[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[v] = c
                
                # Check if this creates a bicolored cycle
                ok, reason = is_valid_partial(idx + 1)
                if ok:
                    if bt(idx + 1):
                        return True
                
                colors[v] = -1
            
            return False
        
        if bt(0):
            return [int(c) for c in colors]
        return None
    
    # First try 3 colors with backtracking
    result = backtrack_k(3, time_limit=3.0)
    if result is not None:
        return result
    
    # Try 4 colors with backtracking
    result = backtrack_k(4, time_limit=5.0)
    if result is not None:
        return result
    
    # Fallback: Local search with 4 colors
    # Initialize with a proper 4-coloring
    colors = [random.randint(0, 3) for _ in range(n)]
    
    # Make it proper
    for _ in range(1000):
        proper = True
        for u, v in edges:
            if colors[u] == colors[v]:
                proper = False
                # Try to fix u
                forbidden = set(colors[nb] for nb in neighbors[u])
                for c in range(4):
                    if c not in forbidden:
                        colors[u] = c
                        break
                if not proper:
                    break
        if proper:
            break
    
    # Now use local search to eliminate bicolored cycles
    k = 4
    curr_viol = count_violations(colors, k)
    
    # If still not proper after 1000 tries, force proper
    for u, v in edges:
        if colors[u] == colors[v]:
            forbidden = set(colors[nb] for nb in neighbors[u])
            for c in range(4):
                if c not in forbidden:
                    colors[u] = c
                    break
    
    # Local search: target vertices in violating cycles
    best_colors = colors[:]
    best_viol = curr_viol
    
    for step in range(50000):
        if time.time() > deadline:
            break
        if curr_viol == 0:
            break
        
        # Find a violating pair and try to break it
        violating_pairs = []
        for i in range(k):
            for j in range(i+1, k):
                parent = list(range(n))
                rank = [0] * n
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                
                cycle_found = False
                cycle_vertices = []
                for u, v in edges:
                    if (colors[u] == i and colors[v] == j) or (colors[u] == j and colors[v] == i):
                        if not union(u, v):
                            cycle_found = True
                            # Find the cycle by tracing back
                            # For simplicity, just pick a random vertex from the violating subgraph
                            break
                if cycle_found:
                    violating_pairs.append((i, j))
        
        if not violating_pairs:
            curr_viol = 0
            break
        
        # Pick a violating pair
        pi, pj = random.choice(violating_pairs)
        
        # Find vertices with color pi or pj that are in the violating subgraph
        candidate_vertices = []
        for u in range(n):
            if colors[u] in (pi, pj):
                # Check if u is connected to another vertex of the other color in the same component
                # Simplified: just include all vertices of these colors
                candidate_vertices.append(u)
        
        if not candidate_vertices:
            # No candidates, try random vertex
            v = random.randint(0, n-1)
        else:
            v = random.choice(candidate_vertices)
        
        old_c = colors[v]
        best_c = old_c
        best_v = curr_viol
        
        for c in range(k):
            if c == old_c:
                continue
            # Check proper
            if any(colors[nb] == c for nb in neighbors[v]):
                continue
            
            colors[v] = c
            v_new = count_violations(colors, k)
            if v_new < best_v:
                best_v = v_new
                best_c = c
            colors[v] = old_c
        
        colors[v] = best_c
        curr_viol = best_v
        
        if curr_viol < best_viol:
            best_viol = curr_viol
            best_colors = colors[:]
    
    # Return the best found
    if best_viol == 0:
        return [int(c) for c in best_colors]
    
    # If we couldn't find a perfect solution, try one more time with a different initialization
    random.seed()
    colors = [random.randint(0, 3) for _ in range(n)]
    for u, v in edges:
        if colors[u] == colors[v]:
            forbidden = set(colors[nb] for nb in neighbors[u])
            for c in range(4):
                if c not in forbidden:
                    colors[u] = c
                    break
    
    curr_viol = count_violations(colors, 4)
    for step in range(30000):
        if time.time() > deadline:
            break
        if curr_viol == 0:
            break
        
        v = random.randint(0, n-1)
        old_c = colors[v]
        best_c = old_c
        best_v = curr_viol
        
        for c in range(4):
            if c == old_c:
                continue
            if any(colors[nb] == c for nb in neighbors[v]):
                continue
            
            colors[v] = c
            v_new = count_violations(colors, 4)
            if v_new < best_v:
                best_v = v_new
                best_c = c
            colors[v] = old_c
        
        colors[v] = best_c
        curr_viol = best_v
    
    if curr_viol == 0:
        return [int(c) for c in colors]
    
    # Return the best we have
    if best_viol == 0:
        return [int(c) for c in best_colors]
    
    # Last resort: return current if proper
    if is_proper(colors, 4):
        return [int(c) for c in colors]
    
    # Absolute fallback: simple 4-coloring
    colors = []
    for i in range(n):
        forbidden = set(colors[adj[i][0]] if adj[i] else set())
        for c in range(4):
            if c not in forbidden:
                colors.append(c)
                break
        else:
            colors.append(0)
    
    return [int(c) for c in colors]

# EVOLVE-BLOCK-END
