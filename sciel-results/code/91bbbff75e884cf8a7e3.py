"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_valid_coloring(coloring):
    """Check if coloring is valid: adjacent vertices differ, and no bicolored cycle."""
    n = P['n']
    # Check adjacent vertices differ
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycles
    colors_used = set(coloring)
    for ca in colors_used:
        for cb in colors_used:
            if ca >= cb:
                continue
            # Build subgraph induced by colors ca and cb
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
            
            for u, v in P['edges']:
                cu, cv = coloring[u], coloring[v]
                if (cu == ca and cv == cb) or (cu == cb and cv == ca):
                    if not union(u, v):
                        return False
    return True

def _count_distinct(coloring):
    return len(set(coloring))

def _greedy_coloring(adj, k, start_vertex=0):
    """Try to color the graph with k colors using greedy with backtracking."""
    n = P['n']
    coloring = [-1] * n
    
    # Order vertices by degree (descending)
    vertices = sorted(range(n), key=lambda v: -len(adj[v]))
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    def check_bicolored_cycle_with_new_edge(v, c, nb, cnb, current_coloring):
        """Check if adding edge (v,nb) with colors (c,cnb) creates a bicolored cycle."""
        # BFS from v in the bicolored subgraph (colors c and cnb) to see if we can reach nb
        # without using the direct edge (v,nb)
        if v == nb:
            return True
        visited = set()
        queue = deque()
        visited.add(v)
        queue.append(v)
        
        while queue:
            cur = queue.popleft()
            for nb2 in adj[cur]:
                if nb2 in visited:
                    continue
                c2 = current_coloring[nb2]
                if c2 in (c, cnb):
                    if nb2 == nb:
                        # Found nb, but check if it's via the direct edge from v
                        if not (cur == v):  # not the direct edge
                            return True
                    visited.add(nb2)
                    queue.append(nb2)
        return False
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = vertices[idx]
        
        for c in range(k):
            # Check if color c is allowed (no same-colored neighbor already assigned)
            conflict = False
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < idx:  # already assigned
                    if coloring[nb] == c:
                        conflict = True
                        break
            if conflict:
                continue
            
            coloring[v] = c
            
            # Check bicolored cycle constraints for edges from v to already-assigned neighbors
            valid = True
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < idx:
                    cnb = coloring[nb]
                    if cnb == c:
                        valid = False
                        break
                    # Check if this creates a bicolored cycle
                    if check_bicolored_cycle_with_new_edge(v, c, nb, cnb, coloring):
                        valid = False
                        break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            coloring[v] = -1
        
        return False
    
    deadline = start_time + 80
    if backtrack(0):
        return coloring
    return None

def _try_3_coloring():
    """Try to find a 3-coloring."""
    global start_time
    adj = _build_adjacency()
    n = P['n']
    coloring = [-1] * n
    
    # Use a smarter approach: try all 3-colorings of a small seed, then extend
    # The graph has structure, so let's try a systematic approach
    
    # Order vertices
    vertices = sorted(range(n), key=lambda v: -len(adj[v]))
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    def check_and_extend(partial_idx, current_coloring):
        """Try to extend the partial coloring to a full valid coloring."""
        if time.time() > deadline:
            return None
        if partial_idx == n:
            return current_coloring[:]
        
        v = vertices[partial_idx]
        
        for c in range(3):
            # Check same-color neighbor constraint
            conflict = False
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < partial_idx:
                    if current_coloring[nb] == c:
                        conflict = True
                        break
            if conflict:
                continue
            
            current_coloring[v] = c
            
            # Check bicolored cycle for each neighbor with different color
            valid = True
            for nb in adj[v]:
                nb_idx = v_to_idx[nb]
                if nb_idx < partial_idx:
                    cnb = current_coloring[nb]
                    if cnb != c:
                        # Check if adding this edge creates a bicolored cycle
                        # BFS from v in (c, cnb)-subgraph to see if nb is reachable
                        # without the direct edge
                        visited = set()
                        queue = deque([v])
                        visited.add(v)
                        found_cycle = False
                        while queue:
                            cur = queue.popleft()
                            for nb2 in adj[cur]:
                                if nb2 in visited:
                                    continue
                                c2 = current_coloring[nb2]
                                if c2 in (c, cnb):
                                    if nb2 == nb and cur != v:
                                        found_cycle = True
                                        break
                                    visited.add(nb2)
                                    queue.append(nb2)
                            if found_cycle:
                                break
                        if found_cycle:
                            valid = False
                            break
            
            if valid:
                result = check_and_extend(partial_idx + 1, current_coloring)
                if result is not None:
                    return result
            
            current_coloring[v] = -1
        
        return None
    
    deadline = start_time + 75
    result = check_and_extend(0, coloring)
    return result

def construct():
    global start_time
    start_time = time.time()
    deadline = start_time + 80
    
    adj = _build_adjacency()
    n = P['n']
    
    # Try 3 colors first
    coloring = _try_3_coloring()
    if coloring is not None and _is_valid_coloring(coloring):
        return [int(c) for c in coloring]
    
    # Try 4 colors with backtracking
    coloring = _greedy_coloring(adj, 4)
    if coloring is not None and _is_valid_coloring(coloring):
        return [int(c) for c in coloring]
    
    # Fallback: use a simple valid coloring with more colors
    # Use greedy with conflict resolution
    coloring = [0] * n
    for v in range(n):
        used = set()
        for nb in adj[v]:
            if nb < v:
                used.add(coloring[nb])
        for c in range(5):
            if c not in used:
                coloring[v] = c
                break
    
    # Verify and fix if needed
    if not _is_valid_coloring(coloring):
        # Last resort: all distinct colors
        coloring = list(range(n))
    
    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END
