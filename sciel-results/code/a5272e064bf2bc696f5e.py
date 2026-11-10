"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_n = 0
    best_colors = None

    def is_valid(n, colors):
        """Check if no color class contains a C4 using numpy for speed."""
        # Build adjacency matrices for each color
        for c in range(5):
            # Build adjacency matrix
            idx = 0
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i, j] = 1
                        adj[j, i] = 1
                    idx += 1
            # Count common neighbors for all pairs: (A @ A)[i,j] = number of common neighbors
            common = adj @ adj
            # Check if any pair has >= 2 common neighbors
            if np.any(common >= 2):
                return False
        return True

    # Try cyclic coloring for small n as baseline
    def cyclic_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                d = min(d, n - d)
                c = d - 1
                if c < 0:
                    c = 0
                elif c > 4:
                    c = 4
                colors.append(c)
        return colors

    for n in range(1, 16):
        if time.time() > deadline:
            break
        colors = cyclic_coloring(n)
        if is_valid(n, colors):
            if n > best_n:
                best_n = n
                best_colors = colors

    # Fast C4 check for greedy construction using numpy
    def greedy_construct(n, seed, time_limit):
        """Greedy random construction with fast C4 checking."""
        rng = random.Random(seed)
        
        # color_matrix[i,j] = color of edge (i,j), -1 if unassigned
        color_matrix = np.full((n, n), -1, dtype=np.int8)
        
        # For each color, maintain adjacency as numpy arrays
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        
        # Precompute edge list
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(edges)
        
        # For fast C4 check: when adding (u,v) in color c, check if there's
        # a path u-a-b-v of length 3 in color c. This means:
        # (adj[c][u] & adj[c][v]) has some vertex a, and adj[c][a] has some b in adj[c][v]
        # Actually: exists a,b with adj[c][u,a]=1, adj[c][a,b]=1, adj[c][b,v]=1, a!=b
        # This is equivalent to: (adj[c][u] @ adj[c] @ adj[c][v]) > 0 after masking u,v
        # But simpler: for each a in N_c(u), check if N_c(a) intersects N_c(v)\{a}
        
        def creates_c4(u, v, c):
            """Check if adding edge (u,v) in color c creates a C4."""
            # Need: exists a,b with adj[c][u,a]=1, adj[c][a,b]=1, adj[c][b,v]=1, a!=b, a!=v, b!=u
            # Equivalent: the number of length-3 paths from u to v in color c (not using edge uv) > 0
            # Using matrix: (adj_c @ adj_c @ adj_c)[u,v] counts walks of length 3
            # But we need simple paths. For now, use the walk count as a necessary condition,
            # then verify. Actually for C4 detection, a walk of length 3 from u to v
            # where the intermediate vertices are distinct from u,v and each other suffices
            # for detecting if a C4 would form.
            
            # Faster: iterate over common neighbors
            # A C4 is formed if there exists a vertex w that is adjacent to both u and v in color c
            # AND w has at least one neighbor in common with... no.
            # 
            # Correct: C4 formed iff exists distinct a,b with:
            #   adj[c][u,a]=1, adj[c][a,b]=1, adj[c][b,v]=1
            # Let's use: for each a where adj[c][u,a]=1 and a!=v:
            #   if adj[c][a] intersects (adj[c][v] \ {a}) then C4
            # i.e., if there exists b in adj[c][a] ∩ adj[c][v] with b != a and b != u
            
            au = adj[c][u]
            av = adj[c][v]
            
            # Get neighbors of u in color c (excluding v)
            candidates = np.where(au == 1)[0]
            for a in candidates:
                if a == v:
                    continue
                # Check if N_c(a) intersects N_c(v) \ {a, u}
                # adj[c][a] & adj[c][v], then check if any index != a and != u
                inter = adj[c][a] & av
                # Exclude a and u from consideration
                inter[a] = 0
                inter[u] = 0
                if np.any(inter):
                    return True
            return False
        
        for (u, v) in edges:
            if time.time() > time_limit:
                return None
            
            found = False
            colors_to_try = [0, 1, 2, 3, 4]
            rng.shuffle(colors_to_try)
            
            for c in colors_to_try:
                if not creates_c4(u, v, c):
                    color_matrix[u, v] = c
                    color_matrix[v, u] = c
                    adj[c][u, v] = 1
                    adj[c][v, u] = 1
                    found = True
                    break
            
            if not found:
                return None
        
        # Build result in lexicographic order
        result = []
        for i in range(n):
            for j in range(i+1, n):
                result.append(int(color_matrix[i, j]))
        
        return result
    
    # Try increasing n
    for n in range(max(best_n + 1, 12), 27):
        if time.time() > deadline:
            break
        
        success = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 0.5:
                break
            
            time_limit = min(remaining * 0.7, 3.0)
            result = greedy_construct(n, seed=attempt * 7919 + 13, time_limit=time_limit)
            
            if result is not None:
                if is_valid(n, result):
                    if n > best_n:
                        best_n = n
                        best_colors = result
                    success = True
                    break
        
        if not success and n > 15:
            # If we can't find one for this n, try a bit more with different strategy
            pass
    
    # Local search improvement: try to fix a near-miss coloring
    def local_search(n, init_colors, time_limit):
        """Try to fix a coloring by recoloring edges to eliminate C4s."""
        rng = random.Random(42)
        colors = list(init_colors)
        
        # Build adjacency
        adj = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        
        # Find all C4s
        def find_c4s():
            c4s = []
            for c in range(5):
                neighbors = [set() for _ in range(n)]
                for (u, v) in adj[c]:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                for u in range(n):
                    for v in range(u+1, n):
                        common = neighbors[u] & neighbors[v]
                        if len(common) >= 2:
                            c4s.append((c, u, v, list(common)[:2]))
            return c4s
        
        # Try to recolor edges to break C4s
        for iteration in range(500):
            if time.time() > time_limit:
                break
            
            c4s = find_c4s()
            if not c4s:
                break
            
            c, u, v, commons = c4s[0]
            # Pick an edge in the C4 to recolor
            # The C4 is u-a-v-b-u where a,b are in commons
            a, b = commons
            # Edges to consider: (u,a), (a,v), (v,b), (b,u)
            edge_options = [(u, a), (a, v), (v, b), (b, u)]
            rng.shuffle(edge_options)
            
            improved = False
            for (ei, ej) in edge_options:
                if ei > ej:
                    ei, ej = ej, ei
                # Find index in colors
                idx_e = None
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if i == ei and j == ej:
                            idx_e = idx
                            break
                        idx += 1
                    if idx_e is not None:
                        break
                
                if idx_e is None:
                    continue
                
                old_c = colors[idx_e]
                # Try other colors
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    # Tentatively recolor
                    adj[old_c].remove((ei, ej))
                    adj[new_c].add((ei, ej))
                    colors[idx_e] = new_c
                    
                    # Check if this creates new C4s or reduces count
                    new_c4s = find_c4s()
                    if len(new_c4s) < len(c4s):
                        improved = True
                        break
                    else:
                        # Revert
                        adj[new_c].remove((ei, ej))
                        adj[old_c].add((ei, ej))
                        colors[idx_e] = old_c
                
                if improved:
                    break
            
            if not improved:
                break
        
        if not find_c4s():
            return colors
        return None
    
    # Try local search from random colorings for larger n
    for n in range(max(best_n + 1, 16), 27):
        if time.time() > deadline:
            break
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            # Create random initial coloring
            rng = random.Random(attempt * 31 + n)
            init = [rng.randint(0, 4) for _ in range(n*(n-1)//2)]
            
            time_limit = min(remaining * 0.5, 20.0)
            result = local_search(n, init, time_limit)
            
            if result is not None:
                if is_valid(n, result):
                    if n > best_n:
                        best_n = n
                        best_colors = result
                    break
    
    if best_colors is None:
        best_n = 11
        best_colors = cyclic_coloring(11)
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
