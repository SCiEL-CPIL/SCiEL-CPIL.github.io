import random
import time

def is_k4_free(adj):
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] & (1 << b)): continue
            for c in range(b+1, n):
                if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                if adj[a] & adj[b] & adj[c]:
                    return False
    return True

def get_largest_triangle_free(adj):
    n = len(adj)
    
    # Precompute for each edge (u,v) the common neighbors
    # A subset S is triangle-free iff for every edge (u,v) in S, 
    # (adj[u] & adj[v] & mask(S)) == 0
    
    # More efficient: iterate subsets by size, check triangle-free quickly
    # For a subset mask m, it's triangle-free iff:
    # For all u in m: (adj[u] & m) has no two adjacent vertices in m
    # i.e., for all u in m: let N_u = adj[u] & m, then N_u is an independent set in G[m]
    # Which means: for all u,v in N_u with u<v: not (adj[u] & (1<<v))
    
    # Fastest check: for each u in m, check if adj[u] & m contains an edge
    # Precompute: for each vertex u, the edges among its neighbors
    
    # Even faster: use the fact that S is triangle-free iff
    # for every triangle (a,b,c) in G, not all of a,b,c are in S
    # So we can enumerate all triangles first, then check subsets
    
    # Find all triangles
    triangles = []
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] & (1 << b)): continue
            common = adj[a] & adj[b]
            # For each c > b that is common neighbor
            c_mask = common & ~((1 << (b+1)) - 1)  # bits above b
            while c_mask:
                c_bit = c_mask & -c_mask
                c = c_bit.bit_length() - 1
                triangles.append((a, b, c))
                c_mask ^= c_bit
    
    if not triangles:
        return n
    
    # Now check subsets from size n down to 1
    # A subset is triangle-free iff it contains no complete triangle
    # Use inclusion-exclusion or just check each triangle
    
    from itertools import combinations
    
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            m = 0
            for v in subset:
                m |= (1 << v)
            
            # Check if any triangle is fully contained in m
            has_triangle = False
            for (a, b, c) in triangles:
                if (m >> a) & 1 and (m >> b) & 1 and (m >> c) & 1:
                    has_triangle = True
                    break
            if not has_triangle:
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    best_l = 18 # Worst case
    best_graph = None # adj list of bitmasks
    
    n = 18
    
    # Helper to generate a random K4-free graph
    def gen_random_k4_free():
        # Start with random graph
        # p=0.5 usually
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        # While not K4-free, remove edge
        # This might take many steps if dense.
        # To speed up, find a K4 and remove an edge.
        
        # Optimization: We can check K4-freeness during generation or repair.
        # Let's just loop repair.
        
        for _ in range(100): # Max iterations
            if is_k4_free(adj):
                return adj
            # Find a K4
            # We can reuse logic from is_k4_free to find one
            found = False
            for a in range(n):
                for b in range(a+1, n):
                    if not (adj[a] & (1 << b)): continue
                    for c in range(b+1, n):
                        if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                        common = adj[a] & adj[b] & adj[c]
                        if common:
                            # Find d
                            # lowest set bit
                            d = (common & -common).bit_length() - 1
                            # Remove edge (a, d)
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                # Should not happen if is_k4_free returned False
                return adj
                
        return adj

    while time.time() < deadline:
        adj = gen_random_k4_free()
        l = get_largest_triangle_free(adj)
        
        if l < best_l:
            best_l = l
            best_graph = adj[:]
        
        # Local search: try edge flips to improve
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try removing edges
                for i in range(n):
                    for j in range(i+1, n):
                        if not (adj[i] & (1 << j)):
                            continue
                        # Try removing edge (i,j)
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        if is_k4_free(adj):
                            l2 = get_largest_triangle_free(adj)
                            if l2 < l:
                                l = l2
                                improved = True
                                if l < best_l:
                                    best_l = l
                                    best_graph = adj[:]
                            else:
                                # Revert
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                        else:
                            # Revert
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                # Try adding edges
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            continue
                        # Try adding edge (i,j)
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        if is_k4_free(adj):
                            l2 = get_largest_triangle_free(adj)
                            if l2 < l:
                                l = l2
                                improved = True
                                if l < best_l:
                                    best_l = l
                                    best_graph = adj[:]
                            else:
                                # Revert
                                adj[i] &= ~(1 << j)
                                adj[j] &= ~(1 << i)
                        else:
                            # Revert
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
        
        # If we've found a very good solution, we can stop early
        if best_l <= 5:
            break

    # Convert best_graph to list of edges
    edges = []
    if best_graph is None:
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
                
    return edges
