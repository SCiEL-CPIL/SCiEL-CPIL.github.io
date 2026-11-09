import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    def can_add_fast(adj, u, v):
        """Check if adding edge (u,v) creates a C4 using bit operations.
        A C4 through new edge (u,v) is u-v-w-x-u where w~v, x~u, w~x.
        So we need: exists w in N(v), x in N(u) with w~x, w!=x, w!=u, x!=v.
        Equivalently: any neighbor of u is adjacent to any neighbor of v (excluding u,v themselves).
        """
        nu = adj[u] & ~(1 << v)  # neighbors of u excluding v
        nv = adj[v] & ~(1 << u)  # neighbors of v excluding u
        c = nu
        while c:
            a_bit = c & -c
            a = a_bit.bit_length() - 1
            c ^= a_bit
            if (adj[a] & nv) != 0:
                return False
        return True

    def can_remove_fast(adj, u, v):
        """Check if removing edge (u,v) keeps C4-free (always true since removing edges can't create C4)."""
        return True

    def build_polar_graph(q):
        """Build a C4-free graph using parabolic quadric Q^+(3,q) over F_q.
        Vertices are points on Q^+(3,q) in PG(3,q), which has (q+1)^2 points.
        Two points are adjacent if they are NOT orthogonal (i.e., their bilinear form is nonzero).
        This gives a C4-free graph with high minimum degree.
        
        For Q^+(3,q): the orthogonality graph has no C4 (it's a polar graph).
        We want the COMPLEMENT of the orthogonality graph to be C4-free... 
        Actually, let me reconsider. The standard result is that the collinearity graph
        of Q^+(3,q) is C4-free. Let me use a simpler construction.
        
        Alternative: Use a bipartite C4-free construction.
        Let L and R each have m vertices. Connect l_i to r_j if j = i^2 mod p or similar.
        But we need a general graph, not necessarily bipartite.
        
        Let me use the following: vertices are pairs (a, b) where a in F_q, b in F_q.
        Two vertices (a1,b1) and (a2,b2) are adjacent if a1*a2 + b1*b2 = 1 (over F_q).
        This is the "orthogonal complement" construction which yields C4-free graphs.
        """
        # Use the standard C4-free construction: 
        # Vertices: F_q x F_q (so n = q^2)
        # (a1,b1) ~ (a2,b2) iff a1*a2 + b1*b2 = 1 in F_q
        # This is C4-free because if (a1,b1)~(a2,b2)~(a3,b3)~(a4,b4)~(a1,b1)
        # then we get a contradiction from the bilinear form.
        
        n = q * q
        adj = [0] * n
        edges = []
        
        def idx(a, b):
            return a * q + b
        
        for a1 in range(q):
            for b1 in range(q):
                u = idx(a1, b1)
                for a2 in range(a1, q):
                    for b2 in range(q):
                        v = idx(a2, b2)
                        if u == v:
                            continue
                        # Check if a1*a2 + b1*b2 = 1 mod q
                        if (a1 * a2 + b1 * b2) % q == 1:
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
                            edges.append([u, v])
        
        return adj, edges, n

    def greedy_fill(adj, deg, n, min_deg_needed, deadline):
        """Greedily add edges to reach min_deg_needed, maintaining C4-free property."""
        in_graph = [[False] * n for _ in range(n)]
        for u in range(n):
            c = adj[u]
            while c:
                b = c & -c
                v = b.bit_length() - 1
                in_graph[u][v] = True
                c ^= b
        
        heap = []
        for u in range(n):
            for v in range(u + 1, n):
                if not in_graph[u][v]:
                    heapq.heappush(heap, (deg[u] + deg[v], random.random(), u, v))
        
        while heap:
            if time.time() > deadline - 10:
                break
            
            pri, _, u, v = heapq.heappop(heap)
            
            if in_graph[u][v]:
                continue
            
            if pri != deg[u] + deg[v]:
                continue
            
            if can_add_fast(adj, u, v):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                deg[u] += 1
                deg[v] += 1
                in_graph[u][v] = True
                in_graph[v][u] = True
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if not in_graph[u][w]:
                        heapq.heappush(heap, (deg[u] + deg[w], random.random(), u, w))
                    if not in_graph[v][w]:
                        heapq.heappush(heap, (deg[v] + deg[w], random.random(), v, w))
        
        return adj, deg

    def degree_balance_swap(adj, deg, n, min_deg_needed, deadline, max_swaps=5000):
        """Try to balance degrees via edge swaps while maintaining C4-free."""
        in_graph = [[False] * n for _ in range(n)]
        for u in range(n):
            c = adj[u]
            while c:
                b = c & -c
                v = b.bit_length() - 1
                in_graph[u][v] = True
                c ^= b
        
        swaps_made = 0
        for _ in range(max_swaps):
            if time.time() > deadline - 10:
                break
            
            if min(deg) >= min_deg_needed:
                break
            
            # Find lowest degree vertex
            low_idx = min(range(n), key=lambda i: deg[i])
            
            # Find a neighbor of low_idx to remove, and a non-neighbor to add
            # Remove edge (low_idx, x) and add edge (low_idx, y) where deg[y] < deg[x]
            # This increases min degree potential
            
            neighbors = []
            c = adj[low_idx]
            while c:
                b = c & -c
                v = b.bit_length() - 1
                neighbors.append(v)
                c ^= b
            
            non_neighbors = [i for i in range(n) if i != low_idx and not in_graph[low_idx][i]]
            
            if not neighbors or not non_neighbors:
                break
            
            # Try to find a valid swap
            random.shuffle(neighbors)
            random.shuffle(non_neighbors)
            
            found_swap = False
            for x in neighbors:
                for y in non_neighbors:
                    if deg[y] >= deg[x]:
                        continue
                    
                    # Check if we can remove (low_idx, x) and add (low_idx, y)
                    # Removing (low_idx, x): always C4-safe
                    # Adding (low_idx, y): check C4
                    # Temporarily remove (low_idx, x)
                    adj[low_idx] &= ~(1 << x)
                    adj[x] &= ~(1 << low_idx)
                    
                    if can_add_fast(adj, low_idx, y):
                        # Do the swap
                        adj[low_idx] |= 1 << y
                        adj[y] |= 1 << low_idx
                        deg[low_idx] = deg[low_idx] - 1 + 1  # net 0
                        deg[x] -= 1
                        deg[y] += 1
                        in_graph[low_idx][x] = False
                        in_graph[x][low_idx] = False
                        in_graph[low_idx][y] = True
                        in_graph[y][low_idx] = True
                        swaps_made += 1
                        found_swap = True
                        break
                    else:
                        # Restore
                        adj[low_idx] |= 1 << x
                        adj[x] |= 1 << low_idx
                    
                    if found_swap:
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break
        
        return adj, deg, swaps_made

    def extract_edges(adj, n):
        edges = []
        for u in range(n):
            c = adj[u]
            while c:
                b = c & -c
                v = b.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                c ^= b
        return edges

    # Try polar graph constructions with different q values
    # q^2 = n, so for n=46 or 47, we need q around 7 (49 vertices)
    # We can start with q=7 (49 vertices) and remove vertices, or use other approaches
    
    # First, try direct greedy with various seeds for n=47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        
        # Try polar graph initialization if possible
        # For n close to q^2, build polar graph and trim
        best_q = int(target_n ** 0.5)
        if best_q * best_q >= target_n and best_q >= 4:
            # Build polar graph with q = best_q
            if time.time() < deadline - 30:
                adj_p, edges_p, n_p = build_polar_graph(best_q)
                # Trim to target_n vertices if n_p > target_n
                if n_p > target_n:
                    # Keep first target_n vertices
                    adj = [0] * target_n
                    deg = [0] * target_n
                    for u in range(target_n):
                        c = adj_p[u]
                        while c:
                            b = c & -c
                            v = b.bit_length() - 1
                            if v < target_n:
                                adj[u] |= 1 << v
                                deg[u] += 1
                            c ^= b
                    # Recompute degrees
                    deg = [bin(adj[u]).count('1') for u in range(target_n)]
                    
                    # Greedy fill
                    adj, deg = greedy_fill(adj, deg, target_n, min_deg_needed, deadline)
                    
                    # Degree balance
                    adj, deg, _ = degree_balance_swap(adj, deg, target_n, min_deg_needed, deadline)
                    
                    if min(deg) >= min_deg_needed:
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = extract_edges(adj, target_n)
                        found = True
                        break
        
        # Try pure greedy with multiple restarts
        for restart in range(100):
            if time.time() > deadline - 5:
                break
                
            random.seed(restart * 7919 + 13 * target_n + 1000)
            
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            in_graph = [[False] * target_n for _ in range(target_n)]
            
            heap = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    heapq.heappush(heap, (0, random.random(), u, v))
            
            while heap:
                if time.time() > deadline - 5:
                    break
                    
                pri, _, u, v = heapq.heappop(heap)
                
                if in_graph[u][v]:
                    continue
                
                if pri != deg[u] + deg[v]:
                    continue
                
                if can_add_fast(adj, u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    in_graph[u][v] = True
                    in_graph[v][u] = True
                    
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        if not in_graph[u][w]:
                            heapq.heappush(heap, (deg[u] + deg[w], random.random(), u, w))
                        if not in_graph[v][w]:
                            heapq.heappush(heap, (deg[v] + deg[w], random.random(), v, w))
            
            if min(deg) >= min_deg_needed:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges[:]
                found = True
                break
        
        if best_n >= target_n:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
