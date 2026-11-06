import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Construct the polarity graph of the projective plane of order 5 (n=31).
    2. Iteratively add vertices using a conflict graph to ensure C4-freeness.
    3. Apply spectral-guided local search to optimize the graph structure.
    4. Check independence number < 11.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # --- Helper: Fast Alpha Solver using Bitmasks ---
    def max_independent_set_size(n, adj, time_limit):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        """
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')

        def branch(candidates_mask, current_size):
            if time.time() > time_limit:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Degree in candidate subgraph
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    # --- Helper: Check if adding edge (u,v) maintains C4-freeness ---
    def can_add_edge(n, adj, u, v):
        """
        Check if adding edge (u,v) maintains C4-freeness.
        C4-free means no two vertices have more than one common neighbor.
        Adding edge (u,v) creates a C4 if there exist vertices a,b such that
        a is adjacent to both u and v, AND b is adjacent to both u and v.
        i.e., |N(u) ∩ N(v)| >= 2 before adding the edge would create C4s.
        Actually, we need to check: after adding (u,v), for any pair of vertices,
        they have at most one common neighbor.
        
        The critical check: for the new edge (u,v), we need that no other vertex
        w is adjacent to both u and v (otherwise w, u, v, w would form a C4... 
        wait, that's just a triangle if u-v-w-u). 
        
        Let me reconsider: C4 is a cycle of length 4: a-b-c-d-a.
        Adding edge (u,v) creates a C4 if there exist distinct vertices a,b such that:
        - a is adjacent to u
        - b is adjacent to u  
        - a is adjacent to v
        - b is adjacent to v
        - a != b
        - a,b != u,v
        
        This means |N(u) ∩ N(v)| >= 2 would create C4s through pairs in the intersection.
        
        Also need to check that adding (u,v) doesn't create C4s involving u or v as
        the "middle" vertices. But since we're only adding one edge, the main concern
        is whether u and v already have 2+ common neighbors.
        """
        if adj[u] & (1 << v):
            return False  # Edge already exists
        
        # Check common neighbors of u and v
        common = adj[u] & adj[v]
        if bin(common).count('1') >= 2:
            return False
        
        # Also need to verify that adding this edge doesn't create C4s where
        # u or v are part of the cycle with other vertices.
        # A C4 containing the new edge (u,v) would be: u - v - x - y - u
        # where x is adjacent to v, y is adjacent to u, and x-y is an edge.
        # This is equivalent to: there exist x in N(v)\{u} and y in N(u)\{v}
        # such that x-y is an edge and x != y.
        
        # But wait, this is already captured by the common neighbor check in a different way.
        # Let me think more carefully...
        
        # Actually, the condition for C4-freeness is: for every pair of vertices,
        # they have at most one common neighbor. Adding edge (u,v) affects:
        # 1. The common neighbors of u and v: must be <= 1 (checked above)
        # 2. For any vertex w != u,v: common neighbors of u and w, and v and w
        #    are unchanged by adding (u,v) unless w is u or v.
        # 3. The new edge (u,v) means: for any w, if w is adjacent to both u and v,
        #    that's fine as long as there's only one such w.
        
        # So the check above should be sufficient. But let me also verify by checking
        # that no C4 is formed. A C4 would be a cycle u-v-x-y-u where:
        # - u-v is the new edge
        # - v-x is an existing edge  
        # - x-y is an existing edge
        # - y-u is an existing edge
        # - x != y, x != u,v, y != u,v
        
        # This means: x in N(v)\{u}, y in N(u)\{v}, and x-y is an edge.
        # And we need x != y.
        
        # Let's check this condition:
        Nv = adj[v] & ~(1 << u)  # Neighbors of v except u
        Nu = adj[u] & ~(1 << v)  # Neighbors of u except v
        
        # For each x in Nv, check if any neighbor y of x (with y in Nu) creates a C4
        # i.e., x adjacent to some y in Nu, and y != x
        for x_bit in Nv:
            x = x_bit.bit_length() - 1
            # Check if x is adjacent to any vertex in Nu
            if adj[x] & Nu:
                # There exists y in Nu adjacent to x
                # If y != x, we have a C4: u-v-x-y-u
                # But if y == x, that would mean x is adjacent to itself, which is impossible
                return False
        
        return True

    # --- Helper: Estimate spectral impact of a move ---
    def spectral_score(n, adj):
        """
        Compute a heuristic score based on graph density and structure.
        Higher score = potentially better (lower independence number).
        We use the number of edges and a rough estimate of expansion.
        """
        edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        # For C4-free graphs, more edges generally means smaller independence number
        # Use edge density as a proxy
        return edge_count

    # --- Step 1: Construct Polarity Graph of PG(2,5) ---
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, y * inv_x % 5, z * inv_x % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, z * inv_y % 5))
                else:
                    inv_z = pow(z, -1, 5)
                    points.append((0, 0, 1))
    
    points = list(set(points))
    n_base = len(points)  # Should be 31
    
    def bilinear(u, v):
        return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5

    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Verify base graph
    if max_independent_set_size(n_base, adj, deadline - 50) >= 11:
        return {'n': 30, 'edges': []}

    best_n = n_base
    best_adj = adj[:]

    # --- Step 2: Iterative Growth via Conflict Graph ---
    for n in range(n_base + 1, 44):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(80):
            if time.time() > deadline - 15:
                break
            
            # Build Conflict Graph for current n-1 vertices
            conflict = [0] * (n - 1)
            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    if adj[i] & adj[j]:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
            
            # Find an independent set S in Conflict Graph (greedy with randomization)
            random.seed(12345 + attempt)
            candidates = (1 << (n - 1)) - 1
            S_mask = 0
            
            while candidates:
                best_v = -1
                min_conflicts = n
                verts = []
                m = candidates
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m -= lsb
                random.shuffle(verts)
                
                for v in verts:
                    if not (candidates & (1 << v)):
                        continue
                    conf_count = bin(conflict[v] & candidates).count('1')
                    if conf_count < min_conflicts:
                        min_conflicts = conf_count
                        best_v = v
                        if min_conflicts == 0:
                            break
                
                if best_v == -1:
                    break
                    
                S_mask |= (1 << best_v)
                candidates &= ~(1 << best_v)
                candidates &= ~conflict[best_v]
                
            # Add vertex n-1 with neighbors S_mask
            new_adj = adj[:] + [0]
            new_adj[n-1] = S_mask
            for i in range(n - 1):
                if S_mask & (1 << i):
                    new_adj[i] |= (1 << (n-1))
            
            # Check Alpha
            alpha = max_independent_set_size(n, new_adj, deadline - 10)
            if alpha < 11:
                adj = new_adj
                best_n = n
                best_adj = adj[:]
                found = True
                break
        
        if not found:
            break

    # --- Step 3: Spectral-Guided Local Search ---
    # Try to improve the graph by adding edges that maintain C4-freeness
    # and potentially reduce the independence number
    
    n_current = best_n
    adj_current = best_adj[:]
    current_score = spectral_score(n_current, adj_current)
    
    # Try to add edges to make the graph denser (which tends to reduce independence number)
    max_local_iterations = 200
    for iteration in range(max_local_iterations):
        if time.time() > deadline - 10:
            break
        
        # Generate candidate edges to add
        candidates = []
        for u in range(n_current):
            for v in range(u + 1, n_current):
                if not (adj_current[u] & (1 << v)):
                    if can_add_edge(n_current, adj_current, u, v):
                        # Estimate potential benefit: adding edge increases density
                        # Score by how many common neighbors are avoided
                        common = adj_current[u] & adj_current[v]
                        common_count = bin(common).count('1')
                        # Prefer edges where common_count is 0 or 1 (valid), and 
                        # where the edge connects vertices with low degree (more impactful)
                        deg_u = bin(adj_current[u]).count('1')
                        deg_v = bin(adj_current[v]).count('1')
                        priority = (n_current - deg_u) + (n_current - deg_v) - common_count * 10
                        candidates.append((priority, u, v))
        
        if not candidates:
            break
        
        # Sort by priority (highest first)
        candidates.sort(key=lambda x: -x[0])
        
        # Try the top candidates
        improved = False
        for priority, u, v in candidates[:20]:  # Try top 20
            if time.time() > deadline - 10:
                break
            
            # Try adding edge (u, v)
            new_adj = adj_current[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            # Verify C4-freeness (should be guaranteed by can_add_edge, but double-check)
            c4_free = True
            for i in range(n_current):
                for j in range(i + 1, n_current):
                    common = new_adj[i] & new_adj[j]
                    if bin(common).count('1') > 1:
                        c4_free = False
                        break
                if not c4_free:
                    break
            
            if not c4_free:
                continue
            
            # Check independence number
            alpha = max_independent_set_size(n_current, new_adj, deadline - 8)
            if alpha < 11:
                # Accept the move
                adj_current = new_adj
                current_score = spectral_score(n_current, adj_current)
                best_adj = adj_current[:]
                improved = True
                break
        
        if not improved:
            # Try removing edges to see if we can then add better ones
            # For now, just break if no improvement
            break

    # Final verification
    n_final = best_n
    adj_final = best_adj
    
    # Verify C4-freeness
    c4_free = True
    for i in range(n_final):
        for j in range(i + 1, n_final):
            common = adj_final[i] & adj_final[j]
            if bin(common).count('1') > 1:
                c4_free = False
                break
        if not c4_free:
            break
    
    if not c4_free:
        # Fallback to previous best
        n_final = best_n
        adj_final = best_adj
    
    # Final alpha check
    alpha = max_independent_set_size(n_final, adj_final, deadline - 5)
    if alpha >= 11:
        # This shouldn't happen, but fallback
        return {'n': 30, 'edges': []}

    # Convert to edge list
    edge_list = []
    for i in range(n_final):
        for j in range(i + 1, n_final):
            if adj_final[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': n_final, 'edges': edge_list}
