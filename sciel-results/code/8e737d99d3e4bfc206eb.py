import time
import random
import sys

def construct():
    start_time = time.time()
    # Leave 10s buffer for safety
    deadline = start_time + 110.0 
    n = 18
    
    def is_k4_free(adj):
        # adj is list of bitmasks
        for u in range(n):
            neigh_mask = adj[u]
            # Extract neighbors
            bits = []
            m = neigh_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                bits.append(v)
                m ^= lsb
            
            # Check if any triangle in neighbors
            # Number of neighbors is small in K4-free graphs? Not necessarily, but avg degree is limited.
            # In K4-free, if u has neighbors v,w, v-w cannot be edge if v,w,u form K3? No.
            # K4-free means no K4. 
            # If u, v, w form a triangle, that's fine.
            # K4 requires u,v,w,x where all connected.
            # So for each triangle (u,v,w), we must ensure no x connected to all three.
            
            # Faster check:
            # For each pair of neighbors (v, w) of u that are connected (i.e., v,w is an edge),
            # check if there is a common neighbor x of u, v, w.
            
            # Let's stick to the simple one but optimized with bitmasks.
            # For each v in bits:
            #   For each w in bits where w > v:
            #     if edge(v,w) exists:
            #        common = adj[u] & adj[v] & adj[w]
            #        if common != 0: return False
        
            for i in range(len(bits)):
                v = bits[i]
                for j in range(i+1, len(bits)):
                    w = bits[j]
                    if (adj[v] >> w) & 1:
                        # u, v, w form a triangle
                        # Check for K4: is there a vertex x adjacent to u, v, and w?
                        # x must be in adj[u] (which is neigh_mask), adj[v], and adj[w]
                        common = neigh_mask & adj[v] & adj[w]
                        # Remove u, v, w from common to avoid self-loops? 
                        # adj[u] doesn't have u. adj[v] doesn't have v.
                        # But common might contain u? No, adj[v] doesn't have u? 
                        # Wait, if u-v is edge, adj[v] has u.
                        # So common contains u? 
                        # adj[u] has v, w.
                        # adj[v] has u, w.
                        # adj[w] has u, v.
                        # Intersection: u is in adj[v] and adj[w]. Is u in adj[u]? No.
                        # So u is NOT in common.
                        # v is in adj[u] and adj[w]. Is v in adj[v]? No.
                        # So v is NOT in common.
                        # w is in adj[u] and adj[v]. Is w in adj[w]? No.
                        # So w is NOT in common.
                        # Thus common only contains vertices x != u,v,w.
                        if common:
                            return False
        return True

    def get_triangles(adj):
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    for w in range(v+1, n):
                        if (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                            mask = (1 << u) | (1 << v) | (1 << w)
                            triangles.append(mask)
        return triangles

    def min_vertex_cover_size(triangles):
        """
        Returns the minimum number of vertices needed to hit all triangles.
        Uses Branch and Bound.
        """
        if not triangles:
            return 0
        
        # Sort triangles by popcount? All have 3.
        # We can use a memoization or just B&B.
        # State: tuple of triangle masks (sorted for consistency)
        # Since n=18, the number of distinct triangle sets encountered in B&B is manageable.
        
        memo = {}
        
        def solve(active_tris):
            if not active_tris:
                return 0
            
            key = tuple(sorted(active_tris))
            if key in memo:
                return memo[key]
            
            # Heuristic: Pick the triangle with the fewest available vertices?
            # Or just pick the first one.
            # To improve pruning, pick a triangle.
            t = active_tris[0]
            
            best = n # Max possible cover size is n
            
            # Try removing each vertex in t
            # We need to iterate through bits of t
            m = t
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Remove v from all triangles
                new_tris = []
                for tri in active_tris:
                    if not (tri & lsb):
                        new_tris.append(tri)
                
                # Recurse
                cost = 1 + solve(new_tris)
                if cost < best:
                    best = cost
                
                m ^= lsb
            
            memo[key] = best
            return best
        
        return solve(triangles)

    def score(adj):
        tris = get_triangles(adj)
        if not tris:
            return n
        min_cover = min_vertex_cover_size(tris)
        return n - min_cover

    # Initialize with K6,6,6 (Turan graph T(18,3))
    # This graph is K4-free.
    # Triangles only exist within parts? No, K6,6,6 is complete multipartite.
    # Edges are between parts. No edges within parts.
    # So no triangles at all?
    # Wait. K6,6,6 has parts A, B, C.
    # Edges (a,b), (b,c), (c,a) exist.
    # Triangle (a,b,c) exists?
    # a in A, b in B, c in C.
    # a-b is edge. b-c is edge. c-a is edge.
    # Yes, there are many triangles.
    # The induced subgraph on any 3 vertices from different parts is a triangle.
    # The induced subgraph on any 2 vertices from same part is empty (independent).
    # The induced subgraph on vertices from 2 parts is bipartite (triangle-free).
    # So we can pick all 12 vertices from A and B. They form a bipartite graph, which is triangle-free.
    # So for K6,6,6, the max triangle-free induced subgraph has size 12.
    # Score = 12.
    
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    current_adj = adj[:]
    current_score = score(current_adj)
    best_adj = current_adj[:]
    best_score = current_score
    
    random.seed(42)
    
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
            
    no_improve_count = 0
    step = 0
    
    while time.time() < deadline:
        step += 1
        if step % 1000 == 0:
            # Optional: print progress to stderr if needed, but keep it quiet
            pass

        # Pick a random edge to flip
        e = random.choice(all_edges)
        u, v = e
        
        # Flip edge
        if (current_adj[u] >> v) & 1:
            # Remove edge
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            removed = True
        else:
            # Add edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            removed = False
            
        # Check K4-free
        if not is_k4_free(current_adj):
            # Revert
            if removed:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            else:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            continue
            
        # Calculate score
        s = score(current_adj)
        
        if s < current_score:
            current_score = s
            no_improve_count = 0
            if s < best_score:
                best_score = s
                best_adj = current_adj[:]
        else:
            no_improve_count += 1
            # Revert if worse
            if removed:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            else:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                
        if no_improve_count > 200:
            # Random restart with a slightly perturbed Turan graph or random K4-free
            # Let's try a random graph and filter K4s
            adj_new = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.4: # Lower density to ensure K4-free is more likely
                        adj_new[i] |= (1 << j)
                        adj_new[j] |= (1 << i)
            
            # Enforce K4-free by removing edges that create K4s
            # Simple pass: remove any edge (u,v) if it completes a K4
            # This is expensive. 
            # Alternative: Start with Turan and flip some edges.
            adj_new = best_adj[:]
            for _ in range(10):
                e2 = random.choice(all_edges)
                u2, v2 = e2
                if (adj_new[u2] >> v2) & 1:
                    adj_new[u2] ^= (1 << v2)
                    adj_new[v2] ^= (1 << u2)
                else:
                    adj_new[u2] |= (1 << v2)
                    adj_new[v2] |= (1 << u2)
                if not is_k4_free(adj_new):
                    # Revert
                    if (adj_new[u2] >> v2) & 1:
                        adj_new[u2] ^= (1 << v2)
                        adj_new[v2] ^= (1 << u2)
                    else:
                        adj_new[u2] |= (1 << v2)
                        adj_new[v2] |= (1 << u2)
            
            s_new = score(adj_new)
            if s_new < current_score:
                current_adj = adj_new
                current_score = s_new
                if s_new < best_score:
                    best_score = s_new
                    best_adj = current_adj[:]
            no_improve_count = 0

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
