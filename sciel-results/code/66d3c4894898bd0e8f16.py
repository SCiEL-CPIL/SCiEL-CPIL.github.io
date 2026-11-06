import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Helper to create triangle mask from u, v, w
    def tri_mask(u, v, w):
        return (1 << u) | (1 << v) | (1 << w)

    def get_initial_triangles(adj):
        tris = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if w > v:
                            if adj[v] & (1 << w):
                                tris.add(tri_mask(u, v, w))
                        temp &= temp - 1
        return tris

    def can_add_edge_no_k4(adj, u, v):
        if adj[u] & (1 << v):
            return False
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            # If w is connected to any other vertex in C, we have a K4
            # Specifically, if adj[w] intersects C (excluding u and v, but u,v are not in C usually? 
            # C is common neighbors of u and v. If w in C is connected to x in C, then u,v,w,x form K4?
            # No. u-v is the edge. u-w and v-w exist. If w-x exists, do we have K4?
            # The K4 would be {u, v, w, x}.
            # Edges: (u,v) exists. (u,w) exists. (v,w) exists.
            # We need (w,x), (u,x), (v,x).
            # x is in C, so (u,x) and (v,x) exist.
            # So if any w in C has an edge to another x in C, we form a K4.
            if adj[w] & C:
                return False
            temp &= temp - 1
        return True

    def add_edge(adj, u, v, triangles):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            if w > v:
                triangles.add(tri_mask(u, v, w))
            elif w < v:
                triangles.add(tri_mask(u, w, v)) # u < w < v? No, u < v is fixed.
                # Triangles are unordered sets of vertices. 
                # My tri_mask doesn't care about order.
                # But I need to ensure I add the correct triangle.
                # The triangle is {u, v, w}.
                # It's fine to just add tri_mask(u, v, w).
                # But I must avoid duplicates. Since I use a set, duplicates are handled.
                pass
            temp &= temp - 1
        # Re-do logic: just iterate all w in C and add tri_mask(u, v, w)
        # The set will handle duplicates.
        
    def remove_edge(adj, u, v, triangles):
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        # Find triangles containing u and v
        # These are formed by common neighbors of u and v (before removal)
        # But we already removed the edge. We need the common neighbors BEFORE removal.
        # So we should calculate C before modifying adj.
        # Let's swap logic: calculate C, then modify adj, then remove tris.
        
    # Revised incremental approach
    def try_add_edge(adj, u, v, triangles):
        if adj[u] & (1 << v):
            return False
        if not can_add_edge_no_k4(adj, u, v):
            return False
        
        # Calculate new triangles
        C = adj[u] & adj[v]
        new_tris = []
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            new_tris.append(tri_mask(u, v, w))
            temp &= temp - 1
            
        # Apply change
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        for t in new_tris:
            triangles.add(t)
            
        return True

    def try_remove_edge(adj, u, v, triangles):
        if not (adj[u] & (1 << v)):
            return False
            
        # Calculate triangles to remove
        C = adj[u] & adj[v]
        tris_to_remove = []
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            tris_to_remove.append(tri_mask(u, v, w))
            temp &= temp - 1
            
        # Apply change
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        for t in tris_to_remove:
            triangles.discard(t)
            
        return True

    def max_triangle_free_size(adj, triangles):
        # Fast branch-and-bound for max triangle-free induced subgraph
        # A set S is triangle-free if no triangle (from `triangles`) is fully contained in S.
        # We want max |S|.
        
        tri_list = list(triangles)
        if not tri_list:
            return n
        
        # Sort triangles by first vertex for pruning
        tri_list.sort()
        
        # Precompute: for each vertex, which triangles contain it
        # tri_vertex_mask[v] = bitmask of triangle indices that contain vertex v
        # Actually, let's use a different approach:
        # For each vertex v, compute the set of triangles that would be "hit" if v is excluded.
        
        # Better approach: branch and bound on vertices
        # State: current_mask (vertices in S), remaining_vertices (candidates to add)
        # Pruning: if current_size + remaining_count <= best, stop
        
        best = [0]
        
        # For quick check: can we add vertex v to current set without creating a triangle?
        # A triangle is created if there exist two vertices u, w in current set such that
        # {u, v, w} is a triangle.
        
        # Precompute for each pair (u, w): the set of vertices v such that {u,v,w} is a triangle
        # Actually, let's precompute: for each vertex v, the "conflict" with existing set.
        
        # Simpler: for each triangle t = (a,b,c), if any two of {a,b,c} are in current set,
        # then the third cannot be added.
        
        # Let's use a different formulation:
        # A set S is triangle-free iff for every triangle (a,b,c), S does not contain all three.
        # Equivalently, S is a vertex cover complement... no.
        # S is triangle-free iff V\S is a triangle transversal (hits every triangle).
        # So max |S| = n - min |T| where T is a minimum triangle transversal.
        
        # Let's solve min triangle transversal via branch and bound.
        # This is a hitting set problem: find minimum T such that T intersects every triangle.
        
        tri_list = list(triangles)
        if not tri_list:
            return n
            
        m = len(tri_list)
        
        # For each triangle, store its 3 vertices
        tri_verts = []
        for t in tri_list:
            verts = []
            for v in range(n):
                if t & (1 << v):
                    verts.append(v)
            tri_verts.append(verts)
        
        # Branch and bound for minimum hitting set
        best_size = [n]  # worst case: take all vertices
        
        def hit_bound(remaining_tris, current_size, used_mask):
            if not remaining_tris:
                if current_size < best_size[0]:
                    best_size[0] = current_size
                return
            
            if current_size >= best_size[0]:
                return
            
            # Lower bound: we need at least ceil(remaining_tris / max_coverage) more vertices
            # where max_coverage is the max number of remaining triangles a single vertex can hit.
            # Quick lower bound: 1 if remaining_tris > 0
            if current_size + 1 >= best_size[0]:
                # Even adding 1 more won't help if we already have best_size[0]-1
                # Actually we need to check if we can solve with current_size + (min_needed)
                pass
            
            # Pick the triangle with fewest available (unused) vertices to branch on
            # This is the "most constrained" triangle
            min_avail = None
            min_avail_count = 999
            for i, verts in enumerate(remaining_tris):
                avail = [v for v in verts if not (used_mask & (1 << v))]
                if len(avail) < min_avail_count:
                    min_avail_count = len(avail)
                    min_avail = avail
                if min_avail_count == 0:
                    # Triangle with no available vertices - impossible, but shouldn't happen
                    return
                if min_avail_count == 1:
                    break
            
            if min_avail is None or min_avail_count == 0:
                return
            
            # Branch: for each vertex in min_avail, try adding it to the hitting set
            for v in min_avail:
                new_used = used_mask | (1 << v)
                # Remove all triangles that contain v
                new_remaining = []
                for verts in remaining_tris:
                    if v not in verts:
                        new_remaining.append(verts)
                hit_bound(new_remaining, current_size + 1, new_used)
        
        hit_bound(tri_verts, 0, 0)
        
        return n - best_size[0]

    def spectral_init(rng):
        """Initialize using spectral approach: create a continuous adjacency,
        optimize to balance K4-free constraint and triangle-free subgraph size,
        then round to get a good starting point."""
        # Start with a random symmetric matrix
        A = np.random.uniform(0, 1, (n, n))
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        # Gradient descent to maximize "density" while penalizing K4s
        # and encouraging small triangle-free subgraphs
        lr = 0.1
        for step in range(50):
            # Compute K4 penalty: for each 4-clique, add penalty
            # Approximate: use eigenvalues or trace-based penalties
            # Simpler: just threshold and check
            B = (A > 0.5).astype(int)
            # Count K4s approximately
            # Just do a few gradient steps based on current structure
            # For now, just randomize with some structure
            pass
        
        # Threshold to get binary adjacency
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if A[u][v] > 0.5:
                    # Check K4-free
                    if can_add_edge_no_k4(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def local_search(adj, triangles, current_val, rng, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if not (adj[u] & (1 << v)):
                        if can_add_edge_no_k4(adj, u, v):
                            # Tentative add
                            # We need to update triangles and check value
                            # To avoid modifying and reverting complex state, we can simulate
                            # But incremental update is fast.
                            # Let's do it in place and revert if bad.
                            
                            # Save state? No, just do it.
                            # Calculate new triangles list
                            C = adj[u] & adj[v]
                            new_tris = []
                            temp = C
                            while temp:
                                w = (temp & -temp).bit_length() - 1
                                new_tris.append(tri_mask(u, v, w))
                                temp &= temp - 1
                            
                            # Apply
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            for t in new_tris:
                                triangles.add(t)
                                
                            new_val = max_triangle_free_size(adj, triangles)
                            
                            if new_val < current_val:
                                current_val = new_val
                                improved = True
                            else:
                                # Revert
                                for t in new_tris:
                                    triangles.discard(t)
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
            
            # Try removing edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if adj[u] & (1 << v):
                        # Calculate triangles to remove
                        C = adj[u] & adj[v]
                        tris_to_remove = []
                        temp = C
                        while temp:
                            w = (temp & -temp).bit_length() - 1
                            tris_to_remove.append(tri_mask(u, v, w))
                            temp &= temp - 1
                        
                        # Apply
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        for t in tris_to_remove:
                            triangles.discard(t)
                        
                        new_val = max_triangle_free_size(adj, triangles)
                        
                        if new_val < current_val:
                            current_val = new_val
                            improved = True
                        else:
                            # Revert
                            for t in tris_to_remove:
                                triangles.add(t)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
        return adj, triangles, current_val

    best_val = n
    best_adj = None
    best_triangles = None
    
    rng = random.Random(42)
    
    # Strategy 1: Tripartite K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    triangles = get_initial_triangles(adj)
    val = max_triangle_free_size(adj, triangles)
    
    if val < best_val:
        best_val = val
        best_adj = adj[:]
        best_triangles = triangles.copy()
    
    # Strategy 2: Spectral-inspired init
    if time.time() < deadline - 30:
        adj_s = spectral_init(rng)
        triangles_s = get_initial_triangles(adj_s)
        val_s = max_triangle_free_size(adj_s, triangles_s)
        if val_s < best_val:
            best_val = val_s
            best_adj = adj_s[:]
            best_triangles = triangles_s.copy()
    
    # Strategy 3: Random Greedy + Local Search with more restarts
    # Since max_triangle_free_size is now fast (branch and bound), we can do more iterations
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random init with different densities
        adj = [0] * n
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(possible_edges)
        
        # Vary the edge probability threshold
        if i % 3 == 0:
            # Dense
            for u, v in possible_edges:
                if can_add_edge_no_k4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        elif i % 3 == 1:
            # Medium - only add if random < 0.7
            for u, v in possible_edges:
                if rng.random() < 0.7 and can_add_edge_no_k4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        else:
            # Sparse - only add if random < 0.4
            for u, v in possible_edges:
                if rng.random() < 0.4 and can_add_edge_no_k4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
        triangles = get_initial_triangles(adj)
        val = max_triangle_free_size(adj, triangles)
        
        if val < best_val:
            best_val = val
            best_adj = adj[:]
            best_triangles = triangles.copy()
            
        # Local Search - can do more since evaluation is faster
        remaining = deadline - time.time()
        if remaining > 5:
            time_limit = time.time() + min(remaining - 5, 30)
            adj, triangles, val = local_search(adj, triangles, val, rng, time_limit)
            
            if val < best_val:
                best_val = val
                best_adj = adj[:]
                best_triangles = triangles.copy()
                
        if best_val <= 8:
            break

    # Final polish on best found
    if time.time() < deadline - 5:
        time_limit = deadline - 2
        best_adj, best_triangles, best_val = local_search(best_adj, best_triangles, best_val, rng, time_limit)
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
