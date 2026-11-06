import time
import random

def construct():
    # Try to solve for n=35 first, then 34, etc.
    for N in [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        start_time = time.time()
        try:
            success, edges = solve(N, start_time)
            if success:
                return {'n': N, 'edges': edges}
        except Exception:
            continue
            
    # Fallback to known valid 17 (Paley graph)
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def solve(N, start_time):
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * N
    # triangles: list of bitmasks representing triangles
    triangles = []
    # ind4: list of bitmasks representing independent sets of size 4
    ind4 = []
    
    # all_prev_mask[i] = (1 << i) - 1
    all_prev = [0] * (N + 1)
    for i in range(N + 1):
        all_prev[i] = (1 << i) - 1

    def is_triangle_free(S):
        # Check if S contains any triangle from the list
        for tri in triangles:
            if (tri & S) == tri:
                return False
        return True

    def no_ind5_in(T):
        # Check if T contains an independent set of size 5
        # This is true if there exists I in ind4 such that I is subset of T
        # and there exists v in T \ I such that v is non-adjacent to all u in I
        # Wait, if I is independent and v is non-adjacent to I, then I U {v} is independent.
        # So we need to check if there is an I in ind4 with I subset of T,
        # such that the set of vertices in T that are non-adjacent to all vertices in I is non-empty.
        # Let NonAdj_I = {v in T | (adj[v] & I) == 0}
        # If NonAdj_I is not empty, then we have an independent set of size 5.
        # Note: v must be distinct from vertices in I. Since I subset T, and we check v in T,
        # if v is in I, (adj[v] & I) is 0 because I is independent? No, adj[v] & I checks neighbors.
        # If v is in I, v is not a neighbor of itself, so (adj[v] & I) does not include v.
        # But we need v to be non-adjacent to *all* u in I.
        # If v is in I, it is non-adjacent to other u in I.
        # So if I subset T, and we pick v in I, then v is non-adjacent to I \ {v}.
        # But we need v non-adjacent to ALL u in I. v is not adjacent to itself.
        # So yes, if I subset T, then any v in I is non-adjacent to all u in I (except itself, but self-loop is not an edge).
        # Wait, independent set definition: no edges between distinct vertices.
        # So v is non-adjacent to all u in I \ {v}.
        # So I U {v} is independent if v is non-adjacent to all u in I.
        # This is always true for v in I.
        # So if I subset T, then T contains an independent set of size 4.
        # Does it contain size 5?
        # We need a 5th vertex.
        # So we need to check if there is an I in ind4 such that I subset T AND |T| >= 5?
        # No, we need a specific 5th vertex.
        # If I subset T, we need to find v in T such that v is non-adjacent to all u in I.
        # If we pick v in I, it works? No, v must be distinct from u?
        # Independent set is a set of vertices. {v, u1, u2, u3, u4}.
        # If v = u1, the set is {u1, u1, u2, u3, u4} which is not a set of 5 distinct vertices.
        # So v must be in T \ I.
        # So we need: exists I in ind4 such that I subset T AND there exists v in T \ I such that (adj[v] & I) == 0.
        
        if len(T) < 5: # T is a bitmask, so we need bit_count
            if T.bit_count() < 5:
                return True
                
        for I in ind4:
            if (I & T) == I:
                # I is subset of T
                # Check if there is v in T \ I non-adjacent to I
                # T \ I is (T & ~I)
                # We need to check if any v in (T & ~I) has (adj[v] & I) == 0
                # This is a bit expensive if done naively.
                # But ind4 size is limited.
                # We can precompute for each I, the set of vertices non-adjacent to I?
                # No, adj changes? No, adj is fixed for previous vertices.
                # We can compute this on the fly.
                remaining = T & ~I
                while remaining:
                    lsb = remaining & -remaining
                    v = lsb.bit_length() - 1
                    if (adj[v] & I) == 0:
                        return False # Found ind5
                    remaining ^= lsb
        return True

    def backtrack(i):
        if time.time() - start_time > 110:
            return False
            
        if i == N:
            return True
            
        # Generate candidate neighbor sets for vertex i
        # We try random subsets of {0, ..., i-1}
        # To improve success rate, we can bias towards certain densities
        # For R(4,6), degree is roughly N/2?
        # Let's try a few random attempts
        
        attempts = 0
        max_attempts = 10000 if i < 20 else 5000
        
        while attempts < max_attempts:
            attempts += 1
            
            # Random subset
            # Probability 0.5
            S = 0
            for j in range(i):
                if random.random() < 0.5:
                    S |= (1 << j)
            
            # Check Triangle Constraint
            if not is_triangle_free(S):
                continue
                
            # Check Ind5 Constraint
            T = all_prev[i] & ~S
            if not no_ind5_in(T):
                continue
                
            # Valid S found
            adj[i] = S
            
            # Update Triangles
            # New triangles: (u, v, i) where u,v in S and u~v
            # We need to find all pairs u,v in S that are adjacent.
            # u < v < i
            new_tris = []
            for u in range(i):
                if (S >> u) & 1:
                    # Neighbors of u in S
                    # u's neighbors are in adj[u]
                    # We need v in S such that v is neighbor of u
                    # v > u to avoid duplicates?
                    # Triangles are stored as bitmasks, so order doesn't matter for the mask,
                    # but we need to generate unique masks.
                    # Mask for triangle (u, v, i) is (1<<u) | (1<<v) | (1<<i)
                    candidates = S & adj[u] & ~( (1 << (u+1)) - 1 ) # v > u
                    while candidates:
                        lsb = candidates & -candidates
                        v = lsb.bit_length() - 1
                        tri_mask = (1 << u) | (1 << v) | (1 << i)
                        new_tris.append(tri_mask)
                        candidates ^= lsb
                        
            triangles.extend(new_tris)
            
            # Update Ind4
            # New ind4s involving i: (u, v, w, i) where u,v,w in T and are independent
            # u,v,w must form an independent set of size 3 in the existing graph, restricted to T.
            # We don't maintain ind3.
            # We can find new ind4s by checking all ind3s?
            # We don't have ind3.
            # We can generate ind4s involving i by:
            # For each pair (u, v) in T that are non-adjacent?
            # No, we need 3 vertices u,v,w in T that are mutually non-adjacent.
            # This is finding independent sets of size 3 in G[T].
            # This is hard.
            # Alternative: We only need ind4 for the check `no_ind5_in`.
            # If we don't update ind4, we will miss new ind4s, and thus might accept invalid T.
            # So we MUST update ind4.
            
            # How to find new ind4s involving i?
            # An ind4 is {i, u, v, w} where u,v,w in T and u,v,w are mutually non-adjacent.
            # So we need to find all independent sets of size 3 in the subgraph induced by T.
            # Let's iterate through pairs of non-adjacent vertices in T?
            # u, v in T, not adjacent.
            # Then find w in T \ {u, v} such that w is not adjacent to u and w is not adjacent to v.
            # And u,v,w distinct.
            
            new_inds = []
            # Get list of vertices in T
            verts_in_T = []
            tmp = T
            while tmp:
                lsb = tmp & -tmp
                verts_in_T.append(lsb.bit_length() - 1)
                tmp ^= lsb
                
            # Iterate through triples? O(|T|^3). |T| ~ i/2. For i=30, 15^3 = 3375. Acceptable.
            for idx1 in range(len(verts_in_T)):
                u = verts_in_T[idx1]
                for idx2 in range(idx1 + 1, len(verts_in_T)):
                    v = verts_in_T[idx2]
                    # Check if u, v are non-adjacent
                    if (adj[u] >> v) & 1:
                        continue
                    for idx3 in range(idx2 + 1, len(verts_in_T)):
                        w = verts_in_T[idx3]
                        # Check if w is non-adjacent to u and v
                        if ((adj[w] >> u) & 1) or ((adj[w] >> v) & 1):
                            continue
                        # u, v, w are independent
                        ind_mask = (1 << u) | (1 << v) | (1 << w) | (1 << i)
                        new_inds.append(ind_mask)
                        
            ind4.extend(new_inds)
            
            if backtrack(i + 1):
                return True
            
            # Backtrack
            # Remove new_tris
            del triangles[-len(new_tris):]
            # Remove new_inds
            del ind4[-len(new_inds):]
            adj[i] = 0
            
        return False

    return backtrack(0), None
