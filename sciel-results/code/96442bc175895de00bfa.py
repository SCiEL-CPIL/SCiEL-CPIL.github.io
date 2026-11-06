import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try n=47 first, then 46
    for n in [47, 46]:
        min_deg = n - 39
        
        # Use bitmasks for adjacency
        # n <= 47, fits in Python int
        adj = [0] * n
        # common[u][v] = number of common neighbors
        common = [[0] * n for _ in range(n)]
        
        def add_edge(u, v):
            # Add edge u-v
            if u == v: return False
            if (adj[u] >> v) & 1: return False # Already exists
            
            # Check if adding this edge creates a C4
            # A C4 is created if there is a w such that w is neighbor of u and v is neighbor of w? No.
            # Adding (u,v) creates a C4 if there exists w such that w is in N(u) and w is in N(v)?
            # No, that would mean u-v-w-x-u?
            # If we add (u,v), and there is a path u-w-v of length 2, then u-v-w-u is a triangle, not C4.
            # A C4 involving the new edge (u,v) must be u-v-x-y-u.
            # So we need x in N(v) and y in N(u) such that x-y is an edge?
            # Or simpler:
            # The new edge (u,v) is part of a C4 if there exists a vertex w such that
            # w is a common neighbor of u and v? No, that forms a triangle.
            # w is a neighbor of u, and there is a path v-z-w of length 2?
            # Let's use the common neighbor count.
            # Before adding (u,v), let k = common[u][v].
            # If k >= 1, then there is a common neighbor w.
            # The cycle would be u-w-v-...-u.
            # If there is another common neighbor w', then u-w-v-w'-u is a C4?
            # No, u-w-v and u-w'-v are two paths of length 2.
            # The cycle is u-w-v-w'-u. This is a C4.
            # So if common[u][v] >= 1, adding (u,v) creates a C4?
            # Wait. If common[u][v] = 1, let w be the common neighbor.
            # Edges: (u,w), (w,v). Adding (u,v) creates triangle u-w-v-u.
            # Does it create a C4?
            # A C4 requires 4 distinct vertices.
            # If common[u][v] >= 2, say w1, w2.
            # Then u-w1-v-w2-u is a C4.
            # So we must ensure common[u][v] == 0 before adding (u,v).
            
            if common[u][v] > 0:
                return False
            
            # Update common counts
            # For any w in N(u), v becomes a common neighbor of (w, v)? No.
            # v is now in N(u).
            # For any w in N(u), the pair (w, v) now has u as a common neighbor?
            # u is in N(w) (since w in N(u)) and u is in N(v) (new edge).
            # So common[w][v] increases by 1.
            # Similarly, for any w in N(v), common[w][u] increases by 1.
            
            # Iterate over neighbors of u
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                common[w][v] += 1
                common[v][w] += 1
                mask_u ^= lsb
                
            # Iterate over neighbors of v
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                common[w][u] += 1
                common[u][w] += 1
                mask_v ^= lsb
                
            # Add to adjacency
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True

        def remove_edge(u, v):
            if not ((adj[u] >> v) & 1): return False
            
            # Update common counts (reverse of add)
            mask_u = adj[u] & ~(1 << v) # Neighbors of u excluding v
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                common[w][v] -= 1
                common[v][w] -= 1
                mask_u ^= lsb
                
            mask_v = adj[v] & ~(1 << u)
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                common[w][u] -= 1
                common[u][w] -= 1
                mask_v ^= lsb
                
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True

        def get_degrees():
            return [bin(adj[i]).count('1') for i in range(n)]
            
        def find_c4():
            for u in range(n):
                for v in range(u+1, n):
                    if common[u][v] >= 2:
                        return (u, v)
            return None

        # Initialize random graph
        # We want to start with some edges.
        # Let's try a random graph with p=0.1
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.1:
                    add_edge(i, j)
                    
        # Ensure minimum degree
        # If degree is too low, try to add edges
        for _ in range(1000):
            if time.time() > deadline: break
            degs = get_degrees()
            low_verts = [i for i in range(n) if degs[i] < min_deg]
            if not low_verts:
                break
            u = random.choice(low_verts)
            # Try to add edge from u to a random vertex
            candidates = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
            random.shuffle(candidates)
            added = False
            for v in candidates:
                if add_edge(u, v):
                    added = True
                    break
            if not added:
                # Cannot add any edge to u without creating C4 or already connected
                # This might be a dead end, restart?
                # For now, just break inner loop
                pass

        # Main loop: remove edges to break C4s
        steps = 0
        while time.time() < deadline:
            c4 = find_c4()
            if c4 is None:
                # Check degree constraint
                degs = get_degrees()
                if min(degs) >= min_deg:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        mask = adj[i]
                        while mask:
                            lsb = mask & -mask
                            v = lsb.bit_length() - 1
                            if i < v:
                                edges.append([i, v])
                            mask ^= lsb
                    return {'n': n, 'edges': edges}
                else:
                    # C4-free but degree too low. Try to add edges.
                    degs = get_degrees()
                    low_verts = [i for i in range(n) if degs[i] < min_deg]
                    if low_verts:
                        u = random.choice(low_verts)
                        candidates = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                        random.shuffle(candidates)
                        for v in candidates:
                            if add_edge(u, v):
                                break
                    continue
                    
            u, v = c4
            # We need to remove an edge from the C4.
            # The C4 is formed by u, v and two common neighbors w1, w2.
            # Edges in the C4: (u,w1), (w1,v), (v,w2), (w2,u).
            # We can remove any of these.
            # We prefer removing an edge that is not critical for degree.
            
            # Find the common neighbors
            common_nbs = []
            mask = adj[u] & adj[v]
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                common_nbs.append(w)
                mask ^= lsb
            
            if len(common_nbs) < 2:
                # Should not happen if common[u][v] >= 2
                continue
                
            w1, w2 = common_nbs[0], common_nbs[1]
            
            # Candidate edges to remove: (u,w1), (w1,v), (v,w2), (w2,u)
            candidates = [(u, w1), (w1, v), (v, w2), (w2, u)]
            random.shuffle(candidates)
            
            removed = False
            for a, b in candidates:
                degs = get_degrees()
                # Check if removing (a,b) violates degree constraint
                if degs[a] > min_deg and degs[b] > min_deg:
                    remove_edge(a, b)
                    removed = True
                    break
            
            if not removed:
                # All edges are critical. 
                # We might need to accept a lower degree or restart.
                # For now, just remove one randomly and hope we can fix it later?
                # Or break and restart search for this n?
                # Let's try removing one anyway, we might fix it by adding other edges.
                a, b = candidates[0]
                remove_edge(a, b)
                
            steps += 1
            if steps > 10000:
                break
                
    # Fallback: return the seed solution or best found
    # If we didn't find 47 or 46, return 39
    return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
