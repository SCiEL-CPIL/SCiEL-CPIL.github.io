import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 18
    
    # Helper to check K4-free using neighborhood triangle check
    def is_k4_free(adj):
        # adj is a list of bitmasks, adj[u] has bits set for neighbors of u
        for u in range(n):
            # Get neighbors of u
            neigh_mask = adj[u]
            # Check if any triangle exists in neigh_mask
            # A triangle in neigh_mask means there are v, w in neigh_mask such that 
            # v-w is an edge and v-u, w-u are edges (which is true by definition of neigh_mask)
            # So we just need to check if the induced subgraph on neigh_mask has a triangle.
            # Iterate through bits in neigh_mask
            bits = []
            m = neigh_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                bits.append(v)
                m ^= lsb
            
            # Check all pairs in bits
            for i in range(len(bits)):
                v = bits[i]
                for j in range(i+1, len(bits)):
                    w = bits[j]
                    # Check if v and w are connected
                    if (adj[v] >> w) & 1:
                        return False
        return True

    def get_triangles(adj):
        """Returns a list of bitmasks representing triangles in the graph."""
        triangles = []
        # For each vertex u, look at its neighbors
        # A triangle is (u, v, w) where v, w are neighbors of u and v-w is an edge.
        # To avoid duplicates, we can just collect all sets {u,v,w}.
        # Since n=18, we can just iterate all triples? 18 choose 3 = 816. Very fast.
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    for w in range(v+1, n):
                        if (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                            mask = (1 << u) | (1 << v) | (1 << w)
                            triangles.append(mask)
        return triangles

    def max_triangle_free_size(adj, triangles):
        """
        Find the largest subset of vertices inducing a triangle-free subgraph.
        A subset S is triangle-free if it does not contain any triangle from the list 'triangles'.
        """
        # If no triangles, the whole graph is triangle-free
        if not triangles:
            return n
            
        # We want the max k such that there exists a subset S of size k with no triangle.
        # Iterate k from n down to 1
        for k in range(n, 0, -1):
            # Check all subsets of size k
            # Using itertools.combinations is fine for n=18
            found = False
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check if this mask contains any triangle
                is_tf = True
                for t in triangles:
                    if (mask & t) == t:
                        is_tf = False
                        break
                if is_tf:
                    found = True
                    break
            if found:
                return k
        return 1 # Should not reach here for n>=1

    def score(adj):
        # Calculate triangles first, then max size
        tris = get_triangles(adj)
        return max_triangle_free_size(adj, tris)

    # Initialize with K6,6,6
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
            
    # Local Search
    no_improve_count = 0
    while time.time() < deadline:
        # Pick a random edge to flip
        e = random.choice(all_edges)
        u, v = e
        
        # Flip edge
        if (current_adj[u] >> v) & 1:
            # Remove edge
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
        else:
            # Add edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
        # Check K4-free
        if not is_k4_free(current_adj):
            # Revert
            if (current_adj[u] >> v) & 1:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
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
            # Accept worse moves with some probability (Simulated Annealing style) or just revert?
            # For simplicity, revert if worse, but we might get stuck.
            # Let's revert if worse, but occasionally do a random restart if stuck.
            if (current_adj[u] >> v) & 1:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
        if no_improve_count > 100:
            # Random restart
            # Try a random graph
            adj_new = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj_new[i] |= (1 << j)
                        adj_new[j] |= (1 << i)
            if is_k4_free(adj_new):
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

