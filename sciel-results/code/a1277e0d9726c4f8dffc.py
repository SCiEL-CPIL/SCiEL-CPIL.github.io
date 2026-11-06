import random
import time

def construct():
    n = 40
    # adj[u] is an integer bitmask representing neighbors of u
    adj = [0] * n
    
    def neighbors_mask(vertices_mask):
        """
        Given a bitmask of vertices, return the bitmask of all their neighbors.
        """
        mask = 0
        while vertices_mask:
            # Get lowest set bit
            lsb = vertices_mask & -vertices_mask
            idx = (lsb.bit_length() - 1)
            mask |= adj[idx]
            vertices_mask ^= lsb
        return mask

    def creates_c6(u, v):
        """
        Check if adding edge (u, v) creates a C6.
        This is true if there is a simple path of length 5 from u to v.
        """
        # D1 = N(u) \ {v}
        d1 = adj[u] & ~(1 << v)
        
        # D2 = N(D1) \ (D1 ∪ {u} ∪ {v})
        d2 = neighbors_mask(d1)
        d2 &= ~(d1 | (1 << u) | (1 << v))
        
        # D3 = N(D2) \ (D2 ∪ D1 ∪ {u} ∪ {v})
        d3 = neighbors_mask(d2)
        d3 &= ~(d2 | d1 | (1 << u) | (1 << v))
        
        # D4 = N(D3) \ (D3 ∪ D2 ∪ D1 ∪ {u} ∪ {v})
        d4 = neighbors_mask(d3)
        d4 &= ~(d4 | d3 | d2 | d1 | (1 << u) | (1 << v))
        
        # D5 = N(D4) \ (D4 ∪ D3 ∪ D2 ∪ D1 ∪ {u} ∪ {v})
        d5 = neighbors_mask(d4)
        d5 &= ~(d5 | d4 | d3 | d2 | d1 | (1 << u) | (1 << v))
        
        # Check if v is in D5
        return (d5 & (1 << v)) != 0

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges():
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lsb
        return edges

    def count_edges():
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    start_time = time.time()
    limit = 110 # seconds
    
    best_edges_list = []
    best_edge_count = 0
    best_adj = [0] * n

    for attempt in range(50):
        if time.time() - start_time > limit:
            break
            
        # Reset
        adj = [0] * n
        
        # Greedy Phase
        # Generate a random order of all pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if not creates_c6(u, v):
                add_edge(u, v)
        
        # Local Search Phase
        # Try to add edges, if stuck, do random swaps to explore
        improved = True
        while improved and (time.time() - start_time < limit - 5):
            improved = False
            
            # Try to add a random missing edge
            # We iterate through a random subset of missing edges
            # To avoid O(N^2) check every time, we can just pick random pairs
            # and check if they are missing and valid.
            
            # Try 100 random additions
            for _ in range(100):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if adj[u] & (1 << v): continue # Already exists
                
                if not creates_c6(u, v):
                    add_edge(u, v)
                    improved = True
                    break
            
            # If no edge was added, try a swap
            if not improved:
                # Pick a random existing edge
                while True:
                    u1 = random.randint(0, n-1)
                    mask = adj[u1]
                    if mask:
                        lsb = mask & -mask
                        v1 = lsb.bit_length() - 1
                        break
                    
                # Pick a random missing edge
                while True:
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 == v2: continue
                    if not (adj[u2] & (1 << v2)):
                        break

                # Remove u1-v1
                remove_edge(u1, v1)
                
                # Try to add u2-v2
                if not creates_c6(u2, v2):
                    add_edge(u2, v2)
                    improved = True
                else:
                    # Revert
                    add_edge(u1, v1)
        
        # Check if this graph is better
        current_edges = count_edges()
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_adj = adj[:]
            best_edges_list = get_edges()

    return best_edges_list
