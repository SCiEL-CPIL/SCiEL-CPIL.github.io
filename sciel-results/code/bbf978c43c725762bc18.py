import random
import time

def is_3_colorable_mask(adj):
    n = 16
    # Order vertices by degree descending
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    color = [-1] * n
    # Precompute neighbor lists for each vertex in order
    neighbor_lists = []
    for u in order:
        mask = adj[u]
        neighbors = []
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            neighbors.append(v)
            mask ^= lsb
        neighbor_lists.append(neighbors)
    
    def backtrack(idx):
        if idx == n:
            return True
        u = order[idx]
        # Check neighbors
        forbidden = 0
        for v in neighbor_lists[idx]:
            cv = color[v]
            if cv != -1:
                forbidden |= (1 << cv)
        
        # Try colors - try the most constrained first
        if (forbidden & 1) == 0:
            color[u] = 0
            if backtrack(idx + 1):
                return True
        if (forbidden & 2) == 0:
            color[u] = 1
            if backtrack(idx + 1):
                return True
        if (forbidden & 4) == 0:
            color[u] = 2
            if backtrack(idx + 1):
                return True
        color[u] = -1
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 138  # Safety margin
    
    best_edges = None
    best_count = -1
    
    # Correct Grötzsch graph edges (11 vertices: 0-10)
    # Vertices 0-4: C5
    # Vertices 5-9: Mycielski copies of 0-4
    # Vertex 10: Apex
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # C5
        [5, 4], [5, 1],  # w0 (5) connected to neighbors of v0 (4,1)
        [6, 0], [6, 2],  # w1 (6) connected to neighbors of v1 (0,2)
        [7, 1], [7, 3],  # w2 (7) connected to neighbors of v2 (1,3)
        [8, 2], [8, 4],  # w3 (8) connected to neighbors of v3 (2,4)
        [9, 3], [9, 0],  # w4 (9) connected to neighbors of v4 (3,0)
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]  # x (10) connected to all w
    ]
    
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        # Initialize adj masks
        adj = [0] * 16
        
        # Add base edges
        for u, v in groetzsch_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Vertices 11-15 are isolated initially.
        
        # Generate candidate edges in random order
        all_possible = [(u, v) for u in range(16) for v in range(u+1, 16)]
        random.shuffle(all_possible)
        
        current_edge_count = len(groetzsch_edges)
        
        # First pass: add all triangle-free edges that maintain non-3-colorability
        for u, v in all_possible:
            if time.time() > deadline:
                break
            
            # Check if edge exists
            if (adj[u] >> v) & 1:
                continue
                
            # Check triangle free
            if (adj[u] & adj[v]) != 0:
                continue
                
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if graph is still NOT 3-colorable
            if is_3_colorable_mask(adj):
                # Remove
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                # Keep
                current_edge_count += 1
        
        # Second pass: try to add more edges in a different order
        if time.time() < deadline - 5:
            all_possible2 = [(u, v) for u in range(16) for v in range(u+1, 16)]
            random.shuffle(all_possible2)
            
            for u, v in all_possible2:
                if time.time() > deadline:
                    break
                
                # Check if edge exists
                if (adj[u] >> v) & 1:
                    continue
                    
                # Check triangle free
                if (adj[u] & adj[v]) != 0:
                    continue
                    
                # Tentatively add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if graph is still NOT 3-colorable
                if is_3_colorable_mask(adj):
                    # Remove
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    # Keep
                    current_edge_count += 1
                    
        if current_edge_count > best_count:
            best_count = current_edge_count
            # Reconstruct edge list
            best_edges = []
            for u in range(16):
                for v in range(u+1, 16):
                    if (adj[u] >> v) & 1:
                        best_edges.append([u, v])
                        
    if best_edges is None:
        return groetzsch_edges
    return best_edges
