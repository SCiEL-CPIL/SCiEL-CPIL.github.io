import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate Clebsch Graph
    # Vertices 0..15 are 5-bit even parity strings
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    adj = [0] * 16
    clebsch_edges = []
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                adj[u_idx] |= (1 << v_idx)
                adj[v_idx] |= (1 << u_idx)
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges)  # 40
    
    # 2. Efficient 3-colorability check using bitmasks and backtracking
    def is_3_colorable(adj_mask):
        n = 16
        colors = [-1] * n
        
        # Precompute neighbor lists
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            mask = adj_mask[i]
            for j in range(n):
                if mask & (1 << j):
                    neighbors[i].append(j)
        
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda x: -len(neighbors[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try each color
            for c in range(3):
                valid = True
                for u in neighbors[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    # 3. Greedy edge addition to Clebsch graph
    all_pairs = [(u, v) for u in range(16) for v in range(u+1, 16)]
    clebsch_set = set((min(u,v), max(u,v)) for u,v in clebsch_edges)
    candidate_edges = [(u,v) for u,v in all_pairs if (u,v) not in clebsch_set]
    
    def can_add_edge(adj_mask, u, v):
        common = adj_mask[u] & adj_mask[v]
        return common == 0
    
    # Local search: try to add edges greedily
    def try_add_edges(adj_mask, current_edges, current_count):
        nonlocal best_edges, best_count
        n = 16
        available = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj_mask[u] & (1 << v)):
                    if can_add_edge(adj_mask, u, v):
                        available.append((u, v))
        
        if not available:
            return current_edges, current_count
        
        # Try adding each available edge
        for u, v in available:
            if time.time() > deadline:
                break
            # Add edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            new_edges = current_edges + [[u, v]]
            new_count = current_count + 1
            
            # Check if still 4-chromatic (not 3-colorable)
            if not is_3_colorable(adj_mask):
                # Found better graph
                if new_count > best_count:
                    best_edges = new_edges
                    best_count = new_count
                # Continue trying to add more edges
                new_edges, new_count = try_add_edges(adj_mask, new_edges, new_count)
            else:
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        return current_edges, current_count
    
    # Start local search from Clebsch graph
    try_add_edges(adj[:], best_edges, best_count)
    
    # 4. Random restarts with greedy construction
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        random.shuffle(candidate_edges)
        adj_mask = [0] * 16
        edges = []
        
        # Start from a random subset of Clebsch edges plus random additions
        for u, v in candidate_edges:
            if time.time() > deadline:
                break
            if can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
        
        # Also add some Clebsch edges
        for u, v in clebsch_edges:
            if not (adj_mask[u] & (1 << v)):
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([u, v])
        
        if len(edges) > best_count:
            if not is_3_colorable(adj_mask):
                best_edges = edges
                best_count = len(edges)
                # Try to add more
                new_edges, new_count = try_add_edges(adj_mask[:], edges, len(edges))
    
    return best_edges
