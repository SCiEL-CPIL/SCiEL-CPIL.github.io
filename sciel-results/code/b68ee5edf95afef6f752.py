import random
import time

def construct():
    """
    Uses Backtracking with Random Restarts and O(n^2) local violation checks.
    """
    start_time = time.time()
    time_limit = 115

    def edge_index(n, u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    def check_violation_with_edge(n, adj, u, v):
        """
        Checks if edge (u,v) in graph adj is part of a K_{2,2,2}.
        adj is a list of bitmasks.
        """
        # u, v are connected in adj
        if not (adj[u] & (1 << v)):
            return False
            
        common = adj[u] & adj[v]
        if common & (common - 1) == 0:
            return False
            
        common_bits = common
        verts = []
        while common_bits:
            lsb = common_bits & (-common_bits)
            idx = (lsb.bit_length() - 1)
            verts.append(idx)
            common_bits ^= lsb
            
        for i_x in range(len(verts)):
            x = verts[i_x]
            for i_y in range(i_x + 1, len(verts)):
                y = verts[i_y]
                mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                cand_a = adj[v] & adj[x] & adj[y] & mask_excl
                if cand_a == 0: continue
                cand_b = adj[u] & adj[x] & adj[y] & mask_excl
                if cand_b == 0: continue
                return True
        return False

    def backtrack(n, colors, adj0, adj1, current_edge_idx, edge_pairs):
        if current_edge_idx >= n * (n - 1) // 2:
            return True
            
        u, v = edge_pairs[current_edge_idx]
        
        # Try color 0
        colors[current_edge_idx] = 0
        adj0[u] |= (1 << v)
        adj0[v] |= (1 << u)
        
        if not check_violation_with_edge(n, adj0, u, v):
            if backtrack(n, colors, adj0, adj1, current_edge_idx + 1, edge_pairs):
                return True
        
        # Undo color 0
        adj0[u] ^= (1 << v)
        adj0[v] ^= (1 << u)
        
        # Try color 1
        colors[current_edge_idx] = 1
        adj1[u] |= (1 << v)
        adj1[v] |= (1 << u)
        
        if not check_violation_with_edge(n, adj1, u, v):
            if backtrack(n, colors, adj0, adj1, current_edge_idx + 1, edge_pairs):
                return True
        
        # Undo color 1
        adj1[u] ^= (1 << v)
        adj1[v] ^= (1 << u)
        
        return False

    for n in range(30, 16, -1):
        if time.time() - start_time > time_limit:
            break
            
        edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edge_pairs)
        
        # Random restarts
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Randomize the order of edges to break symmetry
            # We need to keep track of the original indices for the colors array
            # So we shuffle a list of indices
            indices = list(range(num_edges))
            random.shuffle(indices)
            
            # We need a mapping from shuffled position to original (u,v)
            # But the colors array is indexed by original index.
            # So we just iterate through the shuffled indices.
            
            colors = [-1] * num_edges
            adj0 = [0] * n
            adj1 = [0] * n
            
            # We implement iterative backtracking to avoid recursion depth?
            # n=30, depth 435. Python recursion limit is usually 1000. So it's fine.
            
            # To use random restarts effectively, we can also randomize the value choice (0 vs 1)
            # But the shuffle of edges is the main randomization.
            
            if backtrack(n, colors, adj0, adj1, 0, [edge_pairs[i] for i in indices]):
                return {'n': n, 'colors': colors}

    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
