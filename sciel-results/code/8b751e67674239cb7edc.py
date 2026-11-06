import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 325
    
    def time_left():
        return deadline - time.time()

    # 1. Build Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    # We will try to construct a 42 vertex graph.
    # If we fail, we return the 37 vertex graph.
    # Best valid graph found so far
    best_n = q
    best_adj = [0] * q  # Bitmasks
    
    # Initialize Paley 37 bitmasks
    for i in range(q):
        mask = 0
        for j in range(q):
            if i != j and (j - i) % q in residues:
                mask |= (1 << j)
        best_adj[i] = mask

    # Target size
    target_n = 42
    
    # Initialize a 42 vertex graph
    # Vertices 0-36 are Paley
    # Vertices 37-41 are new
    # Randomly connect new vertices to old ones and each other
    random.seed(42)
    
    def create_initial_graph():
        adj = best_adj[:]
        # Extend adj to 42
        for i in range(q, target_n):
            adj.append(0)
        
        # Connect new vertices to old ones randomly
        for i in range(q, target_n):
            mask = 0
            for j in range(q):
                if random.random() < 0.5:
                    mask |= (1 << j)
            adj[i] = mask
            # Update old vertices' masks to include new ones
            for j in range(q):
                if (mask >> j) & 1:
                    adj[j] |= (1 << i)
            
            # Connect new vertices to other new vertices
            for k in range(q, i):
                if random.random() < 0.5:
                    adj[i] |= (1 << k)
                    adj[k] |= (1 << i)
        
        return adj

    def count_triangles_in_set(adj, vertices):
        """Count triangles in the subgraph induced by vertices."""
        count = 0
        v_list = list(vertices)
        n_v = len(v_list)
        for i in range(n_v):
            u = v_list[i]
            for j in range(i + 1, n_v):
                v = v_list[j]
                if (adj[u] >> v) & 1:
                    # Check common neighbors in v_list
                    # w > v to avoid double counting? 
                    # Actually, just check if any w in v_list connects to both
                    # For efficiency, we can just check all w in v_list
                    for k in range(j + 1, n_v):
                        w = v_list[k]
                        if ((adj[u] >> w) & 1) and ((adj[v] >> w) & 1):
                            count += 1
        return count

    def count_ind_triangles_in_set(adj, vertices):
        """Count independent triangles in the subgraph induced by vertices."""
        count = 0
        v_list = list(vertices)
        n_v = len(v_list)
        for i in range(n_v):
            u = v_list[i]
            for j in range(i + 1, n_v):
                v = v_list[j]
                if not ((adj[u] >> v) & 1):
                    for k in range(j + 1, n_v):
                        w = v_list[k]
                        if not ((adj[u] >> w) & 1) and not ((adj[v] >> w) & 1):
                            count += 1
        return count

    def get_violations(adj, n):
        """Count K5 and independent 5-sets."""
        k5_count = 0
        ind5_count = 0
        
        # Count K5
        # Iterate over edges (u, v)
        for u in range(n):
            mask_u = adj[u]
            for v in range(u + 1, n):
                if (mask_u >> v) & 1:
                    # Common neighbors
                    common = mask_u & adj[v]
                    # Remove u and v from common? No, common neighbors are distinct from u,v
                    # We need 3 common neighbors that form a triangle.
                    # Get list of common neighbors
                    common_list = []
                    # Iterate bits
                    c = common
                    while c:
                        lsb = c & -c
                        idx = lsb.bit_length() - 1
                        common_list.append(idx)
                        c -= lsb
                    
                    if len(common_list) >= 3:
                        # Count triangles in common_list
                        k5_count += count_triangles_in_set(adj, common_list)
        
        # Count Independent 5-sets
        # Iterate over non-edges (u, v)
        for u in range(n):
            mask_u = adj[u]
            # Non-neighbors of u
            non_mask_u = ((1 << n) - 1) ^ mask_u ^ (1 << u)
            
            for v in range(u + 1, n):
                if not ((mask_u >> v) & 1):
                    # Common non-neighbors
                    non_mask_v = ((1 << n) - 1) ^ adj[v] ^ (1 << v)
                    common_non = non_mask_u & non_mask_v
                    
                    common_non_list = []
                    c = common_non
                    while c:
                        lsb = c & -c
                        idx = lsb.bit_length() - 1
                        common_non_list.append(idx)
                        c -= lsb
                    
                    if len(common_non_list) >= 3:
                        # Count independent triangles in common_non_list
                        ind5_count += count_ind_triangles_in_set(adj, common_non_list)
                        
        return k5_count + ind5_count

    def is_valid(adj, n):
        # Quick check: return 0 if no violations
        return get_violations(adj, n) == 0

    # Simulated Annealing
    current_adj = create_initial_graph()
    current_cost = get_violations(current_adj, target_n)
    
    if current_cost == 0:
        return target_n, [(i, j) for i in range(target_n) for j in range(i + 1, target_n) if (current_adj[i] >> j) & 1]

    best_adj_sa = current_adj[:]
    best_cost = current_cost
    
    T = 100.0
    cooling = 0.99
    
    # Precompute bit masks for speed?
    # The bottleneck is get_violations.
    # With n=42, edges ~ 440.
    # For each edge, common neighbors ~ 10.
    # Counting triangles in 10 nodes is fast.
    # Total ops ~ 440 * 10 * 10 = 44000. Very fast.
    # So we can do many steps.
    
    steps = 0
    while time_left() > 2:
        if T < 0.01:
            break
            
        # Pick random edge
        u = random.randint(0, target_n - 1)
        v = random.randint(0, target_n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Flip edge
        current_adj[u] ^= (1 << v)
        current_adj[v] ^= (1 << u)
        
        new_cost = get_violations(current_adj, target_n)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj_sa = current_adj[:]
                if new_cost == 0:
                    break
        else:
            # Revert
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
        T *= cooling
        steps += 1
        
        if best_cost == 0:
            break

    # If we found a valid 42 vertex graph, return it
    if best_cost == 0:
        edges = []
        for i in range(target_n):
            mask = best_adj_sa[i]
            for j in range(i + 1, target_n):
                if (mask >> j) & 1:
                    edges.append((i, j))
        return target_n, edges

    # Otherwise, return Paley 37
    edges = []
    for i in range(q):
        mask = best_adj[i]
        for j in range(i + 1, q):
            if (mask >> j) & 1:
                edges.append((i, j))
    return q, edges

# EVOLVE-BLOCK-END
