import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))
    
    # Generate all edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_index[(x, y)] = len(edges) - 1
                edge_index[(y, x)] = len(edges) - 1
                
    # Precompute conflicts for each edge
    # For each edge e, find the 3 other edges in each of the (d-1) 4-cycles containing e.
    # If all 3 are present, e cannot be added.
    # Store as a list of tuples of edge indices.
    conflicts = [[] for _ in range(total_edges)]
    
    for idx, (u, v) in enumerate(edges):
        # Determine the bit that differs
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u_prime = u ^ bit_j
            v_prime = v ^ bit_j
            
            # The 4-cycle is u - v - v_prime - u_prime - u
            # Edges: (u,v), (v,v_prime), (v_prime,u_prime), (u_prime,u)
            # We are adding (u,v). We need to check if the other 3 are present.
            e1 = (min(u, u_prime), max(u, u_prime))
            e2 = (min(v, v_prime), max(v, v_prime))
            e3 = (min(u_prime, v_prime), max(u_prime, v_prime))
            
            # Get indices
            try:
                idx1 = edge_index[e1]
                idx2 = edge_index[e2]
                idx3 = edge_index[e3]
                conflicts[idx].append((idx1, idx2, idx3))
            except KeyError:
                # Should not happen for hypercube edges
                pass

    def can_add(idx, selected):
        for conflict in conflicts[idx]:
            if all(k in selected for k in conflict):
                return False
        return True

    best_count = 0
    best_edges = []
    
    # Time limit management
    start_time = time.time()
    time_limit = 115 # Leave some buffer
    
    # Number of iterations depends on speed, but 1000 is likely fast enough
    # Each iteration is ~ 448 * 6 checks = ~2688 set lookups. Very fast.
    # We can run many iterations.
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        # Shuffle edges
        current_edges = edges[:]
        rng.shuffle(current_edges)
        
        selected = set()
        count = 0
        
        for e in current_edges:
            idx = edge_index[e]
            if can_add(idx, selected):
                selected.add(idx)
                count += 1
        
        if count > best_count:
            best_count = count
            # Reconstruct the list of edges [u, v]
            best_edges = [list(edges[i]) for i in selected]
            
        # If we hit the theoretical upper bound, we can stop
        if best_count >= 308:
            break

    return best_edges
