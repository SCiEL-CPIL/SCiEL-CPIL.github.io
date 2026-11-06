import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    
    # Build PG(2,5) points
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity graph construction
    # p ~ q if dot(p, q) == 0 mod 5
    adj_mask = [0] * n
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q == 0:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edges.append((i, j))
                
    polarity_edges = len(edges)
    assert polarity_edges == 93, f"Expected 93 edges, got {polarity_edges}"
    
    # Precompute neighbor lists as lists of indices for iteration
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        mask = adj_mask[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            neighbors[i].append(j)
            mask ^= lsb
            
    # Function to count C4s containing a specific edge (u, v)
    def count_c4s_for_edge(u, v):
        count = 0
        # w in N(u) \ {v}
        for w in neighbors[u]:
            if w == v:
                continue
            # x in N(v) \ {u} such that w ~ x
            # We need popcount(N[w] & N[v] & ~1<<u)
            # But we must ensure x != w. 
            # If x == w, then w is in N[w]? No.
            # So the intersection naturally excludes w if w is not in N[w].
            # However, if w is a common neighbor of u and v, w is in N[v].
            # Is w in N[w]? No.
            # So the count is correct.
            mask_x = adj_mask[w] & adj_mask[v] & ~(1 << u)
            count += bin(mask_x).count('1')
        return count

    # Compute C4 count for each edge
    edge_c4_counts = {}
    for e in edges:
        u, v = e
        edge_c4_counts[e] = count_c4s_for_edge(u, v)
        
    total_c4s = sum(edge_c4_counts.values()) // 4
    
    # Find the pair of edges to remove to maximize C4 reduction
    # Reduction = C4(e1) + C4(e2) - Intersection(e1, e2)
    
    best_reduction = -1
    best_pair = None
    
    # Optimization: iterate over pairs of edges
    # m = 93, pairs ~ 4300
    # For each pair, check intersection
    
    for i_idx in range(len(edges)):
        e1 = edges[i_idx]
        u1, v1 = e1
        c4_e1 = edge_c4_counts[e1]
        
        for j_idx in range(i_idx + 1, len(edges)):
            e2 = edges[j_idx]
            u2, v2 = e2
            c4_e2 = edge_c4_counts[e2]
            
            # Check if edges share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                # They share a vertex
                # Determine the shared vertex and the other endpoints
                if u1 == u2:
                    # e1=(u1, v1), e2=(u1, v2)
                    # Shared: u1. Other: v1, v2
                    shared = u1
                    other1 = v1
                    other2 = v2
                elif u1 == v2:
                    # e1=(u1, v1), e2=(u2, u1)
                    shared = u1
                    other1 = v1
                    other2 = u2
                elif v1 == u2:
                    # e1=(u1, v1), e2=(v1, v2)
                    shared = v1
                    other1 = u1
                    other2 = v2
                elif v1 == v2:
                    # e1=(u1, v1), e2=(u2, v1)
                    shared = v1
                    other1 = u1
                    other2 = u2
                
                # Intersection: C4s containing both edges
                # C4 must be other1 - shared - other2 - x - other1
                # x must be in N(other1) \ {shared} and N(other2) \ {shared}
                # And x != other1, x != other2
                mask_x = adj_mask[other1] & adj_mask[other2] & ~(1 << shared) & ~(1 << other1) & ~(1 << other2)
                intersection = bin(mask_x).count('1')
            else:
                # Disjoint edges
                # e1=(u1, v1), e2=(u2, v2)
                # Two possible C4s:
                # 1. u1-v1-u2-v2-u1 (requires v1~u2 and v2~u1)
                # 2. u1-v1-v2-u2-u1 (requires v1~v2 and u1~u2)
                
                inter = 0
                if (adj_mask[v1] & (1 << u2)) and (adj_mask[v2] & (1 << u1)):
                    inter += 1
                if (adj_mask[v1] & (1 << v2)) and (adj_mask[u1] & (1 << u2)):
                    inter += 1
                intersection = inter
            
            reduction = c4_e1 + c4_e2 - intersection
            if reduction > best_reduction:
                best_reduction = reduction
                best_pair = (e1, e2)
                
    # Remove the best pair
    e1, e2 = best_pair
    final_edges = [e for e in edges if e != e1 and e != e2]
    
    assert len(final_edges) == m, f"Expected {m} edges, got {len(final_edges)}"
    
    # Convert to list of lists
    result = [[u, v] for u, v in final_edges]
    return result
# EVOLVE-BLOCK-END
