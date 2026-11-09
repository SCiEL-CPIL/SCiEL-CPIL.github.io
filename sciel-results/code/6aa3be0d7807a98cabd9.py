import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n = 16
    num_edges_total = n * (n - 1) // 2  # 120
    
    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        # u < v
        # Number of edges with first vertex < u: u * (u - 1) / 2 ? No.
        # Standard mapping:
        # (0,1), (0,2), ..., (0,15) -> 0..14
        # (1,2), (1,3), ..., (1,15) -> 15..27
        # ...
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)
            
    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Precompute local Q3 masks on 8 vertices (0..7)
    # Vertices 0..7
    # Q3 edges in abstract space (using standard labeling 000..111)
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    q3_abstract_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (3,7),
        (4,5), (4,6),
        (5,7),
        (6,7)
    ]
    
    # We need to find all unique sets of 12 edges in K8 that form Q3.
    # We can iterate over all permutations of 0..7 and generate the edge set.
    # To avoid duplicates, we store them in a set of frozensets of local edge indices.
    
    # Local edge index mapping for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    local_q3_masks = set()
    
    # To speed up, we can just check all 8! permutations?
    # 40320 is small.
    from itertools import permutations
    for p in permutations(range(8)):
        # p[i] is the vertex in the subset corresponding to abstract node i
        # We need to check if the edges of Q3 mapped by p form a valid Q3?
        # Actually, any permutation of the vertices of Q3 maps Q3 to a subgraph of K8.
        # The set of edges is determined by the image of the abstract edges.
        # Since Q3 is connected and rigid, different permutations that differ by automorphisms
        # will produce the same edge set.
        
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        
        # There should be exactly 12 edges
        if len(edges_set) == 12:
            local_q3_masks.add(frozenset(edges_set))
            
    # Convert to list of bitmasks (integers) for fast operations
    # 28 bits fit in an int
    local_q3_bitmasks = []
    for mask_set in local_q3_masks:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)
        
    # Generate global Q3 masks
    # For each subset of 8 vertices from 0..15
    # Map local 0..7 to global vertices
    # Translate local edge bitmask to global edge bitmask
    
    global_q3_masks = []
    
    # Precompute mapping from local edge idx to global edge idx for a given subset
    # This is expensive to do inside the loop.
    
    # Optimization:
    # For each subset of 8 vertices, create a mapping array local_vertex_to_global_vertex.
    # Then for each local edge (u, v), get global edge index.
    
    all_subsets = list(combinations(range(n), 8))
    
    # To save time, we can sample subsets if the full generation is too slow?
    # But we need ALL Q3s to ensure correctness of the "no Q3" constraint.
    # If we miss a Q3, the result is invalid.
    # So we MUST generate all.
    
    # 12870 subsets * 840 masks = 10.8M iterations.
    # Inside: 12 edge lookups + bit setting.
    # This might take ~10-20s in Python.
    
    for subset in all_subsets:
        # subset is a tuple of 8 global vertex IDs
        # Map local index 0..7 to global vertex
        # local i -> subset[i]
        
        # Precompute local edge to global edge index map for this subset
        # local_edge_idx -> global_edge_idx
        l2g_edge = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g_edge[le] = ge
                
        for local_mask in local_q3_bitmasks:
            global_mask = 0
            # Iterate through bits in local_mask
            # 28 bits, so we can just loop 0..27
            for le in range(28):
                if local_mask & (1 << le):
                    ge = l2g_edge[le]
                    global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    num_q3s = len(global_q3_masks)
    
    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    # active_q3[i] = True if Q3 i is not yet hit.
    # count[e] = number of active Q3s containing edge e.
    
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    
    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        # Iterate through set bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # Initialize counts
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    removed_edges = set()
    current_edge_mask = (1 << num_edges_total) - 1  # All edges present initially
    
    # Greedy Loop
    # We need to efficiently find the edge with max count.
    # Since num_edges_total is 120, we can just scan.
    
    while True:
        if time.time() > deadline:
            break
            
        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if e in removed_edges:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges.add(best_edge)
        current_edge_mask &= ~(1 << best_edge)
        
        # Update active Q3s and counts
        # For each Q3 containing best_edge
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                # This Q3 is now hit.
                # We need to decrement counts for all other edges in this Q3.
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # The remaining edges are the solution
    # Construct edge list
    result_edges = []
    for e in range(num_edges_total):
        if e not in removed_edges:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
