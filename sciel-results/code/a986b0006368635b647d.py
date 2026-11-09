import time
import random
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
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

    # Q3 abstract edges (standard labeling of 3-cube)
    # Vertices 0..7. Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Local edge index mapping for K8 (28 edges)
    local_edge_to_idx = {}
    local_idx_to_edge = [None] * 28
    idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = idx
            local_idx_to_edge[idx] = (u, v)
            idx += 1

    # Precompute local Q3 edge sets as bitmasks of 28 bits
    # We iterate over all permutations of 8 vertices to cover all isomorphisms
    # However, many permutations yield the same edge set. We use a set of frozensets.
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_to_idx[(min(u_val, v_val), max(u_val, v_val))]
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))
    
    # Convert to list of bitmasks for faster operations
    local_q3_bitmasks = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)

    # Precompute global Q3 masks
    # For each subset of 8 vertices, map local edges to global edges
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    # Optimization: Precompute local edge -> global edge mapping for each subset
    # This is the bottleneck. 12870 subsets * 28 edges = ~360k lookups.
    # Then for each subset, iterate over local_q3_bitmasks (~140 unique Q3s per subset? 
    # Actually, the number of unique Q3s in K8 is small. Let's check the size of local_q3_bitmasks).
    
    if time.time() > deadline - 10:
        # If precomputation is too slow, fallback to a simple heuristic
        # Return a sparse graph to be safe, though this will score low.
        # But we should try to finish precomputation.
        pass

    num_local_q3s = len(local_q3_bitmasks)
    
    # To speed up global mask generation, we can precompute for each local Q3 bitmask,
    # the list of local edge indices it contains.
    local_q3_edge_lists = []
    for mask in local_q3_bitmasks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            edges.append(lsb.bit_length() - 1)
            m ^= lsb
        local_q3_edge_lists.append(edges)

    for subset in all_subsets:
        # Map local edge idx to global edge idx
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        # For each local Q3, generate the global mask
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                ge = l2g[le]
                global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    if time.time() > deadline - 5:
        # Fallback if precomputation took too long
        # Return a simple valid graph (e.g., bipartite K8,8 is Q3-free? 
        # Q3 is bipartite. K8,8 contains Q3? 
        # Q3 has 8 vertices. K8,8 has 16. 
        # Does K8,8 contain Q3? Yes, likely. 
        # A safe fallback is a matching or a star.
        # But we want to maximize edges. 
        # Let's return the edges of a random sparse graph.
        return [[i, i+1] for i in range(0, 16, 2)]

    num_q3s = len(global_q3_masks)
    
    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    # removed_edges[e] = True if edge e is removed (not in graph)
    removed_edges = [False] * num_edges_total
    
    # q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # counts[e] = number of active Q3s containing e
    counts = [0] * num_edges_total
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    # active_q3s: boolean list, True if Q3 is not yet hit
    active_q3s = [True] * num_q3s
    
    # Greedy Loop
    while True:
        if time.time() > deadline - 10:
            break
            
        # Find edge with max count among not-removed edges
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        
        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active_q3s[q3_idx]:
                active_q3s[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # Local Search Polish: try to re-add removed edges
    # We iterate through removed edges. If adding edge e back does not create a Q3, we add it.
    # Adding edge e back means setting removed_edges[e] = False.
    # This creates a Q3 if there exists a Q3 mask M such that:
    # 1. e in M
    # 2. All other edges in M are NOT removed (i.e., M \ {e} is a subset of current graph edges)
    # 3. No other edge in M was removed previously? 
    # Wait, the condition for a Q3 to be "present" in the graph G = K \ R is:
    # For all edges f in M, f is NOT in R.
    # So if we try to remove e from R (add to G), the new R' = R \ {e}.
    # A Q3 M is present in G' if M \cap R' = empty.
    # Since R' = R \ {e}, M \cap R' = (M \cap R) \ {e}.
    # So M \cap R' is empty iff (M \cap R) is subset of {e}.
    # Since e is in R, M \cap R contains e.
    # So we need M \cap R == {e}.
    # This means e is the ONLY removed edge in M.
    
    # We can maintain a count for each Q3: how many removed edges does it have?
    # Let's call it removed_count[q3_idx].
    # Initially, after greedy, we can compute this.
    
    removed_count = [0] * num_q3s
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            if removed_edges[e_idx]:
                removed_count[i] += 1
            m ^= lsb
            
    # A Q3 is "dangerous" for re-adding e if removed_count[q3_idx] == 1 and e is in M.
    # Actually, if removed_count[q3_idx] == 1, let e* be the unique removed edge.
    # If we try to re-add e*, then removed_count becomes 0, and the Q3 is formed.
    # So we can re-add e if for all Q3s M containing e, removed_count[M] > 1.
    # Or, if we remove e from R, the new count for M is removed_count[M] - 1.
    # We need new_count[M] > 0 for all M containing e.
    # So we can re-add e if min(removed_count[M] for M in q3s_containing_edge[e]) > 1.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in range(num_edges_total):
            if not removed_edges[e]:
                continue
            
            # Check if we can re-add e
            min_count = 999
            for q3_idx in q3s_containing_edge[e]:
                if removed_count[q3_idx] < min_count:
                    min_count = removed_count[q3_idx]
                if min_count <= 1:
                    break
                    
            if min_count > 1:
                # Safe to re-add
                removed_edges[e] = False
                improved = True
                # Update removed_count for all Q3s containing e
                for q3_idx in q3s_containing_edge[e]:
                    removed_count[q3_idx] -= 1

    # Construct final edge list
    final_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            final_edges.append([u, v])
            
    return final_edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Number of edges: {len(edges)}")
