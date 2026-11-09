"""Combined program: Complement Set Cover with optimized Polish and multi-pass local search."""
import time
import sys
from itertools import combinations, permutations

_START_TIME = None
_DEADLINE = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
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

    # Local edge index mapping for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    # Q3 abstract edges (8 vertices, 12 edges)
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Precompute all local Q3 edge sets (as frozensets of local edge indices)
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))

    # Convert to list of bitmasks and edge lists
    local_q3_bitmasks = []
    local_q3_edge_lists = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        edges = []
        for e in mask_set:
            mask_int |= (1 << e)
            edges.append(e)
        local_q3_bitmasks.append(mask_int)
        local_q3_edge_lists.append(edges)

    # Precompute global Q3 masks
    # For each subset of 8 vertices, map local edges to global edges
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute local edge (u,v) -> local_idx for K8
    local_edge_to_idx = {}
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = local_edge_index(u, v)

    # Build global masks
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
            
    num_q3s = len(global_q3_masks)
    
    if _elapsed() > _DEADLINE:
        # Fallback: return a sparse valid graph
        return [[i, i+1] for i in range(0, 16, 2)]

    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    # Also track active counts for greedy
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # Initialize counts: number of ACTIVE Q3s (hit_count == 0) containing edge e
    active_counts = [0] * num_edges_total
    hit_count = [0] * num_q3s # 0 means not hit (active), >0 means hit
    
    for e in range(num_edges_total):
        active_counts[e] = len(q3s_containing_edge[e])
        
    removed_edges = [False] * num_edges_total
    removed_list = []

    # Greedy Loop
    while True:
        if _elapsed() > _DEADLINE:
            break
            
        # Find edge with max active_count
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if active_counts[e] > max_count:
                max_count = active_counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        removed_list.append(best_edge)
        
        # Update active Q3s and counts
        # For each Q3 containing best_edge, if it was active (hit_count 0), mark it hit
        # And decrement active_counts for all other edges in that Q3
        for q3_idx in q3s_containing_edge[best_edge]:
            if hit_count[q3_idx] == 0:
                hit_count[q3_idx] = 1
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        active_counts[e_idx] -= 1
                    m ^= lsb

    # Polish Phase: Multi-pass local search to re-add removed edges
    # We perform multiple passes because re-adding one edge might make it safe to re-add another.
    # Condition to re-add e: No Q3 containing e has hit_count == 1
    # If hit_count == 1, it means that Q3 is hit ONLY by e (among removed edges).
    # Re-adding e would make hit_count 0, creating a valid Q3.
    
    max_passes = 5
    for pass_num in range(max_passes):
        if _elapsed() > _DEADLINE:
            break
            
        changed = False
        # Iterate over a copy of removed_list to allow re-adding during iteration
        # We iterate backwards to avoid index issues if we modify the list
        for i in range(len(removed_list) - 1, -1, -1):
            e = removed_list[i]
            
            # Check if safe to re-add
            safe = True
            for q3_idx in q3s_containing_edge[e]:
                if hit_count[q3_idx] == 1:
                    safe = False
                    break
            
            if safe:
                # Re-add e
                removed_edges[e] = False
                removed_list.pop(i)
                changed = True
                
                # Update counts: hit_count for Q3s containing e decreases by 1
                # active_counts for other edges in those Q3s increases by 1
                for q3_idx in q3s_containing_edge[e]:
                    if hit_count[q3_idx] > 0:
                        hit_count[q3_idx] -= 1
                        if hit_count[q3_idx] == 0:
                            # This Q3 became active again
                            mask = global_q3_masks[q3_idx]
                            m = mask
                            while m:
                                lsb = m & -m
                                e_idx = lsb.bit_length() - 1
                                if e_idx != e:
                                    active_counts[e_idx] += 1
                                m ^= lsb

        if not changed:
            break

    # Final Graph Construction
    edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            edges.append([u, v])
            
    return edges
