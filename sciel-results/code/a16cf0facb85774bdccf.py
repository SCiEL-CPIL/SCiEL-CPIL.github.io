import time
import random
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

    # Q3 abstract edges
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
    for p in __import__('itertools').permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))

    # Convert to list of bitmasks
    local_q3_bitmasks = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)

    # Precompute global Q3 masks
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    local_edge_to_idx = {}
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = local_edge_index(u, v)

    local_q3_edge_lists = []
    for mask in local_q3_bitmasks:
        edges = []
        for i in range(28):
            if mask & (1 << i):
                edges.append(i)
        local_q3_edge_lists.append(edges)

    # Optimization: Precompute l2g for all subsets
    subset_l2g_maps = []
    for subset in all_subsets:
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
        subset_l2g_maps.append(l2g)

    # Generate global masks
    for l2g in subset_l2g_maps:
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                global_mask |= (1 << l2g[le])
            global_q3_masks.append(global_mask)
            
    if time.time() > deadline:
        return [[i, i+1] for i in range(0, 16, 2)]

    num_q3s = len(global_q3_masks)
    
    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # Greedy Set Cover
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    removed_mask = 0
    removed_bool = [False] * num_edges_total
    
    while True:
        if time.time() > deadline:
            break
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_bool[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        removed_bool[best_edge] = True
        removed_mask |= (1 << best_edge)
        
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # Local Search Polish
    # Goal: Minimize removed_mask (maximize graph edges)
    # Move 1: Remove e from removed_mask if valid.
    
    def is_valid_single_removal(e, current_removed_mask):
        # Check if adding e back (removing from removed set) creates a Q3
        # A Q3 is created if there exists a Q3 containing e such that 
        # all other edges of that Q3 are NOT in current_removed_mask
        # i.e., (q3_mask & ~(1 << e)) & current_removed_mask == 0
        not_e_mask = ~(1 << e)
        for q3_idx in q3s_containing_edge[e]:
            q3_mask = global_q3_masks[q3_idx]
            if (q3_mask & not_e_mask) & current_removed_mask == 0:
                return False
        return True

    def is_valid_pair_removal(e1, e2, current_removed_mask):
        # Check if adding e1 and e2 back creates a Q3
        # New removed mask = current_removed_mask & ~(1<<e1) & ~(1<<e2)
        new_removed_mask = current_removed_mask & ~(1 << e1) & ~(1 << e2)
        
        # Only Q3s containing e1 or e2 can be affected
        # We need to check all Q3s in union of q3s_containing_edge[e1] and q3s_containing_edge[e2]
        
        for q3_idx in q3s_containing_edge[e1]:
            q3_mask = global_q3_masks[q3_idx]
            if q3_mask & new_removed_mask == 0:
                return False
        for q3_idx in q3s_containing_edge[e2]:
            q3_mask = global_q3_masks[q3_idx]
            if q3_mask & new_removed_mask == 0:
                return False
        return True

    # Pass 1: Single removals
    while time.time() < deadline - 5:
        changed = False
        # Get list of removed edges
        rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
        random.shuffle(rem_edges)
        
        for e in rem_edges:
            if not removed_bool[e]: continue # might have been removed by previous op? No, single loop.
            if is_valid_single_removal(e, removed_mask):
                removed_bool[e] = False
                removed_mask &= ~(1 << e)
                changed = True
        if not changed:
            break

    # Pass 2: Pair removals
    while time.time() < deadline - 2:
        changed = False
        rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
        
        if len(rem_edges) < 2:
            break
            
        # Shuffle to randomize order
        random.shuffle(rem_edges)
        
        # Try all pairs? Or just random pairs?
        # |R| is small (20-40). Pairs ~ 800.
        # 800 * (2 * 1000 checks) = 1.6M ops. Fast.
        
        for i in range(len(rem_edges)):
            for j in range(i + 1, len(rem_edges)):
                e1 = rem_edges[i]
                e2 = rem_edges[j]
                if not removed_bool[e1] or not removed_bool[e2]:
                    continue
                if is_valid_pair_removal(e1, e2, removed_mask):
                    removed_bool[e1] = False
                    removed_bool[e2] = False
                    removed_mask &= ~(1 << e1)
                    removed_mask &= ~(1 << e2)
                    changed = True
                    # Restart pair search? Or continue?
                    # If we continue, rem_edges list is stale.
                    # Let's break and restart the outer while loop.
                    break
            if changed:
                break
        
        if not changed:
            break

    # Final verification and output
    result_edges = []
    for e in range(num_edges_total):
        if not removed_bool[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
