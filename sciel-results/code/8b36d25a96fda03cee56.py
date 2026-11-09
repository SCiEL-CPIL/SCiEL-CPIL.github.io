import time
import random
from itertools import combinations, permutations

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
    for p in permutations(range(8)):
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

    # Optimization: Precompute l2g for all subsets? No, just compute on the fly or precompute a list of l2g arrays.
    # 12870 subsets * 28 ints = ~360k ints. Small.
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
            
    best_removed_mask = (1 << num_edges_total) - 1
    best_removed_count = num_edges_total

    def run_search(seed):
        random.seed(seed)
        active = [True] * num_q3s
        counts = [0] * num_edges_total
        for e in range(num_edges_total):
            counts[e] = len(q3s_containing_edge[e])
            
        removed_mask = 0
        removed_bool = [False] * num_edges_total
        
        # Greedy Set Cover with Randomization
        while True:
            if time.time() > deadline:
                return removed_mask, removed_bool
            max_score = -1
            best_edge = -1
            for e in range(num_edges_total):
                if removed_bool[e]:
                    continue
                # Add random perturbation to break ties and explore different paths
                score = counts[e] + random.uniform(0, 0.1)
                if score > max_score:
                    max_score = score
                    best_edge = e
                    
            if best_edge == -1 or counts[best_edge] == 0:
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
        def is_valid_single_removal(e, current_removed_mask):
            not_e_mask = ~(1 << e)
            for q3_idx in q3s_containing_edge[e]:
                q3_mask = global_q3_masks[q3_idx]
                if (q3_mask & not_e_mask) & current_removed_mask == 0:
                    return False
            return True

        def is_valid_pair_removal(e1, e2, current_removed_mask):
            new_removed_mask = current_removed_mask & ~(1 << e1) & ~(1 << e2)
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
            rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
            random.shuffle(rem_edges)
            
            for e in rem_edges:
                if not removed_bool[e]: continue
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
                
            random.shuffle(rem_edges)
            
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
                        break
                if changed:
                    break
            
            if not changed:
                break

        # Pass 3: Swap moves
        while time.time() < deadline - 1:
            changed = False
            rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
            pres_edges = [e for e in range(num_edges_total) if not removed_bool[e]]
            
            if not rem_edges or not pres_edges:
                break
                
            random.shuffle(rem_edges)
            random.shuffle(pres_edges)
            
            swaps_tried = 0
            max_swaps_per_iter = 50 # Limit to keep it fast
            
            for e_add in rem_edges:
                if swaps_tried >= max_swaps_per_iter:
                    break
                mask_add = 1 << e_add
                
                for e_del in pres_edges:
                    if swaps_tried >= max_swaps_per_iter:
                        break
                    swaps_tried += 1
                    mask_del = 1 << e_del
                    valid = True
                    for q3_idx in q3s_containing_edge[e_add]:
                        q3_mask = global_q3_masks[q3_idx]
                        rest_mask = q3_mask & ~mask_add & ~mask_del
                        if rest_mask & removed_mask:
                            valid = False
                            break
                    
                    if valid:
                        removed_bool[e_add] = False
                        removed_bool[e_del] = True
                        removed_mask &= ~mask_add
                        removed_mask |= mask_del
                        changed = True
                        break
            
            if not changed:
                break

        return removed_mask, removed_bool

    # Run multiple searches
    search_count = 0
    while time.time() < deadline - 10:
        seed = random.randint(0, 10000)
        removed_mask, removed_bool = run_search(seed)
        current_removed_count = sum(removed_bool)
        if current_removed_count < best_removed_count:
            best_removed_count = current_removed_count
            best_removed_mask = removed_mask
            best_removed_bool = removed_bool.copy()
        search_count += 1
        # If we've done a few searches, we can reduce the time per search or just stop
        # The greedy phase is fast, local search takes most time.
        # Let's just run as many as we can.

    # Final verification and output
    result_edges = []
    for e in range(num_edges_total):
        if not best_removed_bool[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
