import time
import random
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119 using closed-form
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)

    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Local edge index mapping for K8 using closed-form
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        return u * 7 - u * (u - 1) // 2 + (v - u - 1)

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

    # Precompute l2g for all subsets
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

    # Generate global masks with time guard
    for l2g in subset_l2g_maps:
        if time.time() > deadline:
            return [[i, i+1] for i in range(0, 16, 2)]
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                global_mask |= (1 << l2g[le])
            global_q3_masks.append(global_mask)

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

    # Convert to numpy arrays for faster operations
    q3_edge_lists_np = [np.array(lst, dtype=np.int32) for lst in q3s_containing_edge]
    global_q3_masks_np = np.array(global_q3_masks, dtype=np.uint64) if num_q3s > 0 else np.array([], dtype=np.uint64)

    # Greedy Set Cover using numpy for speed
    active = np.ones(num_q3s, dtype=bool)
    counts = np.array([len(q3s_containing_edge[e]) for e in range(num_edges_total)], dtype=np.int32)
    
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
        
        # Update active and counts using numpy
        q3_indices = q3_edge_lists_np[best_edge]
        newly_inactive = active[q3_indices]
        active[q3_indices] = False
        
        # For each newly inactive Q3, decrement counts of its other edges
        if np.any(newly_inactive):
            for q3_idx in q3_indices[newly_inactive]:
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

    def is_valid_triple_removal(e1, e2, e3, current_removed_mask):
        new_removed_mask = current_removed_mask & ~(1 << e1) & ~(1 << e2) & ~(1 << e3)
        for q3_idx in q3s_containing_edge[e1]:
            if global_q3_masks[q3_idx] & new_removed_mask == 0:
                return False
        for q3_idx in q3s_containing_edge[e2]:
            if global_q3_masks[q3_idx] & new_removed_mask == 0:
                return False
        for q3_idx in q3s_containing_edge[e3]:
            if global_q3_masks[q3_idx] & new_removed_mask == 0:
                return False
        return True

    def is_valid_quad_removal(e1, e2, e3, e4, current_removed_mask):
        new_removed_mask = current_removed_mask & ~(1 << e1) & ~(1 << e2) & ~(1 << e3) & ~(1 << e4)
        for q3_idx in q3s_containing_edge[e1]:
            if global_q3_masks[q3_idx] & new_removed_mask == 0:
                return False
        for q3_idx in q3s_containing_edge[e2]:
            if global_q3_masks[q3_idx] & new_removed_mask == 0:
                return False
        for q3_idx in q3s_containing_edge[e3]:
            if global_q3_masks[q3_idx] & new_removed_mask == 0:
                return False
        for q3_idx in q3s_containing_edge[e4]:
            if global_q3_masks[q3_idx] & new_removed_mask == 0:
                return False
        return True

    # Pass 1: Single removals
    while time.time() < deadline - 15:
        changed = False
        rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
        random.shuffle(rem_edges)
        
        for e in rem_edges:
            if not removed_bool[e]:
                continue
            if is_valid_single_removal(e, removed_mask):
                removed_bool[e] = False
                removed_mask &= ~(1 << e)
                changed = True
        if not changed:
            break

    # Pass 2: Pair removals
    while time.time() < deadline - 10:
        changed = False
        rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
        
        if len(rem_edges) < 2:
            break
            
        random.shuffle(rem_edges)
        
        for i in range(len(rem_edges)):
            if time.time() > deadline - 10:
                break
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

    # Pass 3: Triple removals
    while time.time() < deadline - 5:
        changed = False
        rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
        
        if len(rem_edges) < 3:
            break
        
        random.shuffle(rem_edges)
        
        max_triples = 5000
        count = 0
        for i in range(len(rem_edges)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(rem_edges)):
                if time.time() > deadline - 5:
                    break
                for k in range(j + 1, len(rem_edges)):
                    if time.time() > deadline - 5:
                        break
                    e1, e2, e3 = rem_edges[i], rem_edges[j], rem_edges[k]
                    if not removed_bool[e1] or not removed_bool[e2] or not removed_bool[e3]:
                        continue
                    if is_valid_triple_removal(e1, e2, e3, removed_mask):
                        removed_bool[e1] = False
                        removed_bool[e2] = False
                        removed_bool[e3] = False
                        removed_mask &= ~(1 << e1)
                        removed_mask &= ~(1 << e2)
                        removed_mask &= ~(1 << e3)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
            count += 1
            if count >= max_triples:
                break
        
        if not changed:
            break

    # Pass 4: Quad removals (new)
    while time.time() < deadline - 2:
        changed = False
        rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
        
        if len(rem_edges) < 4:
            break
        
        random.shuffle(rem_edges)
        
        max_quads = 2000
        count = 0
        for i in range(len(rem_edges)):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, len(rem_edges)):
                if time.time() > deadline - 2:
                    break
                for k in range(j + 1, len(rem_edges)):
                    if time.time() > deadline - 2:
                        break
                    for l in range(k + 1, len(rem_edges)):
                        if time.time() > deadline - 2:
                            break
                        e1, e2, e3, e4 = rem_edges[i], rem_edges[j], rem_edges[k], rem_edges[l]
                        if not removed_bool[e1] or not removed_bool[e2] or not removed_bool[e3] or not removed_bool[e4]:
                            continue
                        if is_valid_quad_removal(e1, e2, e3, e4, removed_mask):
                            removed_bool[e1] = False
                            removed_bool[e2] = False
                            removed_bool[e3] = False
                            removed_bool[e4] = False
                            removed_mask &= ~(1 << e1)
                            removed_mask &= ~(1 << e2)
                            removed_mask &= ~(1 << e3)
                            removed_mask &= ~(1 << e4)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
            count += 1
            if count >= max_quads:
                break
        
        if not changed:
            break

    # Simulated Annealing Phase
    # Try random swaps: remove one edge, add another (or vice versa)
    def objective_value():
        return sum(1 for e in range(num_edges_total) if not removed_bool[e])
    
    current_obj = objective_value()
    best_obj = current_obj
    best_removed_mask = removed_mask
    best_removed_bool = removed_bool.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline - 1 and T > T_min:
        rem_edges = [e for e in range(num_edges_total) if removed_bool[e]]
        non_rem_edges = [e for e in range(num_edges_total) if not removed_bool[e]]
        
        if len(rem_edges) == 0 or len(non_rem_edges) == 0:
            break
        
        # Random swap: add one removed edge, remove one non-removed edge
        e_add = random.choice(rem_edges)
        e_remove = random.choice(non_rem_edges)
        
        # Tentatively apply
        old_removed_mask = removed_mask
        removed_bool[e_add] = False
        removed_bool[e_remove] = True
        removed_mask = (removed_mask & ~(1 << e_add)) | (1 << e_remove)
        
        # Check validity
        valid = True
        # Check if adding e_add creates a Q3
        for q3_idx in q3s_containing_edge[e_add]:
            if global_q3_masks[q3_idx] & removed_mask == 0:
                valid = False
                break
        
        if valid:
            new_obj = objective_value()
            delta_obj = new_obj - current_obj
            if delta_obj >= 0 or random.random() < np.exp(delta_obj / T):
                current_obj = new_obj
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_removed_mask = removed_mask
                    best_removed_bool = removed_bool.copy()
            else:
                # Revert
                removed_bool[e_add] = True
                removed_bool[e_remove] = False
                removed_mask = old_removed_mask
        else:
            # Revert
            removed_bool[e_add] = True
            removed_bool[e_remove] = False
            removed_mask = old_removed_mask
        
        T *= cooling
    
    # Restore best solution found
    if best_obj > current_obj:
        removed_mask = best_removed_mask
        removed_bool = best_removed_bool.copy()
    
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
