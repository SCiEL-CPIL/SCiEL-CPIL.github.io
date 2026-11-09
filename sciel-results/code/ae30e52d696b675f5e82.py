"""Combined: Complement Set Cover with numpy-accelerated greedy + multi-strategy polish."""
import time
import random
import sys
import numpy as np
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

    # Convert to list of edge lists
    local_q3_edge_lists = []
    for mask_set in local_q3_edge_sets:
        edges = sorted(mask_set)
        local_q3_edge_lists.append(edges)

    # Precompute global Q3 masks
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute local edge (u,v) -> local_idx for K8
    local_edge_to_idx = {}
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = local_edge_index(u, v)

    # Build global masks as numpy boolean arrays (num_q3s x 120)
    # First count total Q3s
    num_subsets = len(all_subsets)
    num_local_q3 = len(local_q3_edge_lists)
    total_q3_count = num_subsets * num_local_q3
    
    # Store as list of frozensets for now, convert to matrix later
    global_q3_edge_lists = []
    for subset in all_subsets:
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        for edge_list in local_q3_edge_lists:
            global_edges = [l2g[le] for le in edge_list]
            global_q3_edge_lists.append(global_edges)
            
    num_q3s = len(global_q3_edge_lists)
    
    if _elapsed() > _DEADLINE:
        return [[i, i+1] for i in range(0, 16, 2)]

    # Build incidence matrix: q3_edge_matrix[q3_idx, edge_idx] = 1 if edge in Q3
    q3_edge_matrix = np.zeros((num_q3s, num_edges_total), dtype=np.int8)
    for q3_idx, edge_list in enumerate(global_q3_edge_lists):
        for e in edge_list:
            q3_edge_matrix[q3_idx, e] = 1

    # q3s_containing_edge[e] = list of Q3 indices containing edge e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    for q3_idx in range(num_q3s):
        for e in range(num_edges_total):
            if q3_edge_matrix[q3_idx, e]:
                q3s_containing_edge[e].append(q3_idx)

    # Precompute total_q3s per edge
    total_q3s_per_edge = np.array([len(q3s_containing_edge[e]) for e in range(num_edges_total)], dtype=np.float64)
    
    # Convert q3s_containing_edge to numpy arrays for speed
    q3s_containing_edge_np = [np.array(q3s_containing_edge[e], dtype=np.int32) for e in range(num_edges_total)]

    def run_greedy(seed=0):
        """Run one greedy set cover pass. Returns removed_list."""
        rng = random.Random(seed)
        
        active_counts = total_q3s_per_edge.copy()  # numpy array
        hit_count = np.zeros(num_q3s, dtype=np.int32)
        removed = np.zeros(num_edges_total, dtype=bool)
        removed_list = []
        
        # Precompute: for each Q3, which edges it contains (as numpy array)
        q3_edges_arr = []
        for q3_idx in range(num_q3s):
            edges_in_q3 = np.where(q3_edge_matrix[q3_idx] == 1)[0]
            q3_edges_arr.append(edges_in_q3)
        
        # Greedy loop with numpy-accelerated scoring
        # Score = (ac * (ac + 1)) / (total + 0.5)  -- smooth surrogate
        while True:
            if _elapsed() > _DEADLINE:
                break
            
            # Compute scores for all non-removed edges with active_count > 0
            mask = (~removed) & (active_counts > 0)
            if not np.any(mask):
                break
            
            # Smooth score: ac*(ac+1) / (total + 0.5)
            scores = np.zeros(num_edges_total, dtype=np.float64)
            ac_masked = active_counts[mask]
            total_masked = total_q3s_per_edge[mask]
            scores[mask] = (ac_masked * (ac_masked + 1.0)) / (total_masked + 0.5)
            
            # Add small random tie-breaking
            scores[mask] += rng.random() * 0.001
            
            best_edge = int(np.argmax(scores))
            if removed[best_edge] or active_counts[best_edge] <= 0:
                break
            
            # Remove best_edge
            removed[best_edge] = True
            removed_list.append(best_edge)
            
            # Update: for each Q3 containing best_edge, if first hit, decrement other edges' active_counts
            for q3_idx in q3s_containing_edge_np[best_edge]:
                if hit_count[q3_idx] == 0:
                    hit_count[q3_idx] = 1
                    # Decrement active_counts for all other edges in this Q3
                    other_edges = q3_edges_arr[q3_idx]
                    other_edges = other_edges[other_edges != best_edge]
                    active_counts[other_edges] -= 1
        
        # Polish: multiple passes of re-adding edges
        for pass_num in range(8):
            if _elapsed() > _DEADLINE:
                break
            
            changed = False
            # Alternate between reverse and forward order
            if pass_num % 2 == 0:
                order = range(len(removed_list) - 1, -1, -1)
            else:
                order = range(len(removed_list))
            
            for i in list(order):
                if i >= len(removed_list):
                    continue
                e = removed_list[i]
                
                # Check if safe to re-add: no Q3 containing e has hit_count == 1
                q3_indices = q3s_containing_edge_np[e]
                if len(q3_indices) == 0:
                    # No Q3 contains this edge, safe to re-add
                    removed[e] = False
                    removed_list.pop(i)
                    changed = True
                    continue
                    
                hc = hit_count[q3_indices]
                if np.any(hc == 1):
                    continue
                
                # Safe to re-add
                removed[e] = False
                removed_list.pop(i)
                changed = True
                
                # Update: for each Q3 containing e, decrement hit_count
                for q3_idx in q3_indices:
                    if hit_count[q3_idx] > 0:
                        hit_count[q3_idx] -= 1
                        if hit_count[q3_idx] == 0:
                            # Q3 became active again
                            other_edges = q3_edges_arr[q3_idx]
                            other_edges = other_edges[other_edges != e]
                            active_counts[other_edges] += 1
        
        # Swap-based polish: try to swap removed edges with active edges
        if _elapsed() < _DEADLINE - 10.0:
            for i in range(len(removed_list) - 1, -1, -1):
                if i >= len(removed_list):
                    continue
                e = removed_list[i]
                
                # Find blocking Q3s
                q3_indices = q3s_containing_edge_np[e]
                if len(q3_indices) == 0:
                    removed[e] = False
                    removed_list.pop(i)
                    continue
                    
                hc = hit_count[q3_indices]
                blocking = q3_indices[hc == 1]
                
                if len(blocking) == 0:
                    removed[e] = False
                    removed_list.pop(i)
                    continue
                
                # For each blocking Q3, try to find an active edge in it to remove instead
                # Net effect: swap e back, remove another edge
                swapped = False
                for q3_idx in blocking:
                    other_edges = q3_edges_arr[q3_idx]
                    other_edges = other_edges[other_edges != e]
                    # Find active edges in this Q3
                    active_in_q3 = other_edges[~removed[other_edges]]
                    if len(active_in_q3) == 0:
                        continue
                    
                    # Pick the active edge with highest active_count (most impactful removal)
                    best_swap = active_in_q3[np.argmax(active_counts[active_in_q3])]
                    
                    # Perform swap: remove best_swap, re-add e
                    # First, handle removal of best_swap
                    for q3_idx2 in q3s_containing_edge_np[best_swap]:
                        if hit_count[q3_idx2] == 0:
                            hit_count[q3_idx2] = 1
                            other_edges2 = q3_edges_arr[q3_idx2]
                            other_edges2 = other_edges2[other_edges2 != best_swap]
                            active_counts[other_edges2] -= 1
                    removed[best_swap] = True
                    removed_list.append(best_swap)
                    
                    # Now re-add e
                    for q3_idx3 in q3s_containing_edge_np[e]:
                        if hit_count[q3_idx3] > 0:
                            hit_count[q3_idx3] -= 1
                            if hit_count[q3_idx3] == 0:
                                other_edges3 = q3_edges_arr[q3_idx3]
                                other_edges3 = other_edges3[other_edges3 != e]
                                active_counts[other_edges3] += 1
                    removed[e] = False
                    removed_list.pop(i)
                    swapped = True
                    break
                
                if swapped:
                    continue
        
        return removed_list

    # Run multiple greedy passes with different seeds, keep best
    best_removed = None
    best_num_removed = num_edges_total + 1
    
    max_restarts = 20
    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - 5.0:
            break
        
        removed_list = run_greedy(seed=restart * 7 + 13)
        num_removed = len(removed_list)
        
        if num_removed < best_num_removed:
            best_num_removed = num_removed
            best_removed = removed_list[:]
            
            # If we found a very good solution, do extra polish
            if num_removed <= 35 and _elapsed() < _DEADLINE - 20.0:
                # Extra polish passes on the best solution
                active_counts = total_q3s_per_edge.copy()
                hit_count = np.zeros(num_q3s, dtype=np.int32)
                removed = np.zeros(num_edges_total, dtype=bool)
                for e in best_removed:
                    removed[e] = True
                    for q3_idx in q3s_containing_edge_np[e]:
                        if hit_count[q3_idx] == 0:
                            hit_count[q3_idx] = 1
                            other_edges = q3_edges_arr[q3_idx]
                            other_edges = other_edges[other_edges != e]
                            active_counts[other_edges] -= 1
                
                # Multiple polish passes
                for pp in range(15):
                    if _elapsed() > _DEADLINE - 5.0:
                        break
                    changed = False
                    for i in range(len(best_removed) - 1, -1, -1):
                        e = best_removed[i]
                        q3_indices = q3s_containing_edge_np[e]
                        if len(q3_indices) == 0:
                            removed[e] = False
                            best_removed.pop(i)
                            changed = True
                            continue
                        hc = hit_count[q3_indices]
                        if np.any(hc == 1):
                            continue
                        removed[e] = False
                        best_removed.pop(i)
                        changed = True
                        for q3_idx in q3_indices:
                            if hit_count[q3_idx] > 0:
                                hit_count[q3_idx] -= 1
                                if hit_count[q3_idx] == 0:
                                    other_edges = q3_edges_arr[q3_idx]
                                    other_edges = other_edges[other_edges != e]
                                    active_counts[other_edges] += 1
                    if not changed:
                        break
        
        if best_num_removed <= 30:
            break

    if best_removed is None:
        # Fallback: run one more greedy
        best_removed = run_greedy(seed=999)

    # Final verification and edge construction
    edges = []
    for e in range(num_edges_total):
        if e not in set(best_removed):
            u, v = inv_edge_map[e]
            edges.append([u, v])
            
    return edges
