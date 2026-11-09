import time
import random
import sys
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
    
    # Precompute local edge (u,v) -> local_idx for K8
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

    # Now, for each subset:
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
            global_mask = 0
            for le in edge_list:
                ge = l2g[le]
                global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    num_q3s = len(global_q3_masks)
    if time.time() > deadline:
        return [[i, i+1] for i in range(0, 16, 2)]

    # Precompute edge lists for each Q3 for fast intersection/checks
    # q3_edges[q3_idx] = list of 12 edge indices
    q3_edges = []
    for mask in global_q3_masks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            edges.append(e_idx)
            m ^= lsb
        q3_edges.append(edges)

    # q3s_containing_edge[e] = list of Q3 indices containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    for i, edges in enumerate(q3_edges):
        for e in edges:
            q3s_containing_edge[e].append(i)

    def greedy_set_cover():
        removed_edges = [False] * num_edges_total
        active = [True] * num_q3s
        counts = [0] * num_edges_total
        
        for e in range(num_edges_total):
            counts[e] = len(q3s_containing_edge[e])
            
        while True:
            if time.time() > deadline:
                break
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
                
            removed_edges[best_edge] = True
            for q3_idx in q3s_containing_edge[best_edge]:
                if active[q3_idx]:
                    active[q3_idx] = False
                    edges = q3_edges[q3_idx]
                    for e in edges:
                        if e != best_edge:
                            counts[e] -= 1
                            
        return removed_edges

    def local_search(removed_edges):
        removed_set = set(i for i, r in enumerate(removed_edges) if r)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to remove non-critical edges
            for e in list(removed_set):
                # Check if e is critical: i.e., is there a Q3 hit ONLY by e?
                critical = False
                for q3_idx in q3s_containing_edge[e]:
                    # Check if Q3 has any other edge in removed_set
                    other_found = False
                    for ee in q3_edges[q3_idx]:
                        if ee != e and ee in removed_set:
                            other_found = True
                            break
                    if not other_found:
                        critical = True
                        break
                if not critical:
                    removed_set.discard(e)
                    removed_edges[e] = False
                    improved = True
            
            # 2. If no removals, try swaps to escape local minima
            if not improved and time.time() < deadline - 5:
                # Pick a random edge in removed_set
                if removed_set:
                    e = random.choice(list(removed_set))
                    # Compute C_e: Q3s hit only by e
                    C_e = []
                    for q3_idx in q3s_containing_edge[e]:
                        other_found = False
                        for ee in q3_edges[q3_idx]:
                            if ee != e and ee in removed_set:
                                other_found = True
                                break
                        if not other_found:
                            C_e.append(q3_idx)
                    
                    if C_e:
                        # Find intersection of edges in all Q3s in C_e
                        # Start with edges of first Q3
                        intersection = set(q3_edges[C_e[0]])
                        for q3_idx in C_e[1:]:
                            intersection &= set(q3_edges[q3_idx])
                            if not intersection:
                                break
                        
                        if intersection:
                            # Find an edge f in intersection that is NOT in removed_set
                            candidates = [f for f in intersection if f not in removed_set]
                            if candidates:
                                f = random.choice(candidates)
                                # Perform swap
                                removed_set.discard(e)
                                removed_set.add(f)
                                removed_edges[e] = False
                                removed_edges[f] = True
                                improved = True # Trigger another round of removals

        return removed_edges

    best_removed = None
    best_count = num_edges_total + 1
    
    # Run multiple random restarts? 
    # The greedy is deterministic based on tie-breaking. 
    # We can randomize tie-breaking.
    
    # For now, run once with standard greedy, then local search.
    # If time permits, try randomizing the order of edges in greedy.
    
    removed = greedy_set_cover()
    removed = local_search(removed)
    
    count = sum(1 for r in removed if r)
    if count < best_count:
        best_count = count
        best_removed = removed[:]
        
    # Try a few randomized greedy runs if time allows
    attempts = 0
    while time.time() < deadline - 20 and attempts < 5:
        attempts += 1
        # Randomize tie-breaking in greedy
        random.seed(attempts)
        removed_r = [False] * num_edges_total
        active = [True] * num_q3s
        counts = [0] * num_edges_total
        for e in range(num_edges_total):
            counts[e] = len(q3s_containing_edge[e])
            
        while True:
            if time.time() > deadline - 20:
                break
            # Find all edges with max count, pick one randomly
            max_count = -1
            candidates = []
            for e in range(num_edges_total):
                if removed_r[e]:
                    continue
                if counts[e] > max_count:
                    max_count = counts[e]
                    candidates = [e]
                elif counts[e] == max_count:
                    candidates.append(e)
                    
            if not candidates or max_count == 0:
                break
                
            best_edge = random.choice(candidates)
            removed_r[best_edge] = True
            for q3_idx in q3s_containing_edge[best_edge]:
                if active[q3_idx]:
                    active[q3_idx] = False
                    edges = q3_edges[q3_idx]
                    for e in edges:
                        if e != best_edge:
                            counts[e] -= 1
                            
        removed_r = local_search(removed_r)
        count_r = sum(1 for r in removed_r if r)
        if count_r < best_count:
            best_count = count_r
            best_removed = removed_r[:]

    result_edges = []
    for e in range(num_edges_total):
        if not best_removed[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
