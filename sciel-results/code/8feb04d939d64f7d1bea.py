"""Combined program: greedy edge addition with bitmask Q3 detection + local search polish."""
import time
import itertools
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to index 0..119
    edge_idx_map = {}
    inv_edge_map = [None] * num_edges_total
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)
            idx += 1

    # Q3 abstract edges (3-cube: vertices 0..7, edges where Hamming distance = 1)
    q3_abstract_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_abstract_edges.append((i, j))

    # Precompute all global Q3 masks
    # For each 8-vertex subset and each permutation of Q3 vertices onto it,
    # generate the bitmask of 12 edges.
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))

    # Local edge index for K8 (28 edges)
    local_edge_to_idx = {}
    le_idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = le_idx
            le_idx += 1

    # Precompute local Q3 edge index sets (for each permutation of Q3 onto 8 positions)
    # There are 8! = 40320 permutations, but many give the same edge set.
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edge_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            if u_val > v_val:
                u_val, v_val = v_val, u_val
            edge_set.add(local_edge_to_idx[(u_val, v_val)])
        if len(edge_set) == 12:
            local_q3_edge_sets.add(frozenset(edge_set))

    # Convert local Q3 edge sets to lists of local edge indices
    local_q3_edge_lists = []
    for es in local_q3_edge_sets:
        local_q3_edge_lists.append(list(es))

    # For each subset, map local edge idx -> global edge idx
    # Then generate global Q3 masks
    for subset in all_subsets:
        # Build local-to-global edge index mapping
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                if u_glob > v_glob:
                    u_glob, v_glob = v_glob, u_glob
                l2g[le] = edge_idx_map[(u_glob, v_glob)]

        # Generate global masks for this subset
        for edge_list in local_q3_edge_lists:
            mask = 0
            for le in edge_list:
                ge = l2g[le]
                mask |= (1 << ge)
            global_q3_masks.append(mask)

    num_q3s = len(global_q3_masks)

    if time.time() > deadline - 5:
        # Fallback: return a sparse valid graph
        return [[i, i + 1] for i in range(0, 16, 2)]

    # Build incidence: for each edge, list of Q3 indices containing it
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb

    # Initial counts: how many Q3s each edge is in
    counts = [len(q3s_containing_edge[e]) for e in range(num_edges_total)]

    # Greedy construction: start with empty graph, add edges that don't create Q3
    # An edge can be added if no Q3 mask is fully contained in (current_edges | {new_edge})
    # i.e., for every Q3 that contains the new edge, at least one other edge in that Q3 is missing

    current_mask = 0  # bitmask of current edges
    current_edges = []

    # Sort edges by count (fewer Q3 conflicts first) for better greedy performance
    edge_order = sorted(range(num_edges_total), key=lambda e: counts[e])

    # Greedy add
    for e in edge_order:
        if time.time() > deadline - 10:
            break

        # Check if adding edge e creates a Q3
        # A Q3 is created if there exists a Q3 mask M such that M & ~(current_mask | (1<<e)) == 0
        # i.e., M is subset of current_mask | (1<<e)
        # Since e is in M (we only check Q3s containing e), this means M \ {e} ⊆ current_mask
        can_add = True
        for q3_idx in q3s_containing_edge[e]:
            m = global_q3_masks[q3_idx]
            # Check if all edges in M except e are in current_mask
            if (m & ~(current_mask | (1 << e))) != 0:
                # Some edge in M (other than e) is not in current_mask
                # This Q3 is NOT fully contained, so no conflict from this Q3
                continue
            else:
                # All edges in M except e are in current_mask, so adding e completes this Q3
                can_add = False
                break

        if can_add:
            current_mask |= (1 << e)
            current_edges.append(e)

    # Local search polish: try edge swaps
    # For each edge a NOT in current graph, and edge b IN current graph,
    # try adding a and removing b. Accept if still Q3-free and (optionally) improves some metric.
    # Since we want to maximize edges, we keep the same number of edges but might enable future additions.
    # Actually, a swap keeps edge count same. To increase edge count, we need to add an edge without removing.
    # But greedy already tried all edges. So swaps are for rearrangement to enable more additions later.

    # Better approach: after greedy, try to add more edges by temporarily removing some edges.
    # Or: do a second pass with different edge ordering.

    # Let's do a local search: try all single-edge additions again (maybe order matters)
    # Then try swaps to improve structure.

    # Second greedy pass with different ordering
    edge_order2 = sorted(range(num_edges_total), key=lambda e: -counts[e])
    for e in edge_order2:
        if time.time() > deadline - 5:
            break
        if current_mask & (1 << e):
            continue
        can_add = True
        for q3_idx in q3s_containing_edge[e]:
            m = global_q3_masks[q3_idx]
            if (m & ~(current_mask | (1 << e))) != 0:
                continue
            else:
                can_add = False
                break
        if can_add:
            current_mask |= (1 << e)
            current_edges.append(e)

    # Local search: try swaps to enable more additions
    # For each pair (a not in graph, b in graph), try swap
    in_graph = [False] * num_edges_total
    for e in current_edges:
        in_graph[e] = True

    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for a in range(num_edges_total):
            if time.time() > deadline - 3:
                break
            if in_graph[a]:
                continue
            for b in range(num_edges_total):
                if time.time() > deadline - 3:
                    break
                if not in_graph[b]:
                    continue

                # Try swap: remove b, add a
                new_mask = (current_mask & ~(1 << b)) | (1 << a)

                # Check if new_mask is Q3-free
                # A Q3 exists if any global_q3_mask is subset of new_mask
                q3_free = True
                for m in global_q3_masks:
                    if (m & ~new_mask) == 0:
                        q3_free = False
                        break

                if q3_free:
                    # Accept swap
                    current_mask = new_mask
                    in_graph[b] = False
                    in_graph[a] = True
                    current_edges.remove(b)
                    current_edges.append(a)
                    improved = True
                    break  # Restart after one swap

    # Convert edge indices back to [u, v] format
    result = [list(inv_edge_map[e]) for e in current_edges]
    return result
