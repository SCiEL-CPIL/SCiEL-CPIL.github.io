import time
import random
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Edge index mapping
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)

    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Q3 abstract edges (standard labeling 000..111)
    q3_abstract_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (3,7),
        (4,5), (4,6),
        (5,7),
        (6,7)
    ]

    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    # Generate all distinct Q3 edge-sets on 8 vertices
    local_q3_masks = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_masks.add(frozenset(edges_set))

    local_q3_bitmasks = []
    for mask_set in local_q3_masks:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)

    # Generate global Q3 masks for each 8-vertex subset
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))

    for subset in all_subsets:
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
            for le in range(28):
                if local_mask & (1 << le):
                    ge = l2g_edge[le]
                    global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)

    num_q3s = len(global_q3_masks)

    # For each edge, which Q3s does it belong to?
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb

    # Phase 1: Greedy set cover to get a valid starting graph
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])

    removed_edges = set()
    current_edge_mask = (1 << num_edges_total) - 1

    while True:
        if time.time() > deadline:
            break
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
        removed_edges.add(best_edge)
        current_edge_mask &= ~(1 << best_edge)
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

    # Phase 2: Simulated Annealing with incremental violation tracking
    # State: edge_mask, viol_count[e] = number of Q3s containing edge e where all other 11 edges are present
    # A Q3 is violated iff all 12 of its edges are present.
    # We track per-Q3 violation status for fast energy computation.

    # Initialize violation status from greedy result
    q3_violated = [False] * num_q3s
    total_violations = 0
    for i, mask in enumerate(global_q3_masks):
        if (current_edge_mask & mask) == mask:
            q3_violated[i] = True
            total_violations += 1

    curr_mask = current_edge_mask
    curr_edges = bin(curr_mask).count('1')
    best_mask = curr_mask
    best_edges = curr_edges
    best_viol = total_violations

    # SA with incremental updates
    T = 2.0
    cooling_rate = 0.9995
    min_T = 0.05
    all_edge_indices = list(range(num_edges_total))

    while T > min_T:
        if time.time() > deadline:
            break

        e = random.choice(all_edge_indices)
        bit = 1 << e
        new_mask = curr_mask ^ bit

        if not (curr_mask & bit):
            # Adding edge e: check which Q3s containing e become violated
            # A Q3 becomes violated iff all its other 11 edges are already present
            new_viol_delta = 0
            for q3_idx in q3s_containing_edge[e]:
                if not q3_violated[q3_idx]:
                    # Check if all other edges in this Q3 are present
                    q3_mask = global_q3_masks[q3_idx]
                    other_edges = q3_mask & ~bit
                    if (curr_mask & other_edges) == other_edges:
                        new_viol_delta += 1
            new_viol = total_violations + new_viol_delta
            new_edge_count = curr_edges + 1
        else:
            # Removing edge e: check which Q3s containing e stop being violated
            # A Q3 stops being violated iff it was violated and e was one of its edges
            new_viol_delta = 0
            for q3_idx in q3s_containing_edge[e]:
                if q3_violated[q3_idx]:
                    new_viol_delta -= 1
            new_viol = total_violations + new_viol_delta
            new_edge_count = curr_edges - 1

        # Cost: heavily penalize violations, then maximize edges
        curr_cost = total_violations * 10000 - curr_edges
        new_cost = new_viol * 10000 - new_edge_count
        delta_cost = new_cost - curr_cost

        accept = False
        if delta_cost <= 0:
            accept = True
        else:
            # Accept with probability exp(-delta/T)
            if delta_cost < 50 * T:  # avoid overflow
                accept = random.random() < pow(2.718281828459045, -delta_cost / T)

        if accept:
            # Update state
            curr_mask = new_mask
            curr_edges = new_edge_count

            if not (curr_mask & bit):
                # Added edge e
                for q3_idx in q3s_containing_edge[e]:
                    if not q3_violated[q3_idx]:
                        q3_mask = global_q3_masks[q3_idx]
                        other_edges = q3_mask & ~bit
                        if (curr_mask & other_edges) == other_edges:
                            q3_violated[q3_idx] = True
                            total_violations += 1
            else:
                # Removed edge e
                for q3_idx in q3s_containing_edge[e]:
                    if q3_violated[q3_idx]:
                        q3_violated[q3_idx] = False
                        total_violations -= 1

            if total_violations < best_viol or (total_violations == best_viol and curr_edges > best_edges):
                best_viol = total_violations
                best_edges = curr_edges
                best_mask = curr_mask

        T *= cooling_rate

    # If best has violations, fall back to greedy
    if best_viol > 0:
        # Reconstruct from greedy mask (recompute violations)
        final_mask = current_edge_mask
        # Verify greedy is valid
        for i, mask in enumerate(global_q3_masks):
            if (final_mask & mask) == mask:
                # Shouldn't happen, but repair
                break
    else:
        final_mask = best_mask

    result_edges = []
    for e in range(num_edges_total):
        if final_mask & (1 << e):
            u, v = inv_edge_map[e]
            result_edges.append([u, v])

    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
