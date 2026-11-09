import time
import random
import sys
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

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

    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    local_q3_bitmasks = []
    seen_local = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            key = frozenset(edges_set)
            if key not in seen_local:
                seen_local.add(key)
                mask_int = 0
                for e in edges_set:
                    mask_int |= (1 << e)
                local_q3_bitmasks.append(mask_int)

    # Global Q3 masks
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
            m = local_mask
            while m:
                lsb = m & -m
                le = lsb.bit_length() - 1
                ge = l2g_edge[le]
                global_mask |= (1 << ge)
                m ^= lsb
            global_q3_masks.append(global_mask)

    num_q3s = len(global_q3_masks)

    # For each edge, list of Q3 indices containing it
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb

    # Precompute, for each Q3, the list of its 12 edge indices (for fast hit-testing)
    q3_edges_list = []
    for mask in global_q3_masks:
        lst = []
        m = mask
        while m:
            lsb = m & -m
            lst.append(lsb.bit_length() - 1)
            m ^= lsb
        q3_edges_list.append(lst)

    def greedy_set_cover(randomize, seed):
        """Remove edges to hit all Q3s. Returns set of removed edges."""
        rng = random.Random(seed)
        active = [True] * num_q3s
        counts = [len(q3s_containing_edge[e]) for e in range(num_edges_total)]
        removed = set()
        # tie-breaking random values
        if randomize:
            tie = [rng.random() for _ in range(num_edges_total)]
        else:
            tie = [0.0] * num_edges_total

        while True:
            if time.time() > deadline:
                break
            max_count = -1
            best_edge = -1
            best_tie = -1.0
            for e in range(num_edges_total):
                if e in removed:
                    continue
                if counts[e] > max_count or (counts[e] == max_count and tie[e] > best_tie):
                    max_count = counts[e]
                    best_edge = e
                    best_tie = tie[e]
            if best_edge == -1 or max_count == 0:
                break
            removed.add(best_edge)
            for q3_idx in q3s_containing_edge[best_edge]:
                if active[q3_idx]:
                    active[q3_idx] = False
                    for e_idx in q3_edges_list[q3_idx]:
                        if e_idx != best_edge:
                            counts[e_idx] -= 1
        return removed

    def repair_and_refine(removed, seed):
        """Re-add safe edges and do 2-opt swaps. Returns set of remaining edges (present in G)."""
        rng = random.Random(seed)
        present = [False] * num_edges_total
        for e in range(num_edges_total):
            if e not in removed:
                present[e] = True

        def is_safe_to_add(e):
            # A Q3 is created iff some Q3 containing e has all its other 11 edges present
            for q3_idx in q3s_containing_edge[e]:
                all_present = True
                for e2 in q3_edges_list[q3_idx]:
                    if e2 == e:
                        continue
                    if not present[e2]:
                        all_present = False
                        break
                if all_present:
                    return False
            return True

        # Phase A: re-add edges (fixed point)
        removed_list = list(removed)
        rng.shuffle(removed_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in removed_list:
                if time.time() > deadline:
                    break
                if present[e]:
                    continue
                if is_safe_to_add(e):
                    present[e] = True
                    improved = True

        # Phase B: 2-opt — try to add e (currently removed) and remove f (currently present)
        # This keeps edge count same but may enable further re-adds. Try to find a net-gain move.
        # A move: add e, remove f. Net edge count unchanged. But it may unblock adding more edges.
        # Simpler: try to add e and remove the "least useful" present edge f that is in a Q3 with e.
        # We'll do: for each removed e, try to find present f such that after (add e, remove f),
        # the graph is still Q3-free. Accept if it allows a subsequent re-add (net +1).
        # To keep it simple and fast, do a single pass of "add e, remove f" only when it leads to
        # an immediate net gain via re-add.
        for _ in range(3):
            if time.time() > deadline:
                break
            progress = False
            removed_now = [e for e in range(num_edges_total) if not present[e]]
            rng.shuffle(removed_now)
            for e in removed_now:
                if time.time() > deadline:
                    break
                if present[e]:
                    continue
                # candidate f: present edges that share a Q3 with e
                cand_f = set()
                for q3_idx in q3s_containing_edge[e]:
                    for e2 in q3_edges_list[q3_idx]:
                        if e2 != e and present[e2]:
                            cand_f.add(e2)
                for f in cand_f:
                    if time.time() > deadline:
                        break
                    # simulate: add e, remove f
                    present[e] = True
                    present[f] = False
                    # check Q3-free
                    ok = True
                    for q3_idx in range(num_q3s):
                        allp = True
                        for e2 in q3_edges_list[q3_idx]:
                            if not present[e2]:
                                allp = False
                                break
                        if allp:
                            ok = False
                            break
                    if not ok:
                        present[e] = False
                        present[f] = True
                        continue
                    # ok: now try to re-add any removed edge to gain
                    gained = False
                    for e2 in removed_now:
                        if e2 == e or e2 == f or present[e2]:
                            continue
                        if is_safe_to_add(e2):
                            present[e2] = True
                            gained = True
                            break
                    if not gained:
                        # no net gain, revert
                        present[e] = False
                        present[f] = True
                    else:
                        progress = True
            if not progress:
                break

        return present

    best_present = None
    best_count = -1

    # Run multiple greedy passes with different randomization
    num_passes = 12
    for p in range(num_passes):
        if time.time() > deadline - 30:
            break
        randomize = (p > 0)
        seed = 1000 + p * 7
        removed = greedy_set_cover(randomize, seed)
        present = repair_and_refine(removed, seed + 1)
        cnt = sum(present)
        if cnt > best_count:
            best_count = cnt
            best_present = present[:]

    # Final polish on the best: one more aggressive re-add pass
    if best_present is not None:
        present = best_present[:]
        removed = [e for e in range(num_edges_total) if not present[e]]
        random.Random(42).shuffle(removed)
        def is_safe_to_add(e):
            for q3_idx in q3s_containing_edge[e]:
                all_present = True
                for e2 in q3_edges_list[q3_idx]:
                    if e2 == e:
                        continue
                    if not present[e2]:
                        all_present = False
                        break
                if all_present:
                    return False
            return True
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for e in removed:
                if time.time() > deadline - 2:
                    break
                if present[e]:
                    continue
                if is_safe_to_add(e):
                    present[e] = True
                    improved = True
        if sum(present) > best_count:
            best_present = present[:]
            best_count = sum(best_present)

    result_edges = []
    for e in range(num_edges_total):
        if best_present[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])

    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
