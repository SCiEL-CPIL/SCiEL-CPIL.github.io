"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 128
    all_elements = list(range(1, N + 1))
    M = len(all_elements)
    idx = {v: i for i, v in enumerate(all_elements)}  # v -> 0..127

    # Pre-compute LCM table for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = lcm(i, j)

    # Find all bad triples: (a, b, c) with a < b < c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))

    if time.time() > deadline:
        return list(range(1, N + 1))

    # Conflict structure: for each v, list of (u, w) with {v,u,w} bad.
    # Also build a bitmask per (v, u) of w's, so can_add is a single bitwise AND.
    conflict_map = defaultdict(list)
    degree = defaultdict(int)
    for a, b, c in bad_triples:
        conflict_map[a].append((b, c))
        conflict_map[b].append((a, c))
        conflict_map[c].append((a, b))
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1

    # Bitmask per (v, u): bits of all w such that {v,u,w} is a bad triple.
    mask = {}
    for v, pairs in conflict_map.items():
        for u, w in pairs:
            mask[(v, u)] = mask.get((v, u), 0) | (1 << idx[w])

    def can_add(v, sel_mask):
        # True if adding v to the set described by sel_mask stays valid.
        for u, w in conflict_map[v]:
            if (sel_mask >> idx[u]) & 1:
                if mask[(v, u)] & sel_mask:
                    return False
        return True

    # Saturation pass: repeatedly add any element that fits, until fixed point.
    def try_add_more(sel_mask):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in all_elements:
                if not (sel_mask >> idx[v]) & 1:
                    if can_add(v, sel_mask):
                        sel_mask |= (1 << idx[v])
                        improved = True
        return sel_mask

    # ---- Phase 1: greedy with multiple orderings ----
    best_mask = 0
    best_size = 0

    def greedy(order):
        sel = 0
        for v in order:
            if time.time() > deadline:
                break
            if can_add(v, sel):
                sel |= (1 << idx[v])
        return sel

    # Deterministic: least-constrained first
    order_deg = sorted(all_elements, key=lambda x: degree[x])
    m = greedy(order_deg)
    best_mask, best_size = m, m.bit_count()

    # ---- Phase 2: randomized restarts (diverse orderings) ----
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        strategy = restart % 5
        if strategy == 0:
            pert = sorted(all_elements, key=lambda x: degree[x] + random.random())
        elif strategy == 1:
            pert = sorted(all_elements, key=lambda x: -degree[x] + random.random())
        elif strategy == 2:
            pert = all_elements[:]
            random.shuffle(pert)
        elif strategy == 3:
            buckets = defaultdict(list)
            for x in all_elements:
                buckets[degree[x]].append(x)
            pert = []
            for d in sorted(buckets.keys()):
                b = buckets[d]
                random.shuffle(b)
                pert.extend(b)
        else:
            # Inverse-degree weighted sampling
            remaining = all_elements[:]
            pert = []
            while remaining:
                wsum = sum(1.0 / (degree[x] + 1) for x in remaining)
                r = random.random() * wsum
                cum = 0.0
                for i, x in enumerate(remaining):
                    cum += 1.0 / (degree[x] + 1)
                    if cum >= r:
                        pert.append(x)
                        remaining.pop(i)
                        break
        m = greedy(pert)
        # Saturate the greedy result (cheap, often adds a few elements)
        m = try_add_more(m)
        sz = m.bit_count()
        if sz > best_size:
            best_mask, best_size = m, sz

    # ---- Phase 3: local search ----
    cur = best_mask
    cur_size = best_size

    # 3a. 1-for-2 swap: remove one, add two that fit -> net +1
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for v in all_elements:
                if not (cur >> idx[v]) & 1:
                    continue
                if time.time() > deadline - 1.0:
                    break
                base = cur & ~(1 << idx[v])
                # Candidates that fit on base, sorted by degree (least constrained first)
                cands = [w for w in all_elements
                         if not (base >> idx[w]) & 1 and can_add(w, base)]
                cands.sort(key=lambda x: degree[x])
                top = cands[:25]
                found = False
                for i in range(len(top)):
                    for j in range(i + 1, len(top)):
                        w1, w2 = top[i], top[j]
                        if can_add(w2, base | (1 << idx[w1])):
                            new_set = base | (1 << idx[w1]) | (1 << idx[w2])
                            if new_set.bit_count() > cur_size:
                                cur = new_set
                                cur_size = cur.bit_count()
                                improved = True
                                found = True
                                break
                    if found:
                        break

    # 3b. 1-swap that enables new additions: replace v_out with v_in, then saturate.
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            selected = [v for v in all_elements if (cur >> idx[v]) & 1]
            unselected = [v for v in all_elements if not (cur >> idx[v]) & 1]
            for v_out in selected:
                if time.time() > deadline - 1.0:
                    break
                base = cur & ~(1 << idx[v_out])
                for v_in in unselected:
                    if can_add(v_in, base):
                        cand = base | (1 << idx[v_in])
                        cand = try_add_more(cand)
                        if cand.bit_count() > cur_size:
                            cur = cand
                            cur_size = cand.bit_count()
                            improved = True
                            break
                if improved:
                    break

    if cur_size > best_size:
        best_mask, best_size = cur, cur_size

    # Final safety saturation
    best_mask = try_add_more(best_mask)
    best_size = best_mask.bit_count()

    # Ensure non-empty
    result = [v for v in all_elements if (best_mask >> idx[v]) & 1]
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
