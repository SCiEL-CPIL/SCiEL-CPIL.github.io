import time
import math
import heapq
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before the 150s wall-clock kill

    N = 512
    # weights[a] = 1/(512 - a), larger for a closer to 511
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # ---- conflict graph: edge (a,b) iff gcd(a,b) != 1 ----
    # conflict_bits[a] : bitset of numbers b != a with gcd(a,b) != 1
    conflict_bits = [0] * N
    for a in range(1, N):
        bits = 0
        for b in range(1, N):
            if b != a and gcd(a, b) != 1:
                bits |= (1 << b)
        conflict_bits[a] = bits

    def get_score(s):
        return sum(weights[a] for a in s)

    # ---- warm start: greedy (highest weight first) + local search ----
    chosen = []
    chosen_set = set()
    for a in range(N - 1, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)

    best_score = get_score(chosen)
    best_set = list(chosen)

    def is_coprime_to_all(a, s):
        return all(gcd(a, b) == 1 for b in s)

    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 30.0:
        improved = False
        for i in range(len(chosen)):
            if (time.time() - start_time) > DEADLINE - 30.0:
                break
            a = chosen[i]
            others = [x for x in chosen if x != a]
            others_set = chosen_set - {a}
            # find largest valid replacement b > a
            for b in range(N - 1, a, -1):
                if b in others_set:
                    continue
                if is_coprime_to_all(b, others):
                    new_score = best_score - weights[a] + weights[b]
                    if new_score > best_score + 1e-15:
                        chosen[i] = b
                        chosen_set.remove(a)
                        chosen_set.add(b)
                        best_score = new_score
                        improved = True
                        break
            if improved:
                break

    # ---- exact Branch-and-Bound (best-first, coloring upper bound) ----
    # State: (available_mask, current_score, chosen_list)
    # Upper bound = current_score + sum of per-color-class max weight among available.
    # A color class is a set of pairwise-conflicting vertices, so at most one
    # can be taken -> sum of class maxima is a valid upper bound.

    def color_bound(available_mask):
        """Greedy (weight-desc) coloring of the available conflict subgraph.
        Returns sum over color classes of the max weight in that class."""
        if available_mask == 0:
            return 0.0
        # collect available vertices
        verts = []
        m = available_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        # order by weight descending
        verts.sort(key=lambda v: weights[v], reverse=True)
        class_max = []          # max weight per color class
        class_conflict = []     # bitset of vertices in that class (for membership)
        placed = [False] * len(verts)
        for idx, v in enumerate(verts):
            cv = conflict_bits[v]
            best_c = -1
            best_w = -1.0
            for c in range(len(class_max)):
                # v conflicts with every member of class c?
                if (cv & class_conflict[c]) == class_conflict[c]:
                    if weights[v] > best_w:
                        best_w = weights[v]
                        best_c = c
            if best_c >= 0:
                if weights[v] > class_max[best_c]:
                    class_max[best_c] = weights[v]
                class_conflict[best_c] |= (1 << v)
            else:
                class_max.append(weights[v])
                class_conflict.append(1 << v)
        return sum(class_max)

    init_avail = 0
    for a in range(1, N):
        init_avail |= (1 << a)

    # priority queue, max-heap on (bound, score)
    # entry: (-bound, -score, avail_mask, chosen_tuple)
    init_bound = best_score + color_bound(init_avail)
    pq = []
    heapq.heappush(pq, (-init_bound, -best_score, init_avail, tuple(chosen)))

    best_bb_score = best_score
    best_bb_set = list(chosen)
    nodes = 0

    while pq and (time.time() - start_time) < DEADLINE:
        neg_bound, neg_score, avail, chosen_tuple = heapq.heappop(pq)
        bound = -neg_bound
        score = -neg_score
        if bound <= best_bb_score + 1e-15:
            # cannot improve; since best-first, no remaining node can either
            break
        if avail == 0:
            if score > best_bb_score:
                best_bb_score = score
                best_bb_set = list(chosen_tuple)
            continue
        nodes += 1

        # choose branch vertex: highest weight available
        m = avail
        lsb = m & (-m)
        v = lsb.bit_length() - 1

        # Option 1: take v
        take_avail = avail & ~conflict_bits[v] & ~(1 << v)
        take_score = score + weights[v]
        take_bound = take_score + color_bound(take_avail)
        if take_bound > best_bb_score + 1e-15:
            heapq.heappush(pq, (-take_bound, -take_score, take_avail, chosen_tuple + (v,)))
        elif take_score > best_bb_score:
            best_bb_score = take_score
            best_bb_set = list(chosen_tuple) + [v]

        # Option 2: skip v
        skip_avail = avail & ~(1 << v)
        skip_bound = score + color_bound(skip_avail)
        if skip_bound > best_bb_score + 1e-15:
            heapq.heappush(pq, (-skip_bound, -score, skip_avail, chosen_tuple))

    if best_bb_score >= best_score:
        best_score = best_bb_score
        best_set = best_bb_set

    # ---- final safety: guarantee a valid pairwise-coprime set ----
    if not best_set:
        best_set = list(chosen)
    # validate
    ok = True
    seen = set()
    for a in best_set:
        if a in seen or a < 1 or a > 511:
            ok = False
            break
        seen.add(a)
    if ok:
        for i in range(len(best_set)):
            for j in range(i + 1, len(best_set)):
                if gcd(best_set[i], best_set[j]) != 1:
                    ok = False
                    break
            if not ok:
                break
    if not ok:
        # rebuild a valid greedy set
        rebuilt = []
        for a in range(N - 1, 0, -1):
            if all(gcd(a, b) == 1 for b in rebuilt):
                rebuilt.append(a)
        best_set = rebuilt

    return best_set
