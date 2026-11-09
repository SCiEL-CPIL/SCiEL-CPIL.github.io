# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256

    # 1. Precompute conflict graph using bitmasks
    adj = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Complement graph: comp_adj[i] has bit j set iff i != j and NO conflict
    comp_adj = [0] * (N + 1)
    all_bits = (1 << (N + 1)) - 1
    for i in range(1, N + 1):
        comp_adj[i] = (all_bits ^ (1 << i)) & ~adj[i]

    # 2. Greedy initialization: find a good initial independent set
    def greedy_is(order):
        s = set()
        for v in order:
            ok = True
            for u in s:
                if adj[v] & (1 << u):
                    ok = False
                    break
            if ok:
                s.add(v)
        return s

    nums = list(range(1, N + 1))
    best_size = 0
    best_mask = 0

    for order in [nums, list(reversed(nums)),
                  [x for x in nums if x % 2 == 1] + [x for x in nums if x % 2 == 0],
                  [x for x in nums if x % 2 == 0] + [x for x in nums if x % 2 == 1],
                  sorted(nums, key=lambda x: (adj[x]).bit_count()),
                  sorted(nums, key=lambda x: (adj[x]).bit_count(), reverse=True),
                  sorted(nums, key=lambda x: (comp_adj[x]).bit_count(), reverse=True)]:
        s = greedy_is(order)
        if len(s) > best_size:
            best_size = len(s)
            best_mask = 0
            for v in s:
                best_mask |= (1 << v)

    # 3. Tomita's Maximum Clique algorithm on the complement graph
    g_best_size = best_size
    g_best_mask = best_mask

    def color_sort(P):
        """Sequential greedy coloring for tight upper bound.
        Process vertices in decreasing degree order for tighter coloring."""
        if P == 0:
            return [], 0

        verts = []
        temp = P
        while temp:
            bit = temp & (-temp)
            v = bit.bit_length() - 1
            verts.append(v)
            temp ^= bit

        # Sort by degree in induced subgraph descending for tighter coloring
        degs = [((comp_adj[v] & P).bit_count(), v) for v in verts]
        degs.sort(reverse=True)

        color_classes = []
        color_of = {}

        for _, v in degs:
            v_bit = 1 << v
            v_neigh = comp_adj[v] & P
            placed = False
            for ci, cc in enumerate(color_classes):
                if v_neigh & cc == 0:
                    color_classes[ci] |= v_bit
                    color_of[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append(v_bit)
                color_of[v] = len(color_classes) - 1

        num_colors = len(color_classes)
        # Vertices with higher color come first in branching (tighter bound per vertex)
        sorted_verts = sorted(verts, key=lambda x: color_of[x], reverse=True)
        return sorted_verts, num_colors

    def expand(R_size, R_mask, P):
        nonlocal g_best_size, g_best_mask

        if P == 0:
            if R_size > g_best_size:
                g_best_size = R_size
                g_best_mask = R_mask
            return

        if time.time() > deadline:
            return

        # Quick prune: even if we take all remaining, can't beat best
        if R_size + P.bit_count() <= g_best_size:
            return

        sorted_verts, ub = color_sort(P)

        if R_size + ub <= g_best_size:
            return

        current_P = P
        for v in sorted_verts:
            v_bit = 1 << v
            if not (current_P & v_bit):
                continue

            current_P ^= v_bit
            new_P = current_P & comp_adj[v]

            if R_size + 1 + new_P.bit_count() <= g_best_size:
                continue

            expand(R_size + 1, R_mask | v_bit, new_P)

    all_verts = list(range(1, N + 1))
    all_verts.sort(key=lambda x: (comp_adj[x]).bit_count(), reverse=True)

    initial_P = 0
    for v in all_verts:
        initial_P |= (1 << v)

    expand(0, 0, initial_P)

    # 4. Extract result from best_mask
    result = []
    temp = g_best_mask
    while temp:
        bit = temp & (-temp)
        v = bit.bit_length() - 1
        result.append(v)
        temp ^= bit

    if not result:
        result = list(range(1, N + 1, 2))

    # 5. Final verification
    result.sort()
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid or len(result) == 0:
        return list(range(1, N + 1, 2))

    return result
# EVOLVE-BLOCK-END
