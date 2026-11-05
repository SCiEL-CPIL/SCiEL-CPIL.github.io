import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 115  # safety margin before 150s kill

    # Precompute all 4-APs
    aps = []
    aps_at = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a + d, a + 2 * d, a + 3 * d)
            aps.append(idxs)
            for i in range(4):
                cur = idxs[i]
                others = tuple(idxs[j] for j in range(4) if j != i)
                aps_at[cur].append(others)

    num_aps = len(aps)

    def count_mono(cols):
        cnt = 0
        for ap in aps:
            if cols[ap[0]] == cols[ap[1]] == cols[ap[2]] == cols[ap[3]]:
                cnt += 1
        return cnt

    def delta_flip(cols, i):
        """Compute cost change if we flip position i."""
        d = 0
        ci = cols[i]
        for (j, k, l) in aps_at[i]:
            cj, ck, cl = cols[j], cols[k], cols[l]
            # Before: is it mono?
            if ci == cj == ck == cl:
                d -= 1
            # After: would it be mono?
            if (1 - ci) == cj == ck == cl:
                d += 1
        return d

    def delta_2flip(cols, i, j):
        """Compute cost change if we flip both positions i and j simultaneously."""
        d = 0
        ci, cj = cols[i], cols[j]
        nci, ncj = 1 - ci, 1 - cj
        for (a, b, c) in aps_at[i]:
            ca, cb, cc = cols[a], cols[b], cols[c]
            # Before
            if ci == ca == cb == cc:
                d -= 1
            # After: i flipped to nci, j might also be in this AP
            # Need to check if j is among a,b,c
            if j == a:
                # j also flips
                if nci == ncj == cb == cc:
                    d += 1
            elif j == b:
                if nci == ca == ncj == cc:
                    d += 1
            elif j == c:
                if nci == ca == cb == ncj:
                    d += 1
            else:
                # j not in this AP, only i flips
                if nci == ca == cb == cc:
                    d += 1
        for (a, b, c) in aps_at[j]:
            # Skip APs already counted from i's perspective (where i is in the AP)
            if i == a or i == b or i == c:
                continue
            ca, cb, cc = cols[a], cols[b], cols[c]
            # Before
            if cj == ca == cb == cc:
                d -= 1
            # After: j flips to ncj, i not in this AP
            if ncj == ca == cb == cc:
                d += 1
        return d

    def local_search_1flip(cols, max_iters=None, deadline=deadline):
        """1-flip local search: flip if it reduces cost, or accept with small prob."""
        cost = count_mono(cols)
        cols = cols[:]
        stall = 0
        iters = 0
        while time.time() < deadline:
            i = random.randint(0, N - 1)
            d = delta_flip(cols, i)
            if d < 0:
                cols[i] = 1 - cols[i]
                cost += d
                stall = 0
            elif d == 0:
                if random.random() < 0.3:
                    cols[i] = 1 - cols[i]
                stall = 0
            else:
                # Accept with small probability to escape
                if random.random() < 0.05:
                    cols[i] = 1 - cols[i]
                    cost += d
                else:
                    stall += 1
            iters += 1
            if max_iters and iters >= max_iters:
                break
            if stall > 2000:
                break
        return cols, cost

    def local_search_2flip(cols, deadline=deadline):
        """2-flip local search: try pairs of flips."""
        cost = count_mono(cols)
        cols = cols[:]
        stall = 0
        while time.time() < deadline:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            d = delta_2flip(cols, i, j)
            if d < 0:
                cols[i] = 1 - cols[i]
                cols[j] = 1 - cols[j]
                cost += d
                stall = 0
            elif d == 0:
                if random.random() < 0.2:
                    cols[i] = 1 - cols[i]
                    cols[j] = 1 - cols[j]
                stall = 0
            else:
                if random.random() < 0.03:
                    cols[i] = 1 - cols[i]
                    cols[j] = 1 - cols[j]
                    cost += d
                else:
                    stall += 1
            if stall > 3000:
                break
        return cols, cost

    # Generate diverse initial colorings
    def init_random():
        return [random.randint(0, 1) for _ in range(N)]

    def init_thue_morse():
        """Thue-Morse sequence: t(n) = parity of number of 1s in binary n."""
        cols = []
        for i in range(N):
            cols.append(bin(i).count('1') % 2)
        return cols

    def init_greedy():
        """Greedy: assign each position the color that minimizes current mono-4-AP count."""
        cols = [None] * N
        # Process in order, for each position try both colors
        # We need to count mono APs incrementally
        # For position i, an AP is mono if all 4 positions have same color
        # When we assign cols[i], we can check all APs containing i where other 3 are already assigned
        # But APs where other positions aren't assigned yet can't be evaluated
        # Simpler: just do random init and let local search handle it
        return init_random()

    def init_block_pattern():
        """Alternating blocks of size 4."""
        return [(i // 4) % 2 for i in range(N)]

    def init_block_pattern2():
        """Alternating blocks of size 8."""
        return [(i // 8) % 2 for i in range(N)]

    # Multi-start strategy
    best_cost = float('inf')
    best_cols = None

    # Time budget per start
    num_starts = 8
    time_per_start = (deadline - time.time()) / num_starts

    inits = [
        init_random,
        init_thue_morse,
        init_random,
        init_block_pattern,
        init_random,
        init_block_pattern2,
        init_random,
        init_thue_morse,
    ]

    for start_idx in range(num_starts):
        if time.time() >= deadline - 5:
            break

        cols = inits[start_idx]()
        cost = count_mono(cols)

        # Phase 1: 1-flip local search
        cols, cost = local_search_1flip(cols, deadline=deadline)
        if cost < best_cost:
            best_cost = cost
            best_cols = cols[:]

        # Phase 2: 2-flip local search from the result
        if time.time() < deadline - 5:
            cols2, cost2 = local_search_2flip(cols, deadline=deadline)
            if cost2 < best_cost:
                best_cost = cost2
                best_cols = cols2[:]

        # Phase 3: another 1-flip pass from the 2-flip result
        if time.time() < deadline - 5:
            cols3, cost3 = local_search_1flip(cols2, deadline=deadline)
            if cost3 < best_cost:
                best_cost = cost3
                best_cols = cols3[:]

    # Final verification
    if best_cols is None:
        best_cols = [0] * N
        best_cost = count_mono(best_cols)

    return best_cols
