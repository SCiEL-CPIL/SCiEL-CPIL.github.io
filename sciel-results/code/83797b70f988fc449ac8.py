import time
import sys
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256

    def lcm(a, b):
        return a // gcd(a, b) * b

    # Precompute global conflict: adj_val[v] = set of u such that lcm(u,v) <= 256
    adj_val = [set() for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= N:
                adj_val[a].add(b)
                adj_val[b].add(a)

    def solve_mwis(values, time_budget):
        """Solve max-weight independent set on the induced subgraph of `values`
        using branch-and-bound with strong local search. Returns (sum, set_of_values)."""
        if not values:
            return 0.0, []

        verts = sorted(values)
        n = len(verts)
        val_to_idx = {v: i for i, v in enumerate(verts)}

        # Local adjacency (as bitmasks for speed)
        adj = [0] * n
        for i in range(n):
            vi = verts[i]
            mask = 0
            for j in range(i + 1, n):
                if verts[j] in adj_val[vi]:
                    mask |= (1 << j)
                    adj[j] |= (1 << i)
            adj[i] = mask

        weights = [1.0 / v for v in verts]

        # Greedy seed (multiple orders)
        def greedy(order):
            chosen = []
            avail = (1 << n) - 1
            for i in order:
                if (avail >> i) & 1:
                    chosen.append(i)
                    avail &= ~(adj[i] | (1 << i))
            return chosen

        # Local search: 1-add, 1-swap, 2-swap
        def local_search(chosen, deadline_local):
            current = set(chosen)
            cur_sum = sum(weights[i] for i in current)
            cur_mask = 0
            for i in current:
                cur_mask |= (1 << i)

            # Precompute conflict-with-current for each non-chosen
            def conflicts_with_current(i):
                return adj[i] & cur_mask

            improved = True
            while improved and time.time() < deadline_local:
                improved = False
                # 1-add
                for i in range(n):
                    if (cur_mask >> i) & 1:
                        continue
                    if not (adj[i] & cur_mask):
                        current.add(i)
                        cur_mask |= (1 << i)
                        cur_sum += weights[i]
                        improved = True
                if not improved:
                    # 1-swap
                    best_gain = 0.0
                    best_i = -1
                    best_j = -1
                    for j in range(n):
                        if (cur_mask >> j) & 1:
                            continue
                        C = adj[j] & cur_mask
                        if C == 0:
                            continue
                        # find min weight in C
                        mn = 1e9
                        mni = -1
                        cm = C
                        while cm:
                            lb = cm & (-cm)
                            i2 = lb.bit_length() - 1
                            if weights[i2] < mn:
                                mn = weights[i2]
                                mni = i2
                            cm &= cm - 1
                        gain = weights[j] - mn
                        if gain > best_gain:
                            best_gain = gain
                            best_i = mni
                            best_j = j
                    if best_gain > 0:
                        current.discard(best_i)
                        current.add(best_j)
                        cur_mask &= ~(1 << best_i)
                        cur_mask |= (1 << best_j)
                        cur_sum += best_gain
                        improved = True
                if not improved:
                    # 2-swap
                    # candidate non-chosen with |C| <= 2
                    cand = []
                    for j in range(n):
                        if (cur_mask >> j) & 1:
                            continue
                        C = adj[j] & cur_mask
                        if 1 <= (C & (C - 1) == 0 or bin(C).count('1') <= 2) and C:
                            # count bits
                            cc = C
                            cnt = 0
                            while cc:
                                cnt += 1
                                cc &= cc - 1
                            if 1 <= cnt <= 2:
                                cand.append((j, C))
                    cand.sort(key=lambda x: weights[x[0]], reverse=True)
                    cand = cand[:60]
                    best_gain = 0.0
                    best_move = None
                    for a in range(len(cand)):
                        j1, C1 = cand[a]
                        for b in range(a + 1, len(cand)):
                            j2, C2 = cand[b]
                            if (adj[j1] >> j2) & 1:
                                continue
                            U = C1 | C2
                            cc = U
                            cnt = 0
                            while cc:
                                cnt += 1
                                cc &= cc - 1
                            if cnt > 2 or cnt == 0:
                                continue
                            gain = weights[j1] + weights[j2]
                            um = U
                            while um:
                                lb = um & (-um)
                                i2 = lb.bit_length() - 1
                                gain -= weights[i2]
                                um &= um - 1
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (U, j1, j2)
                    if best_move is not None and best_gain > 0:
                        U, j1, j2 = best_move
                        um = U
                        while um:
                            lb = um & (-um)
                            i2 = lb.bit_length() - 1
                            current.discard(i2)
                            um &= um - 1
                        current.add(j1)
                        current.add(j2)
                        cur_mask &= ~U
                        cur_mask |= (1 << j1) | (1 << j2)
                        cur_sum += best_gain
                        improved = True
            return cur_sum, current

        # Branch and bound
        # Use a recursive solver with ordering by weight.
        # Bound: fractional greedy upper bound.
        best_sum = -1.0
        best_set = None

        # Initial solution via multiple greedy seeds + local search
        random.seed(12345)
        seeds = []
        # seed 1: by weight desc
        seeds.append(sorted(range(n), key=lambda i: verts[i]))
        # seed 2: by weight asc (large values first)
        seeds.append(sorted(range(n), key=lambda i: -verts[i]))
        # seed 3: random
        for _ in range(3):
            s = list(range(n))
            random.shuffle(s)
            seeds.append(s)

        for s in seeds:
            g = greedy(s)
            ls_sum, ls_set = local_search(g, deadline - 8.0)
            if ls_sum > best_sum:
                best_sum = ls_sum
                best_set = set(ls_set)

        # Branch and bound to improve
        # We branch on a vertex: include or exclude.
        # Use a good ordering: pick vertex with high weight / high degree.
        # Bound: current_sum + sum of weights of all vertices that could still be added
        #   (i.e., not conflicting with chosen). A tighter bound: greedy fractional on remaining.

        def bound(remaining_mask, chosen_mask):
            # upper bound = chosen_sum + greedy fractional over remaining independent of chosen
            # simple bound: sum of weights of vertices in remaining not conflicting with chosen
            # and not conflicting with each other (relaxation: ignore mutual conflicts)
            total = 0.0
            m = remaining_mask
            # vertices that conflict with chosen are already removed from remaining
            while m:
                lb = m & (-m)
                i = lb.bit_length() - 1
                total += weights[i]
                m &= m - 1
            return total

        # Better bound: for remaining, take a greedy independent set weight as a lower bound,
        # and use LP-like: sort by weight, take greedily.
        def greedy_bound(remaining_mask):
            total = 0.0
            avail = remaining_mask
            while avail:
                # pick max weight vertex in avail
                best_i = -1
                best_w = -1.0
                m = avail
                while m:
                    lb = m & (-m)
                    i = lb.bit_length() - 1
                    if weights[i] > best_w:
                        best_w = weights[i]
                        best_i = i
                    m &= m - 1
                if best_i < 0:
                    break
                total += weights[best_i]
                avail &= ~(adj[best_i] | (1 << best_i))
            return total

        # Branch and bound with time limit
        # State: chosen_mask, remaining_mask (candidates not yet decided, excluding those conflicting with chosen)
        # We maintain chosen_sum.
        # Order: pick a vertex to branch on (high weight, high degree).

        def bb(chosen_mask, remaining_mask, chosen_sum):
            nonlocal best_sum, best_set
            if time.time() > deadline - 6.0:
                return
            if remaining_mask == 0:
                if chosen_sum > best_sum:
                    best_sum = chosen_sum
                    best_set = set(i for i in range(n) if (chosen_mask >> i) & 1)
                return
            # bound
            ub = chosen_sum + greedy_bound(remaining_mask)
            if ub <= best_sum + 1e-12:
                return
            # pick branch vertex: highest weight in remaining
            m = remaining_mask
            best_i = -1
            best_w = -1.0
            while m:
                lb = m & (-m)
                i = lb.bit_length() - 1
                if weights[i] > best_w:
                    best_w = weights[i]
                    best_i = i
                m &= m - 1
            # Branch: include best_i first (likely better)
            # Include
            new_chosen = chosen_mask | (1 << best_i)
            new_remaining = (remaining_mask & ~(1 << best_i)) & ~adj[best_i]
            bb(new_chosen, new_remaining, chosen_sum + weights[best_i])
            if time.time() > deadline - 6.0:
                return
            # Exclude
            new_remaining = remaining_mask & ~(1 << best_i)
            bb(chosen_mask, new_remaining, chosen_sum)

        # Run B&B from a few starting states (the best local-search solution as a seed
        # is hard to encode as a B&B start; instead run B&B from scratch with the
        # incumbent set as initial best).
        # To keep it fast, run B&B with a node limit.
        node_count = [0]
        max_nodes = 200000

        def bb_limited(chosen_mask, remaining_mask, chosen_sum):
            nonlocal best_sum, best_set
            node_count[0] += 1
            if time.time() > deadline - 6.0 or node_count[0] > max_nodes:
                return
            if remaining_mask == 0:
                if chosen_sum > best_sum:
                    best_sum = chosen_sum
                    best_set = set(i for i in range(n) if (chosen_mask >> i) & 1)
                return
            ub = chosen_sum + greedy_bound(remaining_mask)
            if ub <= best_sum + 1e-12:
                return
            m = remaining_mask
            best_i = -1
            best_w = -1.0
            while m:
                lb = m & (-m)
                i = lb.bit_length() - 1
                if weights[i] > best_w:
                    best_w = weights[i]
                    best_i = i
                m &= m - 1
            # Include
            new_chosen = chosen_mask | (1 << best_i)
            new_remaining = (remaining_mask & ~(1 << best_i)) & ~adj[best_i]
            bb_limited(new_chosen, new_remaining, chosen_sum + weights[best_i])
            if time.time() > deadline - 6.0 or node_count[0] > max_nodes:
                return
            # Exclude
            new_remaining = remaining_mask & ~(1 << best_i)
            bb_limited(chosen_mask, new_remaining, chosen_sum)

        if time.time() < deadline - 10.0:
            bb_limited(0, (1 << n) - 1, 0.0)

        # Final local search polish on best_set
        if best_set is None:
            best_set = set()
        ls_sum, ls_set = local_search(list(best_set), deadline - 3.0)
        if ls_sum > best_sum:
            best_sum = ls_sum
            best_set = ls_set

        return best_sum, [verts[i] for i in best_set]

    # Strategy: at most one number from 2..16 (all mutually conflict).
    # Case 0: none from 2..16.
    V0 = set(range(17, 257))
    sum0, set0 = solve_mwis(V0, deadline - 5.0)
    best_score = sum0
    best_candidate = set0

    # Case x: exactly one small x in 2..16.
    for x in range(2, 17):
        if time.time() > deadline - 5.0:
            break
        Vx = []
        for y in range(2, 257):
            if y == x:
                continue
            if lcm(x, y) > 256:
                Vx.append(y)
        if not Vx:
            score = 1.0 / x
            set_x = [x]
        else:
            sum_x, set_x = solve_mwis(Vx, deadline - 5.0)
            score = 1.0 / x + sum_x
            set_x.append(x)
        if score > best_score:
            best_score = score
            best_candidate = set_x

    return best_candidate
