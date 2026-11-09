"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    start_time = time.time()
    DEADLINE = start_time + 135.0   # safety margin before the 150 s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def color_graph(edges, deadline):
        if not edges:
            return []

        m = len(edges)
        if m == 0:
            return []

        # Build vertex adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Cover of an edge (u,v) = {u,v} ∪ N(u) ∪ N(v)
        # Two edges conflict iff an endpoint of one lies in the other's cover.
        covers = []
        for u, v in edges:
            cov = {u, v}
            cov.update(adj.get(u, set()))
            cov.update(adj.get(v, set()))
            covers.append(cov)

        # Conflict bitmasks
        conflict_masks = [0] * m
        for i in range(m):
            cov_i = covers[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov_i or v_j in cov_i:
                    mask |= (1 << j)
            conflict_masks[i] = mask

        conflict_deg = [bin(cm).count('1') for cm in conflict_masks]
        full_mask = (1 << m) - 1

        # ---- Exact maximum independent set via branch & bound on conflict graph ----
        # An independent set in the conflict graph = a valid induced-matching colour class.
        # We need the largest one to pack each colour class as tightly as possible.
        def max_independent_set(avail, deadline_local):
            # Returns (size, list_of_indices) for a maximum independent set in avail.
            # Branch & bound: order candidates by ascending degree within avail,
            # and bound by greedy upper bound (greedy independent set size).
            best_size = 0
            best_set = []

            # Precompute degree within the available set for ordering
            def deg_in(i, mask):
                return bin(conflict_masks[i] & mask).count('1')

            # Greedy independent set used both as a seed and as an upper bound.
            def greedy_is(mask):
                # pick lowest-degree-first, remove conflicts
                order = [i for i in range(m) if (mask >> i) & 1]
                order.sort(key=lambda i: deg_in(i, mask))
                chosen = []
                cur = mask
                for i in order:
                    if not (cur >> i) & 1:
                        continue
                    chosen.append(i)
                    cur &= ~(conflict_masks[i] | (1 << i))
                return chosen

            seed = greedy_is(avail)
            best_size = len(seed)
            best_set = seed[:]

            # Simple recursive B&B with node limit to respect deadline.
            nodes = [0]

            def rec(cand, chosen, cur_avail):
                # cand: list of candidate indices (bitmask cur_avail)
                if time.time() > deadline_local:
                    return
                nodes[0] += 1
                if not cand:
                    if len(chosen) > best_size:
                        best_size = len(chosen)
                        best_set = chosen[:]
                    return
                # Upper bound: |chosen| + greedy independent set of remaining
                bound = len(chosen) + len(greedy_is(cur_avail))
                if bound <= best_size:
                    return
                # branch on first candidate
                i = cand[0]
                rest = cand[1:]
                # include i
                rec(rest, chosen + [i], cur_avail & ~(conflict_masks[i] | (1 << i)))
                # exclude i
                rec(rest, chosen, cur_avail & ~(1 << i))

            # Use a node budget to keep runtime bounded; the greedy seed is already good.
            deadline_local = min(deadline, start_time + 130.0)
            # Only run B&B if the set is small enough to be tractable.
            if m <= 60:
                rec(list(range(m)), [], avail)

            return best_size, best_set

        # ---- Greedy colouring with a given ordering ----
        def greedy_color(order):
            color = [-1] * m
            color_used_by = []  # bitmask of edges per colour
            max_color = 0
            for i in order:
                cm = conflict_masks[i]
                taken = set()
                for c, mask in enumerate(color_used_by):
                    if cm & mask:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                if c >= len(color_used_by):
                    color_used_by.append(0)
                color_used_by[c] |= (1 << i)
                if c > max_color:
                    max_color = c
            return color, max_color

        # ---- MIS-based packing: repeatedly extract a large independent set ----
        def mis_color(deadline_local):
            color = [-1] * m
            uncolored = full_mask
            color_count = 0
            while uncolored and time.time() < deadline_local:
                # Extract a large independent set among uncolored using B&B/greedy
                size, iset = max_independent_set(uncolored, deadline_local)
                if not iset:
                    break
                for i in iset:
                    color[i] = color_count
                    uncolored &= ~(1 << i)
                color_count += 1
            # Fallback greedy for any leftovers
            remaining = [i for i in range(m) if color[i] == -1]
            for i in remaining:
                cm = conflict_masks[i]
                taken = set()
                for c in range(color_count):
                    mask = 0
                    for j in range(m):
                        if color[j] == c:
                            mask |= (1 << j)
                    if cm & mask:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                color_count = max(color_count, c + 1)
            return color, color_count

        # ---- Local search: try to merge two colour classes ----
        def try_merge(color, c1, c2, deadline_local):
            # Check if all edges of c1 and c2 form a single independent set
            edges_c1 = [i for i in range(m) if color[i] == c1]
            edges_c2 = [i for i in range(m) if color[i] == c2]
            all_edges = edges_c1 + edges_c2
            # Verify no conflicts within the union
            for a in range(len(all_edges)):
                for b in range(a + 1, len(all_edges)):
                    ia, ib = all_edges[a], all_edges[b]
                    if (conflict_masks[ia] >> ib) & 1:
                        return None
            new_color = color[:]
            for i in edges_c2:
                new_color[i] = c1
            return new_color

        def local_search(color, max_color, deadline_local):
            colors_used = sorted(set(color))
            improved = True
            while improved and time.time() < deadline_local:
                improved = False
                # Try merging pairs; when successful, recompute used colours.
                colors_used = sorted(set(color))
                for ci in range(len(colors_used)):
                    for cj in range(ci + 1, len(colors_used)):
                        c1, c2 = colors_used[ci], colors_used[cj]
                        if c2 == max_color:
                            # merging into a lower colour reduces count
                            nc = try_merge(color, c1, c2, deadline_local)
                            if nc is not None:
                                color = nc
                                improved = True
                                break
                    if improved:
                        break
            return color, len(set(color))

        # ---- Generate orderings ----
        orderings = []
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i], reverse=True))
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i]))
        orderings.append(sorted(range(m), key=lambda i: len(adj.get(edges[i][0], set())) + len(adj.get(edges[i][1], set())), reverse=True))

        rng = Random(42)
        for _ in range(15):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)
        for _ in range(10):
            high = [i for i in range(m) if conflict_deg[i] > m // 2]
            low = [i for i in range(m) if conflict_deg[i] <= m // 2]
            rng.shuffle(high)
            rng.shuffle(low)
            orderings.append(high + low)

        best_color = None
        best_max = m + 1

        dl = min(deadline, start_time + 125.0)

        # Greedy attempts
        for order in orderings:
            if time.time() > dl:
                break
            color, mc = greedy_color(order)
            if mc < best_max:
                best_max = mc
                best_color = color[:]

        # MIS packing attempts
        if time.time() < dl:
            color, mc = mis_color(dl)
            if mc < best_max:
                best_max = mc
                best_color = color[:]

        # Local search refinement on the best
        if best_color is not None and time.time() < dl:
            color, mc = local_search(best_color, best_max, dl)
            if mc < best_max:
                best_max = mc
                best_color = color[:]

        return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # Fallback simple greedy
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue

        edges = instance(*spec)
        colors = color_graph(edges, DEADLINE)
        out.append(colors)

    return out
# EVOLVE-BLOCK-END
