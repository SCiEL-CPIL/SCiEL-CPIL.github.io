"""Strong edge colouring: bitmask greedy + aggressive local search + colour merging."""
import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmask(edges):
        """Build the strong edge-colouring conflict graph as bitmasks."""
        m = len(edges)
        # vertex -> list of edge indices incident to it
        v2e = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v2e[u].append(i)
            v2e[v].append(i)

        # adjacency for "joined by an edge" check
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # conflict[i] is a bitmask of all edges that conflict with edge i
        conflict = [0] * m
        for i in range(m):
            u, v = edges[i]
            # All edges sharing a vertex with edge i
            candidates = set(v2e[u]) | set(v2e[v])
            # All edges "joined" by an edge to edge i
            # i.e., edges (x,y) where x or y is in {u,v} or adjacent to u or v
            neighbors = set(adj[u]) | set(adj[v])
            for x in neighbors:
                candidates |= set(v2e[x])
            # Remove self
            candidates.discard(i)
            # Build bitmask
            mask = 0
            for j in candidates:
                if j < m:
                    mask |= (1 << j)
            conflict[i] = mask
        return conflict

    def greedy_color_bitmask(edges, conflict, order):
        """Assign colours using greedy first-fit in the given edge order, tracking colour masks."""
        m = len(edges)
        colour = [0] * m
        # colour_masks[c] is a bitmask of edges with colour c
        colour_masks = [0] * (m + 2)  # 1-indexed colours
        max_colour = 0

        for idx in order:
            # Find the smallest colour not used by any conflicting edge
            # An edge j conflicts with idx iff (conflict[idx] >> j) & 1
            # For colour c to be forbidden, there exists j with colour[j]==c and (conflict[idx]>>j)&1
            # We can check: colour_masks[c] & conflict[idx] != 0
            c = 1
            while colour_masks[c] & conflict[idx]:
                c += 1
            colour[idx] = c
            colour_masks[c] |= (1 << idx)
            if c > max_colour:
                max_colour = c
        return colour, max_colour, colour_masks

    def try_remove_colour(edges, conflict, colour, max_colour, colour_masks, rng, time_limit=3.0):
        """Try to reduce colour count by re-colouring edges from the highest colour."""
        m = len(edges)
        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            for target_c in range(max_colour, 1, -1):
                if time.time() > DEADLINE - 5.0:
                    return colour, max_colour, colour_masks, False
                edges_with_c = []
                mask = colour_masks[target_c]
                if mask == 0:
                    continue
                # Extract edges with this colour
                for i in range(m):
                    if (mask >> i) & 1:
                        edges_with_c.append(i)
                if not edges_with_c:
                    continue
                # Try to re-colour one of them to a lower colour
                rng.shuffle(edges_with_c)
                for i in edges_with_c:
                    # Find forbidden colours for edge i
                    # A colour c is forbidden if colour_masks[c] & conflict[i] != 0
                    found = False
                    for c in range(1, target_c):
                        if not (colour_masks[c] & conflict[i]):
                            # Re-colour edge i to colour c
                            colour[i] = c
                            colour_masks[target_c] &= ~(1 << i)
                            colour_masks[c] |= (1 << i)
                            # Check if target_c is still used
                            if colour_masks[target_c] == 0:
                                max_colour -= 1
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break
        return colour, max_colour, colour_masks, improved

    def two_swap(edges, conflict, colour, max_colour, colour_masks, rng, time_limit=2.0):
        """Try 2-swap moves: swap colours of two non-conflicting edges."""
        m = len(edges)
        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            if time.time() > DEADLINE - 5.0:
                return colour, max_colour, colour_masks, False
            # Try to find two edges that can swap colours to reduce max colour
            # Focus on edges with the highest colour
            target_c = max_colour
            edges_with_c = []
            mask = colour_masks[target_c]
            for i in range(m):
                if (mask >> i) & 1:
                    edges_with_c.append(i)
            if len(edges_with_c) < 2:
                continue
            rng.shuffle(edges_with_c)
            for a in edges_with_c:
                for b in edges_with_c:
                    if a == b:
                        continue
                    # Check if a and b can swap colours
                    # After swap, a gets colour of b, b gets colour of a (which is target_c)
                    # Actually, let's try: a gets some lower colour c1, b gets some lower colour c2
                    # But that's complex. Simpler: try to move edge a to a lower colour,
                    # which is already done in try_remove_colour.
                    # For 2-swap: pick two edges from different colour classes and swap their colours.
                    # This can help if it allows subsequent removal of a colour.
                    pass
            # Alternative 2-swap: pick edge i from colour c1 and edge j from colour c2,
            # swap their colours if valid.
            # For now, let's do a simpler version: try to re-colour an edge to a colour
            # that is currently "light" (few edges) to enable merging.
            # Actually, let's implement a proper 2-swap:
            # Pick two edges i, j with colours ci, cj. Swap if:
            #   - i and j don't conflict
            #   - After swap, i's new colour cj is not used by any conflict of i
            #   - After swap, j's new colour ci is not used by any conflict of j
            # This is complex to check efficiently. Let's do a simpler heuristic:
            # Try to find an edge i with colour c1 and an edge j with colour c2 (c1 != c2)
            # such that i and j don't conflict, and swapping their colours is valid.
            # For efficiency, only try edges with the highest colour.
            for a in edges_with_c:
                ca = colour[a]
                # Try to find b with a different colour such that swap is valid
                # b should have a colour cb that is not forbidden for a
                # and ca should not be forbidden for b
                # And a and b should not conflict
                for b in range(m):
                    if b == a or (conflict[a] >> b) & 1:
                        continue
                    cb = colour[b]
                    if cb == ca:
                        continue
                    # Check if cb is forbidden for a: colour_masks[cb] & conflict[a] != 0
                    # But we're removing b from cb, so we need to check if any OTHER edge with colour cb conflicts with a
                    # This is getting complex. Let's simplify: just try single re-colouring more aggressively.
                    pass
        return colour, max_colour, colour_masks, improved

    def merge_colours(edges, conflict, colour, max_colour, colour_masks, time_limit=2.0):
        """Try to merge two colour classes if their union is an induced matching."""
        m = len(edges)
        t_start = time.time()
        improved = True
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            if time.time() > DEADLINE - 5.0:
                return colour, max_colour, colour_masks, False
            # Try to merge colour c1 into colour c2 (c1 > c2)
            # This is valid if no two edges in the union conflict.
            # i.e., for every edge i with colour c1, colour_masks[c2] & conflict[i] == 0
            # and for every edge j with colour c2, colour_masks[c1] & conflict[j] == 0
            # But since conflict is symmetric, we just need:
            # for every i with colour c1: colour_masks[c2] & conflict[i] == 0
            for c1 in range(max_colour, 1, -1):
                if colour_masks[c1] == 0:
                    continue
                for c2 in range(1, c1):
                    if colour_masks[c2] == 0:
                        continue
                    # Check if we can merge c1 into c2
                    mask_c1 = colour_masks[c1]
                    valid = True
                    for i in range(m):
                        if (mask_c1 >> i) & 1:
                            if colour_masks[c2] & conflict[i]:
                                valid = False
                                break
                    if valid:
                        # Merge: change all edges from c1 to c2
                        for i in range(m):
                            if (mask_c1 >> i) & 1:
                                colour[i] = c2
                        colour_masks[c2] |= mask_c1
                        colour_masks[c1] = 0
                        max_colour -= 1
                        improved = True
                        break
                if improved:
                    break
        return colour, max_colour, colour_masks, improved

    def local_search_bitmask(edges, conflict, colour, max_colour, colour_masks, rng, time_limit=3.0):
        """Combined local search: try_remove_colour, merge_colours, and perturbation."""
        m = len(edges)
        t_start = time.time()
        while time.time() - t_start < time_limit and time.time() < DEADLINE - 5.0:
            # Try to remove a colour
            colour, max_colour, colour_masks, imp1 = try_remove_colour(
                edges, conflict, colour, max_colour, colour_masks, rng, 0.5
            )
            # Try to merge colours
            colour, max_colour, colour_masks, imp2 = merge_colours(
                edges, conflict, colour, max_colour, colour_masks, 0.5
            )
            if not imp1 and not imp2:
                # Perturb: randomly re-colour a few edges
                if max_colour > 1:
                    for _ in range(3):
                        i = rng.randrange(m)
                        old_c = colour[i]
                        # Try to re-colour i to a random lower colour
                        candidates = []
                        for c in range(1, old_c):
                            if not (colour_masks[c] & conflict[i]):
                                candidates.append(c)
                        if candidates:
                            new_c = rng.choice(candidates)
                            colour[i] = new_c
                            colour_masks[old_c] &= ~(1 << i)
                            colour_masks[new_c] |= (1 << i)
            else:
                # Continue searching
                pass
        return colour, max_colour, colour_masks

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        conflict = build_conflict_bitmask(edges)

        # Build vertex degrees for edge ordering
        vdeg = [0] * n
        for u, v in edges:
            vdeg[u] += 1
            vdeg[v] += 1

        best_colour = None
        best_max = m + 1
        best_masks = None

        rng = random.Random(42)

        # Try multiple orderings
        orderings = []
        # 1. Original sorted order
        orderings.append(list(range(m)))
        # 2. Descending by max endpoint degree
        orderings.append(sorted(range(m), key=lambda i: max(vdeg[edges[i][0]], vdeg[edges[i][1]]), reverse=True))
        # 3. Descending by sum of endpoint degrees
        orderings.append(sorted(range(m), key=lambda i: vdeg[edges[i][0]] + vdeg[edges[i][1]], reverse=True))
        # 4. Random shuffles
        for _ in range(30):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)

        for order in orderings:
            if time.time() > DEADLINE - 10.0:
                break
            colour, max_c, colour_masks = greedy_color_bitmask(edges, conflict, order)
            if max_c < best_max:
                best_max = max_c
                best_colour = colour[:]
                best_masks = colour_masks[:]
            # Try local improvement
            if time.time() < DEADLINE - 15.0:
                colour_imp, max_imp, masks_imp = local_search_bitmask(
                    edges, conflict, colour, max_c, colour_masks, rng, 2.0
                )
                if max_imp < best_max:
                    best_max = max_imp
                    best_colour = colour_imp[:]
                    best_masks = masks_imp[:]

        # More random restarts if time permits
        if best_colour is not None and time.time() < DEADLINE - 20.0:
            for _ in range(100):
                if time.time() > DEADLINE - 10.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                colour, max_c, colour_masks = greedy_color_bitmask(edges, conflict, order)
                if max_c < best_max:
                    best_max = max_c
                    best_colour = colour[:]
                    best_masks = colour_masks[:]
                if time.time() < DEADLINE - 15.0:
                    colour_imp, max_imp, masks_imp = local_search_bitmask(
                        edges, conflict, colour, max_c, colour_masks, rng, 2.0
                    )
                    if max_imp < best_max:
                        best_max = max_imp
                        best_colour = colour_imp[:]
                        best_masks = masks_imp[:]

        if best_colour is None:
            # Fallback: just use greedy with original order
            colour, _, _ = greedy_color_bitmask(edges, conflict, list(range(m)))
            return colour
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5.0:
            # Not enough time, return empty lists (shouldn't happen)
            out.append([])
            continue
        result = solve_instance(*spec)
        out.append(result)

    return out
