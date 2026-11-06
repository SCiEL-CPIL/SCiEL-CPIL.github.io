"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from random import Random

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit

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

    def build_conflict_bitmask(edges, adj):
        """Build conflict graph as list of bitmasks (Python ints)."""
        m = len(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}
        conflict = [0] * m

        for i, (u, v) in enumerate(edges):
            # Edges sharing a vertex with u or v
            for w in adj.get(u, ()):
                e2 = (min(u, w), max(u, w))
                if e2 in edge_idx and e2 != (u, v):
                    j = edge_idx[e2]
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
            for w in adj.get(v, ()):
                e2 = (min(v, w), max(v, w))
                if e2 in edge_idx and e2 != (u, v):
                    j = edge_idx[e2]
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
            # Edges at distance 2 from u or v (induced matching constraint)
            for x in adj.get(u, ()):
                for y in adj.get(x, ()):
                    if y == u:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
            for x in adj.get(v, ()):
                for y in adj.get(x, ()):
                    if y == v:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        j = edge_idx[e2]
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        return conflict

    def greedy_colour(conflict, m, order):
        """Greedy colouring with bitmask feasibility check."""
        colour = [-1] * m
        colour_count = 0
        # For each colour, track which edges have that colour (as bitmask)
        colour_masks = []  # list of bitmasks, one per colour

        for i in order:
            # Check which colours are available
            assigned = False
            for c in range(len(colour_masks)):
                if (conflict[i] & colour_masks[c]) == 0:
                    colour[i] = c
                    colour_masks[c] |= (1 << i)
                    assigned = True
                    break
            if not assigned:
                colour[i] = len(colour_masks)
                colour_masks.append(1 << i)
            if colour[i] >= colour_count:
                colour_count = colour[i] + 1

        return colour, colour_count

    def try_remove_colour(colour, colour_count, conflict, m, rng):
        """Try to eliminate the highest colour class by moving its edges."""
        if colour_count <= 1:
            return colour, colour_count, False

        max_c = colour_count - 1
        to_move = [i for i in range(m) if colour[i] == max_c]
        if not to_move:
            return colour, colour_count - 1, True

        # Try to reassign each edge in to_move to a lower colour
        new_colour = list(colour)
        # Build colour masks for current assignment (excluding max_c edges)
        colour_masks = [0] * max_c
        for i in range(m):
            if colour[i] < max_c:
                colour_masks[colour[i]] |= (1 << i)

        for i in to_move:
            placed = False
            for c in range(max_c):
                if (conflict[i] & colour_masks[c]) == 0:
                    new_colour[i] = c
                    colour_masks[c] |= (1 << i)
                    placed = True
                    break
            if not placed:
                return colour, colour_count, False

        return new_colour, max_c, True

    def simulated_annealing(conflict, m, colour, colour_count, rng, time_budget):
        """Simulated annealing to reduce colour count."""
        best_colour = list(colour)
        best_cc = colour_count
        current_colour = list(colour)
        current_cc = colour_count

        start_time = time.time()
        T = 1.0
        cooling = 0.995
        min_T = 0.01

        while time.time() - start_time < time_budget and T > min_T:
            # Pick a random edge
            i = rng.randrange(m)
            old_c = current_colour[i]

            # Try to move it to another colour
            candidates = list(range(current_cc))
            candidates.remove(old_c)
            rng.shuffle(candidates)

            moved = False
            for c in candidates:
                # Check feasibility: no conflict with edges of colour c
                # Build mask of edges with colour c
                mask_c = 0
                for j in range(m):
                    if current_colour[j] == c:
                        mask_c |= (1 << j)
                if (conflict[i] & mask_c) == 0:
                    current_colour[i] = c
                    moved = True
                    break

            if not moved:
                T *= cooling
                continue

            new_cc = current_cc
            # Check if we can remove the old colour
            old_still_used = False
            for j in range(m):
                if current_colour[j] == old_c:
                    old_still_used = True
                    break

            if not old_still_used and old_c == current_cc - 1:
                new_cc = current_cc - 1

            # Accept or reject
            delta = new_cc - current_cc
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                if new_cc < best_cc:
                    best_cc = new_cc
                    best_colour = list(current_colour)
                current_cc = new_cc
            else:
                current_colour[i] = old_c

            # Periodically try to remove top colour
            if rng.random() < 0.1:
                current_colour, current_cc, _ = try_remove_colour(
                    current_colour, current_cc, conflict, m, rng
                )
                if current_cc < best_cc:
                    best_cc = current_cc
                    best_colour = list(current_colour)

            T *= cooling

        return best_colour, best_cc

    def local_search_compression(conflict, m, colour, colour_count, rng, time_budget):
        """Iteratively try to remove the highest colour class."""
        start_time = time.time()
        while time.time() - start_time < time_budget:
            colour, colour_count, success = try_remove_colour(
                colour, colour_count, conflict, m, rng
            )
            if not success:
                break
        return colour, colour_count

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            out.append([])
            continue

        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue

        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        conflict = build_conflict_bitmask(edges, adj)

        # Multiple greedy restarts with different orderings
        rng = Random(42)
        best_colour = None
        best_cc = float('inf')

        # Deterministic orderings
        orderings = []
        # By degree descending
        orderings.append(sorted(range(m), key=lambda i: -conflict[i].bit_count()))
        # By degree ascending
        orderings.append(sorted(range(m), key=lambda i: conflict[i].bit_count()))
        # Natural order
        orderings.append(list(range(m)))

        # Random orderings
        for _ in range(20):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)

        for order in orderings:
            if time.time() > deadline:
                break
            c, cc = greedy_colour(conflict, m, order)
            if cc < best_cc:
                best_cc = cc
                best_colour = list(c)

        if best_colour is None:
            out.append([])
            continue

        # Local search compression
        remaining_time = deadline - time.time()
        alloc_local = min(5.0, remaining_time * 0.2)
        best_colour, best_cc = local_search_compression(
            conflict, m, best_colour, best_cc, rng, alloc_local
        )

        # Simulated annealing
        remaining_time = deadline - time.time()
        alloc_sa = min(remaining_time * 0.5, 40.0)
        if alloc_sa > 1.0:
            sa_colour, sa_cc = simulated_annealing(
                conflict, m, best_colour, best_cc, rng, alloc_sa
            )
            if sa_cc < best_cc:
                best_cc = sa_cc
                best_colour = sa_colour
                # Compress after SA
                remaining_time = deadline - time.time()
                if remaining_time > 1.0:
                    best_colour, best_cc = local_search_compression(
                        conflict, m, best_colour, best_cc, rng,
                        min(3.0, remaining_time * 0.3)
                    )

        # More greedy restarts if time remains
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            for _ in range(30):
                if time.time() > deadline:
                    break
                order = list(range(m))
                rng.shuffle(order)
                c, cc = greedy_colour(conflict, m, order)
                if cc < best_cc:
                    best_cc = cc
                    best_colour = list(c)
                    # Quick compression
                    remaining_time = deadline - time.time()
                    if remaining_time > 1.0:
                        best_colour, best_cc = local_search_compression(
                            conflict, m, best_colour, best_cc, rng,
                            min(1.0, remaining_time * 0.2)
                        )

        out.append([best_colour[i] for i in range(m)])

    return out
# EVOLVE-BLOCK-END
