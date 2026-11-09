"""Strong edge colouring with local search for colour reduction."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, deadline):
        if not edges:
            return []
        n = max(max(u, v) for u, v in edges) + 1
        m = len(edges)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Edge index
        edge_idx = {}
        for i, e in enumerate(edges):
            edge_idx[e] = i

        # Precompute conflict sets: for each edge i, the set of edge indices that conflict with it
        # Two edges conflict if they share a vertex OR are joined by an edge
        conflicts = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Share a vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
                # Joined by an edge?
                if v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1] or u2 in adj[u1]:
                    conflicts[i].add(j)
                    conflicts[j].add(i)

        # Initial greedy colouring: order edges by degree (conflict count) descending
        order = sorted(range(m), key=lambda i: -len(conflicts[i]))
        colour = [None] * m
        used_colours = 0
        for i in order:
            # Find first colour not used by any conflicting edge
            taken = set()
            for j in conflicts[i]:
                if colour[j] is not None:
                    taken.add(colour[j])
            c = 0
            while c in taken:
                c += 1
            colour[i] = c
            if c + 1 > used_colours:
                used_colours = c + 1

        # Local search: try to reduce colour count
        # For colour c (from high to low), try to eliminate it by moving its edges to other colours
        for c in range(used_colours - 1, 0, -1):
            if time.time() > deadline:
                break
            # Collect edges with colour c
            edges_of_c = [i for i in range(m) if colour[i] == c]
            if not edges_of_c:
                continue

            # Try to move each edge to an existing lower colour
            success = True
            for i in edges_of_c:
                if time.time() > deadline:
                    success = False
                    break
                # Find a colour c' != c such that no conflict of i has colour c'
                # (and after we move i, no conflict of i still has colour c)
                found = False
                for c2 in range(c):
                    ok = True
                    for j in conflicts[i]:
                        if colour[j] == c2:
                            ok = False
                            break
                    if ok:
                        colour[i] = c2
                        found = True
                        break
                if not found:
                    success = False
                    break

            if success:
                used_colours -= 1
            else:
                # Try swap-based approach: for each edge i in colour c,
                # try to swap with an edge k in colour c' to free up space
                for i in edges_of_c:
                    if time.time() > deadline:
                        break
                    # For each target colour c2 < c
                    for c2 in range(c):
                        # Find an edge k with colour c2 that conflicts with i
                        # If we move k to colour c (after i moves to c2), check validity
                        # Actually simpler: try to find c2 where only one conflict blocks
                        blockers = [j for j in conflicts[i] if colour[j] == c2]
                        if len(blockers) == 1:
                            k = blockers[0]
                            # Move k to colour c: check no other conflict of k has colour c
                            # (excluding i, since i will move to c2)
                            ok = True
                            for j in conflicts[k]:
                                if j == i:
                                    continue
                                if colour[j] == c:
                                    ok = False
                                    break
                            if ok:
                                colour[i] = c2
                                colour[k] = c
                                # Now i is in c2, k is in c
                                # We haven't eliminated c, but we've rearranged
                                # Continue trying to eliminate c
                                break

                # After swaps, try again to eliminate c
                edges_of_c = [i for i in range(m) if colour[i] == c]
                if edges_of_c:
                    success2 = True
                    for i in edges_of_c:
                        if time.time() > deadline:
                            success2 = False
                            break
                        found = False
                        for c2 in range(c):
                            ok = True
                            for j in conflicts[i]:
                                if colour[j] == c2:
                                    ok = False
                                    break
                            if ok:
                                colour[i] = c2
                                found = True
                                break
                        if not found:
                            success2 = False
                            break
                    if success2:
                        used_colours -= 1

        # Recompact colours
        colour_map = {}
        new_colour = [0] * m
        next_c = 0
        for i in range(m):
            if colour[i] not in colour_map:
                colour_map[colour[i]] = next_c
                next_c += 1
            new_colour[i] = colour_map[colour[i]]
        return new_colour

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        cols = solve_instance(edges, DEADLINE)
        results.append(cols)

    return results
# EVOLVE-BLOCK-END
