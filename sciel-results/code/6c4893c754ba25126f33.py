"""Model-and-optimize: Min-Conflicts local search for list coloring on bipartite graphs."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 15.0

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        """
        Solves the list coloring problem using Min-Conflicts local search.
        This is a stochastic local search method that works well for constraint satisfaction
        problems like graph coloring, especially on sparse graphs.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Build a set of edges for quick lookup
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))

        rng = Random(12345)

        # Multiple restarts of Min-Conflicts
        max_restarts = 200
        max_steps_per_restart = 50000

        for restart in range(max_restarts):
            if deadline():
                break

            # Initialize: assign each vertex a random color from its list
            colour = [rng.choice(lists[v]) for v in range(n)]

            # Precompute conflict counts
            # conflict_count[v] = number of neighbors of v that have the same color as v
            conflict_count = [0] * n
            for u in range(n):
                for v in adj[u]:
                    if colour[u] == colour[v]:
                        conflict_count[u] += 1

            # Also check list violations
            list_violation = [colour[v] not in lists[v] for v in range(n)]

            def total_conflicts():
                # Count each conflicting edge once (both endpoints see it, so divide by 2)
                # But also count list violations
                tc = sum(conflict_count) // 2
                tc += sum(list_violation)
                return tc

            # Check if solution is valid
            if total_conflicts() == 0:
                return [int(c) for c in colour]

            # Min-Conflicts: repeatedly pick a conflicted vertex and reassign
            for step in range(max_steps_per_restart):
                if deadline():
                    break

                # Find all conflicted vertices
                # A vertex is conflicted if it has a list violation or a neighbor with same color
                conflicted = []
                for v in range(n):
                    if list_violation[v]:
                        conflicted.append(v)
                    elif conflict_count[v] > 0:
                        conflicted.append(v)

                if not conflicted:
                    # Valid solution found
                    return [int(c) for c in colour]

                # Pick a random conflicted vertex
                v = rng.choice(conflicted)

                # Find the color from lists[v] that minimizes total conflicts
                best_colour = None
                best_conflict_delta = float('inf')

                current_colour = colour[v]

                # Try each color in lists[v]
                for c in lists[v]:
                    if c == current_colour:
                        continue

                    # Calculate change in conflict count
                    # When we change v's color from current_colour to c:
                    # - For each neighbor u:
                    #   - If colour[u] == current_colour, conflict_count[u] decreases by 1 (good)
                    #   - If colour[u] == c, conflict_count[u] increases by 1 (bad)
                    #   - conflict_count[v] changes based on neighbors

                    delta = 0

                    # Effect on v's own conflict count:
                    # v currently conflicts with neighbors having current_colour
                    # v will conflict with neighbors having c
                    for u in adj[v]:
                        if colour[u] == current_colour:
                            # This neighbor no longer conflicts with v
                            delta -= 1
                        if colour[u] == c:
                            # This neighbor now conflicts with v
                            delta += 1

                    # Effect on neighbors' conflict counts:
                    for u in adj[v]:
                        if colour[u] == current_colour:
                            # u's conflict count decreases by 1
                            delta -= 1
                        if colour[u] == c:
                            # u's conflict count increases by 1
                            delta += 1

                    # List violation change:
                    if list_violation[v]:
                        # Currently violating list, after change might not be
                        if c in lists[v]:
                            delta -= 1  # Remove one list violation

                    if delta < best_conflict_delta:
                        best_conflict_delta = delta
                        best_colour = c

                # If no better color found, just pick a random one from the list
                if best_colour is None:
                    best_colour = rng.choice(lists[v])

                # Apply the change
                old_colour = colour[v]
                colour[v] = best_colour

                # Update conflict counts
                # v's conflict count changes
                new_v_conflicts = 0
                for u in adj[v]:
                    if colour[u] == best_colour:
                        new_v_conflicts += 1
                old_v_conflicts = conflict_count[v]
                conflict_count[v] = new_v_conflicts

                # Neighbors' conflict counts change
                for u in adj[v]:
                    if colour[u] == old_colour:
                        conflict_count[u] -= 1
                    if colour[u] == best_colour:
                        conflict_count[u] += 1

                # Update list violation
                list_violation[v] = (best_colour not in lists[v])

                # Check if solution is valid
                if total_conflicts() == 0:
                    return [int(c) for c in colour]

        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            n, edges, lists = instance(splits, seed)
            rng_fallback = Random(777)
            out.append([int(rng_fallback.choice(lists[v])) for v in range(n)])
            continue

        n, edges, lists = instance(splits, seed)
        colour = solve_instance(n, edges, lists)

        if colour is None:
            # Fallback to random if solver fails
            rng_fallback = Random(777)
            colour = [int(rng_fallback.choice(lists[v])) for v in range(n)]

        out.append(colour)

    return out
# EVOLVE-BLOCK-END
