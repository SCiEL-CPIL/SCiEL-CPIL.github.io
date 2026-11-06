import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10 s safety margin before 150 s kill

    def elapsed():
        return time.time() - start_time

    # 1. Graph generation (exact match to verifier)
    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    instances = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = get_instance(*spec)
        n = spec[0]
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute F(e): vertices that cannot co-occur with e in an induced matching
        edge_F = []
        for e in edges:
            u, v = e
            F = set(adj[u])
            F.update(adj[v])
            F.add(u)
            F.add(v)
            edge_F.append(frozenset(F))

        m = len(edges)
        # Pairwise conflict matrix: conflict[i][j] = True if edges i,j cannot share a colour
        conflict = [[False] * m for _ in range(m)]
        for i in range(m):
            Fi = edge_F[i]
            for j in range(i + 1, m):
                if not Fi.isdisjoint(edge_F[j]):
                    conflict[i][j] = True
                    conflict[j][i] = True

        instances.append({
            'n': n,
            'edges': edges,
            'edge_F': edge_F,
            'adj': adj,
            'conflict': conflict,
            'm': m,
        })

    results = []

    def solve_instance(inst):
        n = inst['n']
        edges = inst['edges']
        edge_F = inst['edge_F']
        conflict = inst['conflict']
        m = inst['m']

        # Edge index lookup
        edge_idx = {e: i for i, e in enumerate(edges)}

        # State: colour[i] = colour class index for edge i, or -1
        # class_edges[c] = list of edge indices in colour class c
        # class_verts[c] = frozenset of vertices in colour class c

        def greedy(order):
            """First-fit greedy colouring given an edge ordering (list of edge indices)."""
            colour = [-1] * m
            class_verts = []   # list of sets
            class_edges = []   # list of lists
            for i in order:
                Fi = edge_F[i]
                placed = False
                for c in range(len(class_verts)):
                    if class_verts[c].isdisjoint(Fi):
                        colour[i] = c
                        class_verts[c].add(edges[i][0])
                        class_verts[c].add(edges[i][1])
                        class_edges[c].append(i)
                        placed = True
                        break
                if not placed:
                    c = len(class_verts)
                    colour[i] = c
                    class_verts.append({edges[i][0], edges[i][1]})
                    class_edges.append([i])
            return colour, class_verts, class_edges

        def compact(colour, class_verts, class_edges):
            """Remove empty classes and reindex."""
            new_cv = []
            new_ce = []
            old_to_new = {}
            for c in range(len(class_edges)):
                if class_edges[c]:
                    old_to_new[c] = len(new_cv)
                    new_cv.append(class_verts[c])
                    new_ce.append(class_edges[c])
            colour2 = [old_to_new.get(c, -1) for c in colour]
            return colour2, new_cv, new_ce

        def try_relocate(colour, class_verts, class_edges, i, from_c):
            """Try to move edge i from class from_c to a lower-index class. Returns True if moved."""
            Fi = edge_F[i]
            for c2 in range(from_c):
                if class_verts[c2].isdisjoint(Fi):
                    # Move
                    class_edges[from_c].remove(i)
                    class_verts[from_c].discard(edges[i][0])
                    class_verts[from_c].discard(edges[i][1])
                    class_edges[c2].append(i)
                    class_verts[c2].add(edges[i][0])
                    class_verts[c2].add(edges[i][1])
                    colour[i] = c2
                    return True
            return False

        def try_swap(colour, class_verts, class_edges, i1, c1, i2, c2):
            """Try to swap edge i1 (in c1) with edge i2 (in c2). Returns True if swapped."""
            Fi1 = edge_F[i1]
            Fi2 = edge_F[i2]
            v1 = {edges[i1][0], edges[i1][1]}
            v2 = {edges[i2][0], edges[i2][1]}
            # Check i1 can go to c2 without i2
            if not (class_verts[c2] - v2).isdisjoint(Fi1):
                return False
            # Check i2 can go to c1 without i1
            if not (class_verts[c1] - v1).isdisjoint(Fi2):
                return False
            # Perform swap
            class_edges[c1].remove(i1)
            class_verts[c1] -= v1
            class_edges[c2].remove(i2)
            class_verts[c2] -= v2
            class_edges[c1].append(i2)
            class_verts[c1] |= v2
            class_edges[c2].append(i1)
            class_verts[c2] |= v1
            colour[i1] = c2
            colour[i2] = c1
            return True

        def local_search(colour, class_verts, class_edges):
            """Iterate relocations and swaps until no improvement or deadline."""
            while elapsed() < deadline - 5:
                improved = False

                # Relocate edges from high-index classes to lower classes
                for c1 in range(len(class_edges) - 1, -1, -1):
                    if not class_edges[c1]:
                        continue
                    for i in list(class_edges[c1]):
                        if colour[i] != c1:
                            continue
                        if try_relocate(colour, class_verts, class_edges, i, c1):
                            improved = True
                            break
                    if improved:
                        break

                if improved:
                    colour, class_verts, class_edges = compact(colour, class_verts, class_edges)
                    continue

                # Try swaps: focus on singleton high classes
                if elapsed() < deadline - 10:
                    for c1 in range(len(class_edges) - 1, 0, -1):
                        if not class_edges[c1]:
                            continue
                        if len(class_edges[c1]) == 1:
                            i1 = class_edges[c1][0]
                            for c2 in range(c1):
                                if not class_edges[c2]:
                                    continue
                                for i2 in list(class_edges[c2]):
                                    if try_swap(colour, class_verts, class_edges, i1, c1, i2, c2):
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break

                    if improved:
                        colour, class_verts, class_edges = compact(colour, class_verts, class_edges)
                        continue

                # No improvement; try a broader swap pass (any edge in high class)
                if elapsed() < deadline - 15:
                    for c1 in range(len(class_edges) - 1, 0, -1):
                        if not class_edges[c1] or len(class_edges[c1]) < 2:
                            continue
                        for i1 in list(class_edges[c1]):
                            if colour[i1] != c1:
                                continue
                            for c2 in range(c1):
                                if not class_edges[c2]:
                                    continue
                                for i2 in list(class_edges[c2]):
                                    if try_swap(colour, class_verts, class_edges, i1, c1, i2, c2):
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break

                    if improved:
                        colour, class_verts, class_edges = compact(colour, class_verts, class_edges)
                        continue

                # No more improvements possible
                break

            colour, class_verts, class_edges = compact(colour, class_verts, class_edges)
            return colour, len(class_edges)

        # Generate different edge orderings
        def make_orderings():
            orderings = []
            # Degree-sum descending
            deg_sum = [len(inst['adj'][edges[i][0]]) + len(inst['adj'][edges[i][1]]) for i in range(m)]
            orderings.append(sorted(range(m), key=lambda i: -deg_sum[i]))
            # Degree-sum ascending
            orderings.append(sorted(range(m), key=lambda i: deg_sum[i]))
            # Random orderings
            rng = random.Random(42)
            for _ in range(12):
                order = list(range(m))
                rng.shuffle(order)
                orderings.append(order)
            # Mixed: high-degree first, then random
            rng2 = random.Random(123)
            for _ in range(8):
                order = list(range(m))
                rng2.shuffle(order)
                order.sort(key=lambda i: -deg_sum[i] * 0.7 - rng2.random() * 0.3)
                orderings.append(order)
            return orderings

        orderings = make_orderings()

        best_colour = None
        best_count = m  # worst case: each edge its own colour
        best_cv = None
        best_ce = None

        for order in orderings:
            if elapsed() > deadline - 5:
                break
            colour, cv, ce = greedy(order)
            count = len(ce)
            if count < best_count:
                # Run local search on this solution
                colour2, count2 = local_search(colour, cv, ce)
                if count2 < best_count:
                    best_count = count2
                    best_colour = colour2
                    # Rebuild cv, ce from best_colour for final output
                    best_cv, best_ce = compact(best_colour, cv, ce)

        # If no local search was run (deadline), use the best greedy
        if best_colour is None:
            colour, cv, ce = greedy(orderings[0])
            best_colour = colour
            best_count = len(ce)

        # Final compact to get clean indices
        colour_final, cv_final, ce_final = compact(best_colour, best_cv, best_ce)

        # Build result in original sorted edge order
        res = [0] * m
        for i in range(m):
            res[i] = colour_final[i]
        return res

    for inst in instances:
        if elapsed() > deadline - 3:
            # Emergency: return trivial (each edge unique colour) to be safe
            results.append(list(range(inst['m'])))
            continue
        results.append(solve_instance(inst))

    return results
