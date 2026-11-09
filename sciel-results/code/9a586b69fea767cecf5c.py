import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict matrix as bitsets for speed
        # conflict[i] is a set of j that conflict with i
        conflict = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict[i].add(j)
                        conflict[j].add(i)

        def greedy_coloring(order, rng):
            """Greedy coloring in a given edge order."""
            colors = [0] * m
            # pos maps edge index -> position in order
            pos = [0] * m
            for idx, e in enumerate(order):
                pos[e] = idx
            # Process in order
            color_of_edge = [None] * m
            for e in order:
                taken = set()
                for j in conflict[e]:
                    if color_of_edge[j] is not None:
                        taken.add(color_of_edge[j])
                c = 0
                while c in taken:
                    c += 1
                color_of_edge[e] = c
            return color_of_edge

        def count_conflicts(colors):
            """Count number of conflicting pairs with same color."""
            cnt = 0
            for i in range(m):
                for j in conflict[i]:
                    if j > i and colors[i] == colors[j]:
                        cnt += 1
            return cnt

        def num_colors(colors):
            return max(colors) + 1 if m > 0 else 0

        def try_eliminate_color(colors, target_color):
            """Try to reassign all edges with target_color to other colors to eliminate it."""
            edges_in_color = [i for i in range(m) if colors[i] == target_color]
            if not edges_in_color:
                return True, colors
            # For each edge in this color, find available colors
            # Try to assign them to existing lower colors
            new_colors = colors[:]
            # Sort by number of conflicts with same-color edges (hardest first)
            edges_in_color.sort(key=lambda i: len(conflict[i]) - 1, reverse=True)
            for i in edges_in_color:
                # Find a valid color different from target_color
                taken = set()
                for j in conflict[i]:
                    if j != i and new_colors[j] != target_color:
                        taken.add(new_colors[j])
                # Try lowest available color
                c = 0
                while c in taken:
                    c += 1
                new_colors[i] = c
            # Verify no conflicts
            if count_conflicts(new_colors) == 0:
                return True, new_colors
            return False, colors

        def swap_local_search(colors, rng, time_budget):
            """Local search using swaps of colors between pairs of edges."""
            current_conflicts = count_conflicts(colors)
            current_nc = num_colors(colors)
            best_colors = colors[:]
            best_score = current_nc  # We track number of colors; conflicts should be 0
            start = time.time()

            while time.time() - start < time_budget and current_conflicts == 0:
                # Try to reduce number of colors
                nc = num_colors(best_colors)
                if nc == 0:
                    break
                # Try to eliminate the highest color
                eliminated, new_colors = try_eliminate_color(best_colors, nc - 1)
                if eliminated:
                    best_colors = new_colors
                    continue
                # Try swaps: pick two edges, swap their colors
                i = rng.randrange(m)
                j = rng.randrange(m)
                if i == j:
                    continue
                ci = best_colors[i]
                cj = best_colors[j]
                if ci == cj:
                    continue
                # Check if swap is valid
                # Edge i moves to cj: check no conflict with existing cj edges
                valid = True
                for k in conflict[i]:
                    if best_colors[k] == cj:
                        valid = False
                        break
                if valid:
                    for k in conflict[j]:
                        if best_colors[k] == ci:
                            valid = False
                            break
                if valid:
                    best_colors[i] = cj
                    best_colors[j] = ci
                    # Check if this reduces color count (one color might become empty)
                    nc_new = num_colors(best_colors)
                    if nc_new < num_colors(best_colors):
                        pass  # Good, keep it
                    # Always keep valid swaps
                else:
                    # Try random perturbation: move one edge to a random valid color
                    i = rng.randrange(m)
                    old_c = best_colors[i]
                    # Find all valid colors for edge i
                    taken = set()
                    for k in conflict[i]:
                        taken.add(best_colors[k])
                    # Try a color not in taken
                    c = 0
                    while c in taken:
                        c += 1
                    if c != old_c:
                        best_colors[i] = c
            return best_colors

        def multi_start_search(time_budget):
            """Try multiple initializations and local searches."""
            best_colors = None
            best_nc = float('inf')
            start = time.time()

            # Generate multiple orderings
            orderings = []
            # Natural order
            orderings.append(list(range(m)))
            # Reverse
            orderings.append(list(range(m - 1, -1, -1)))
            # By degree (high degree first)
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            orderings.append(sorted(range(m), key=lambda i: deg[edges[i][0]] + deg[edges[i][1]], reverse=True))
            # Random orderings
            for _ in range(3):
                rng_ord = random.Random(seed + 999 + len(orderings))
                ord_list = list(range(m))
                rng_ord.shuffle(ord_list)
                orderings.append(ord_list)

            for ord_idx, order in enumerate(orderings):
                if time.time() - start > time_budget:
                    break
                rng = random.Random(seed + 777 + ord_idx)
                colors = greedy_coloring(order, rng)
                nc = num_colors(colors)
                # Run local search with remaining time
                remaining = time_budget - (time.time() - start)
                if remaining > 0.5:
                    colors = swap_local_search(colors, rng, remaining * 0.7)
                    nc = num_colors(colors)
                if nc < best_nc:
                    best_nc = nc
                    best_colors = colors[:]

            return best_colors if best_colors is not None else greedy_coloring(list(range(m)), random.Random(seed))

        # Main search
        result = multi_start_search(time_limit)

        # Final: renumber colors to 0..k-1
        color_map = {}
        next_c = 0
        final_colors = []
        for c in result:
            if c not in color_map:
                color_map[c] = next_c
                next_c += 1
            final_colors.append(color_map[c])

        return final_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: simple greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = set(e)
                for x in e:
                    near.update(adj[x])
                taken = set()
                for f, c in colors.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue

        # Allocate time: give more to harder instances
        time_limit = max(1.0, remaining - 2.0)
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)

    return results
