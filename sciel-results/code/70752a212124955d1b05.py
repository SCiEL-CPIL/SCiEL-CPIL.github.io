"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

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

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_conflict(e1, e2, adj):
        """Check if two edges conflict (share vertex or are joined by an edge)."""
        u1, v1 = e1
        u2, v2 = e2
        # Share a vertex
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return True
        # Joined by an edge: any endpoint of e1 adjacent to any endpoint of e2
        if v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1] or u2 in adj[u1]:
            return True
        return False

    def is_induced_matching(edge_list, adj):
        """Check if a list of edges forms an induced matching."""
        for i in range(len(edge_list)):
            for j in range(i + 1, len(edge_list)):
                if edge_conflict(edge_list[i], edge_list[j], adj):
                    return False
        return True

    def find_maximal_induced_matchings(edges, adj, rng, max_find=2000):
        """Find many maximal induced matchings using randomized greedy construction."""
        matchings = []
        seen = set()
        n_edges = len(edges)
        if n_edges == 0:
            return matchings

        for attempt in range(max_find):
            if time.time() > deadline - 15:
                break
            # Randomized greedy: shuffle edges, greedily add if compatible
            shuffled = list(edges)
            rng.shuffle(shuffled)
            current = []
            for e in shuffled:
                # Check compatibility with all edges in current
                ok = True
                for f in current:
                    if edge_conflict(e, f, adj):
                        ok = False
                        break
                if ok:
                    current.append(e)
            # Verify it's an induced matching (should be by construction)
            if is_induced_matching(current, adj):
                key = frozenset(current)
                if key not in seen:
                    seen.add(key)
                    matchings.append(current)
            if time.time() > deadline - 15:
                break
        return matchings

    def greedy_set_cover(edges, adj, rng):
        """
        Set cover approach:
        - Universe: all edges
        - Sets: maximal induced matchings
        - Greedily pick the set covering most uncovered edges
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []

        edge_to_idx = {e: i for i, e in enumerate(edges)}

        # Find a pool of induced matchings
        matchings = find_maximal_induced_matchings(edges, adj, rng, max_find=1500)

        if not matchings:
            # Fallback: each edge is its own matching
            matchings = [[e] for e in edges]

        # Represent each matching as a frozenset of edge indices
        matching_sets = []
        for m in matchings:
            s = frozenset(edge_to_idx[e] for e in m)
            matching_sets.append(s)

        # Greedy set cover
        uncovered = set(range(n_edges))
        assignment = [None] * n_edges
        color_idx = 0
        used_sets = set()

        while uncovered:
            # Find the set that covers the most uncovered edges
            best_set = None
            best_count = -1
            best_idx = -1
            for i, s in enumerate(matching_sets):
                if i in used_sets:
                    continue
                count = len(s & uncovered)
                if count > best_count:
                    best_count = count
                    best_set = s
                    best_idx = i
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break
            if best_set is None or best_count <= 0:
                # Fallback: assign remaining edges one per color
                for e_idx in sorted(uncovered):
                    assignment[e_idx] = color_idx
                    color_idx += 1
                break

            used_sets.add(best_idx)
            for e_idx in best_set:
                if e_idx in uncovered:
                    assignment[e_idx] = color_idx
                    uncovered.discard(e_idx)
            color_idx += 1

        # Compact colors
        if all(a is not None for a in assignment):
            # Remap to 0..k-1
            color_map = {}
            next_c = 0
            for a in assignment:
                if a not in color_map:
                    color_map[a] = next_c
                    next_c += 1
            return [color_map[a] for a in assignment]
        else:
            # Should not happen, but fallback
            return [0] * n_edges

    def greedy_color(edges, adj, n, rng):
        """Greedy coloring: assign each edge the smallest available color."""
        color = {}
        color_edges = {}
        for e in edges:
            u, v = e
            near = {u, v}
            for x in (u, v):
                near.update(adj[x])
            taken = set()
            for c, elist in color_edges.items():
                for f in elist:
                    fu, fv = f
                    if fu in near or fv in near:
                        taken.add(c)
                        break
            c = 0
            while c in taken:
                c += 1
            color[e] = c
            color_edges.setdefault(c, []).append(e)
        return color, color_edges

    def try_merge(color_edges, adj, rng):
        """Try to merge two color classes."""
        colors = list(color_edges.keys())
        if len(colors) < 2:
            return None
        for _ in range(100):
            c1, c2 = rng.sample(colors, 2)
            compatible = True
            for e1 in color_edges[c1]:
                for e2 in color_edges[c2]:
                    if edge_conflict(e1, e2, adj):
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                new_edges = color_edges[c1] + color_edges[c2]
                new_color_edges = {c: v for c, v in color_edges.items() if c not in (c1, c2)}
                new_color_edges[c1] = new_edges
                return new_color_edges, True
        return None

    def local_search(color, color_edges, adj, n, rng, max_iter=2000):
        """Local search: try to reduce number of colors by moving edges."""
        best_color = dict(color)
        best_color_edges = {c: list(v) for c, v in color_edges.items()}
        best_num_colors = len(best_color_edges)

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            colors = list(best_color_edges.keys())
            if len(colors) < 2:
                break
            src_c = rng.choice(colors)
            if not best_color_edges[src_c]:
                continue
            e = rng.choice(best_color_edges[src_c])
            target_colors = [c for c in colors if c != src_c and best_color_edges[c]]
            if not target_colors:
                continue
            dst_c = rng.choice(target_colors)
            u, v = e
            near = {u, v}
            for x in (u, v):
                near.update(adj[x])
            can_add = True
            for f in best_color_edges[dst_c]:
                fu, fv = f
                if fu in near or fv in near:
                    can_add = False
                    break
            if can_add:
                best_color_edges[src_c].remove(e)
                best_color_edges[dst_c].append(e)
                best_color[e] = dst_c
                if not best_color_edges[src_c]:
                    del best_color_edges[src_c]
                if len(best_color_edges) < best_num_colors:
                    best_num_colors = len(best_color_edges)

        return best_color, best_color_edges, best_num_colors

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        rng = Random(seed * 1000 + 42)

        best_result = None
        best_score = float('inf')

        # Strategy 1: Set cover with induced matchings
        sc_colors = greedy_set_cover(edges, adj, rng)
        if sc_colors is not None:
            sc_score = max(sc_colors) + 1 if sc_colors else 0
            if sc_score < best_score:
                best_score = sc_score
                best_result = sc_colors

        # Strategy 2: Greedy + merge + local search (multiple restarts)
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            shuffled = list(edges)
            rng.shuffle(shuffled)

            color, color_edges = greedy_color(shuffled, adj, n, rng)

            for _ in range(30):
                if time.time() > deadline - 10:
                    break
                result = try_merge(color_edges, adj, rng)
                if result is None:
                    break
                color_edges, _ = result
                color = {}
                for c, elist in color_edges.items():
                    for e in elist:
                        color[e] = c

            color, color_edges, num_colors = local_search(color, color_edges, adj, n, rng, max_iter=1500)

            if num_colors < best_score:
                best_score = num_colors
                best_result = [color[e] for e in edges]

        return best_result

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        result = solve_instance(*spec)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
