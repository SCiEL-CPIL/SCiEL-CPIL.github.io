"""Strong edge coloring via greedy set-cover over candidate induced matchings."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        m = len(edges)
        vert_to_edges = [[] for _ in range(n_max)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)

        conflict = [set() for _ in range(m)]

        for u in range(n_max):
            e_list = vert_to_edges[u]
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    ei = e_list[i]
                    ej = e_list[j]
                    conflict[ei].add(ej)
                    conflict[ej].add(ei)

        for i, (u, v) in enumerate(edges):
            neighbors_u = adj.get(u, set())
            neighbors_v = adj.get(v, set())
            for j in range(i + 1, m):
                if j in conflict[i]:
                    continue
                w, x = edges[j]
                if w in neighbors_u or x in neighbors_u or w in neighbors_v or x in neighbors_v:
                    conflict[i].add(j)
                    conflict[j].add(i)

        return conflict

    def greedy_coloring(conflict, m):
        assign = [0] * m
        for i in range(m):
            used = set(assign[j] for j in conflict[i])
            c = 0
            while c in used:
                c += 1
            assign[i] = c
        return assign

    def is_induced_matching(edge_set, conflict):
        # Check that no two edges in edge_set conflict
        for i, j in zip(sorted(edge_set), sorted(edge_set)):
            pass
        edges_list = sorted(edge_set)
        for idx_a in range(len(edges_list)):
            for idx_b in range(idx_a + 1, len(edges_list)):
                if edges_list[idx_b] in conflict[edges_list[idx_a]]:
                    return False
        return True

    def enumerate_matchings(edges, conflict, m, max_matchings=2000, rng=None):
        """Generate a pool of candidate induced matchings using greedy + randomization."""
        matchings = set()
        if rng is None:
            rng = random.Random(42)

        # Deterministic greedy: repeatedly pick the edge with fewest conflicts among uncovered
        # We'll do multiple passes with different orderings.

        # Order 1: by index
        # Order 2: by degree in conflict graph (ascending)
        # Order 3: random shuffles

        orderings = []
        orderings.append(list(range(m)))
        # Sort by conflict degree ascending
        orderings.append(sorted(range(m), key=lambda i: len(conflict[i])))
        orderings.append(sorted(range(m), key=lambda i: -len(conflict[i])))
        # Several random orderings
        for _ in range(20):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)

        for order in orderings:
            covered = [False] * m
            current_matching = []
            for i in order:
                if covered[i]:
                    continue
                # Check if i can be added: no conflict with any edge already in matching
                can_add = True
                for j in current_matching:
                    if j in conflict[i]:
                        can_add = False
                        break
                if can_add:
                    current_matching.append(i)
                    covered[i] = True
                    # Mark all conflicting edges as covered (they can't be in this matching)
                    for j in conflict[i]:
                        covered[j] = True

            if current_matching:
                matchings.add(tuple(current_matching))

            if len(matchings) >= max_matchings:
                break

        # Also try: for each edge as forced first, greedy complete
        for start in range(min(m, 50)):
            covered = [False] * m
            current_matching = [start]
            covered[start] = True
            for j in conflict[start]:
                covered[j] = True

            # Greedy complete with random tie-breaking
            for i in range(m):
                if covered[i] or i == start:
                    continue
                can_add = True
                for j in current_matching:
                    if j in conflict[i]:
                        can_add = False
                        break
                if can_add:
                    current_matching.append(i)
                    covered[i] = True
                    for j in conflict[i]:
                        covered[j] = True

            if current_matching:
                matchings.add(tuple(current_matching))

            if len(matchings) >= max_matchings:
                break

        return list(matchings)

    def solve_instance(n, d, seed, time_limit=30):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        global n_max
        n_max = n

        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        conflict = build_conflict(edges, adj)

        # Fallback greedy coloring
        fallback = greedy_coloring(conflict, m)
        fallback_count = len(set(fallback))

        # Enumerate candidate induced matchings
        rng = random.Random(seed + 12345)
        candidate_matchings = enumerate_matchings(edges, conflict, m, max_matchings=3000, rng=rng)

        if not candidate_matchings:
            return fallback

        # Greedy set cover: select matchings to cover all edges
        # Score for a matching = (new_edges_covered) / (1 + overlap_penalty)
        # where overlap_penalty encourages diversity

        start_time = time.time()
        uncovered = set(range(m))
        selected_matchings = []
        edge_to_color = [-1] * m

        # Track how many times each edge has been "covered" by selected matchings
        # to compute redundancy penalty
        edge_coverage_count = [0] * m

        while uncovered and time.time() - start_time < time_limit:
            best_score = -1
            best_matching = None
            best_new_edges = None

            for matching in candidate_matchings:
                # Count new edges this matching covers
                new_edges = []
                for e in matching:
                    if e in uncovered:
                        new_edges.append(e)

                if not new_edges:
                    continue

                # Lagrangian-style score: reward new coverage, penalize redundancy
                # and also penalize if matching is large (wastes potential for future)
                new_count = len(new_edges)
                # Penalty: sum of coverage counts of edges in this matching (redundancy)
                redundancy = sum(edge_coverage_count[e] for e in matching)
                # Score: new coverage / (1 + redundancy_factor)
                score = new_count / (1.0 + 0.1 * redundancy)

                if score > best_score:
                    best_score = score
                    best_matching = matching
                    best_new_edges = new_edges

            if best_matching is None:
                # No matching covers any new edge; break
                break

            # Assign color
            color = len(selected_matchings)
            for e in best_matching:
                if edge_to_color[e] == -1:
                    edge_to_color[e] = color
                    uncovered.discard(e)
                # Update coverage count for all edges in this matching
                edge_coverage_count[e] += 1

            selected_matchings.append(best_matching)

            # Check time
            if time.time() - start_time >= time_limit:
                break

        # If not all edges are covered, assign remaining edges greedily
        if uncovered:
            # Use greedy coloring for the remaining edges, continuing from existing colors
            # Actually, simpler: just do a full greedy coloring as fallback for uncovered
            # But we want to keep the good matchings we found.
            # Let's assign uncovered edges one by one with the minimum available color.
            for e in sorted(uncovered):
                used_colors = set()
                for j in conflict[e]:
                    if edge_to_color[j] != -1:
                        used_colors.add(edge_to_color[j])
                c = 0
                while c in used_colors:
                    c += 1
                edge_to_color[e] = c
                uncovered.discard(e)

        # Verify and return
        assign = edge_to_color[:]
        # Final verification
        valid = True
        for i in range(m):
            c = assign[i]
            for j in conflict[i]:
                if assign[j] == c:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            assign = fallback

        return assign

    out = []
    deadline = time.time() + 115

    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 10:
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            global n_max
            n_max = n
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            conflict = build_conflict(edges, adj)
            assign = greedy_coloring(conflict, m)
            out.append(assign)
        else:
            t_limit = min(remaining - 5, 35)
            out.append(solve_instance(*spec, time_limit=t_limit))

    return out
# EVOLVE-BLOCK-END
