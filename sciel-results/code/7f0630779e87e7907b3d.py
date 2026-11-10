"""Clique partition of split graphs: independent-set-based C-subset pooling."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def cover_remaining_C_edges(remaining_C_edges, deadline):
        """Cover a set of C-edges with as few cliques as possible using a
        greedy max-common-neighbor clique extraction."""
        cliques = []
        while remaining_C_edges:
            if time.time() > deadline:
                break
            # Build adjacency of remaining graph
            remaining_adj = defaultdict(set)
            for u, v in remaining_C_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)

            # Pick the edge with the most common neighbors -> largest clique
            max_common = -1
            best_edge = None
            for u, v in remaining_C_edges:
                common = len(remaining_adj[u] & remaining_adj[v])
                if common > max_common:
                    max_common = common
                    best_edge = (u, v)

            if best_edge is None:
                u, v = next(iter(remaining_C_edges))
            else:
                u, v = best_edge

            # Form the clique as {u, v} plus all common neighbors
            S = {u, v} | (remaining_adj[u] & remaining_adj[v])
            S_list = sorted(S)
            cliques.append(S_list)

            # Remove all edges inside this clique
            for a_idx in range(len(S_list)):
                for b_idx in range(a_idx + 1, len(S_list)):
                    remaining_C_edges.discard((S_list[a_idx], S_list[b_idx]))

        return cliques

    def find_max_independent_set(C_edges, C_vertices, deadline):
        """Find a large independent set in the C-edge graph (i.e., a set of C-vertices
        with no edges between them). Uses greedy with random restarts."""
        C_set = set(C_vertices)
        best_is = set()
        
        # Multiple random orderings for greedy
        rng = random.Random(42)
        for trial in range(20):
            if time.time() > deadline:
                break
            order = list(C_set)
            rng.shuffle(order)
            
            IS = set()
            for v in order:
                # v can be added if no neighbor in IS
                can_add = True
                for u in IS:
                    if (min(u, v), max(u, v)) in C_edges:
                        can_add = False
                        break
                if can_add:
                    IS.add(v)
            
            if len(IS) > len(best_is):
                best_is = set(IS)
        
        return best_is

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build cross-neighborhoods: N[i] = C-vertices adjacent to I-vertex i
        N = {i: set() for i in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)

        # All C-edges
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))

        I_vertices_list = [i for i in range(k, n) if N[i]]
        
        # Find a large independent set in C (vertices with no C-edges between them)
        # This set can be shared across multiple I-vertices
        C_vertices = list(range(k))
        IS_C = find_max_independent_set(C_edges, C_vertices, deadline)
        
        # Build candidate orderings: descending, ascending, and random shuffles
        strategies = []
        strategies.append(sorted(I_vertices_list, key=lambda i: len(N[i]), reverse=True))
        strategies.append(sorted(I_vertices_list, key=lambda i: len(N[i])))

        rng = random.Random(seed)
        n_random = 40
        for _ in range(n_random):
            strategies.append(rng.sample(I_vertices_list, len(I_vertices_list)))

        best_cliques = None
        best_count = float('inf')

        for order in strategies:
            if time.time() > deadline - 5:
                break

            covered_C_edges = set()
            cliques = []

            for i in order:
                if time.time() > deadline - 5:
                    break
                if not N[i]:
                    continue

                remaining_N = set(N[i])
                while remaining_N:
                    if time.time() > deadline - 5:
                        break

                    # Greedily build largest S of remaining_N with no internal
                    # C-edge already covered
                    S = set()
                    for u in sorted(remaining_N):
                        can_add = True
                        for v in S:
                            e = (min(u, v), max(u, v))
                            if e in covered_C_edges:
                                can_add = False
                                break
                        if can_add:
                            S.add(u)

                    if not S:
                        S = {next(iter(remaining_N))}

                    clique = sorted([i] + list(S))
                    cliques.append(clique)

                    S_list = sorted(S)
                    for a_idx in range(len(S_list)):
                        for b_idx in range(a_idx + 1, len(S_list)):
                            covered_C_edges.add((S_list[a_idx], S_list[b_idx]))

                    remaining_N -= S

            # Cover remaining C-edges
            remaining_C_edges = C_edges - covered_C_edges
            if remaining_C_edges:
                cliques.extend(cover_remaining_C_edges(remaining_C_edges, deadline))

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

            # Early stop if we hit a very good solution
            if count <= best_count and best_count < 2 * k * k // 3:
                break

        if best_cliques is None:
            # Fallback: one edge per clique
            return [list(e) for e in edges]

        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        # Ensure types are plain Python ints and lists
        clean = []
        for c in cliques:
            clean.append([int(v) for v in c])
        results.append(clean)

    return results
# EVOLVE-BLOCK-END
