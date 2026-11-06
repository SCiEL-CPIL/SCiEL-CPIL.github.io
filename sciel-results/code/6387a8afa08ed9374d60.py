import time
from random import Random

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def popcount(x):
        return bin(x).count('1')

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Track remaining edges
        remaining_edges = set()
        for u, v in edges:
            remaining_edges.add((min(u, v), max(u, v)))

        cliques = []

        def find_candidate_cliques(active_adj, n):
            """Generate multiple candidate cliques using greedy growth from different starts."""
            active_vertices = [v for v in range(n) if active_adj[v] != 0]
            if not active_vertices:
                return []
            active_vertices.sort(key=lambda v: popcount(active_adj[v]), reverse=True)

            candidates = []
            seen = set()

            for start in active_vertices[:min(len(active_vertices), 20)]:
                if time.time() > deadline - 15:
                    break
                clique = [start]
                P = active_adj[start] & ~(1 << start)
                while P:
                    best_v = -1
                    best_d = -1
                    tmp = P
                    while tmp:
                        lsb = tmp & (-tmp)
                        v = lsb.bit_length() - 1
                        d = popcount(active_adj[v] & P)
                        if d > best_d:
                            best_d = d
                            best_v = v
                        tmp ^= lsb
                    if best_v < 0:
                        break
                    clique.append(best_v)
                    P = P & active_adj[best_v]
                    P &= ~(1 << best_v)

                if len(clique) >= 2:
                    key = frozenset(clique)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(clique)

            return candidates

        def coverage_ratio(clique, active_adj, n):
            """Count how many remaining edges are covered by this clique."""
            clique_set = set(clique)
            covered = 0
            for u in clique:
                mask = active_adj[u] & (1 << n)
                # Only count edges to vertices in clique with higher index
                for v in clique:
                    if v > u and (mask >> v) & 1:
                        covered += 1
            return covered

        while remaining_edges:
            if time.time() > deadline - 20:
                rem_list = list(remaining_edges)
                for e in rem_list:
                    cliques.append(list(e))
                break

            # Build active adjacency from remaining edges
            active_adj = [0] * n
            for u, v in remaining_edges:
                active_adj[u] |= (1 << v)
                active_adj[v] |= (1 << u)

            # Generate candidate cliques and pick the one with best coverage/size ratio
            candidates = find_candidate_cliques(active_adj, n)

            best_clique = None
            best_ratio = -1

            for cand in candidates:
                cov = coverage_ratio(cand, active_adj, n)
                if cov > 0:
                    ratio = cov / len(cand)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_clique = cand

            if best_clique is None or len(best_clique) < 2:
                e = next(iter(remaining_edges))
                cliques.append(list(e))
                remaining_edges.discard(e)
                continue

            # Remove all edges within the chosen clique
            clique_set = set(best_clique)
            to_remove = set()
            for u, v in remaining_edges:
                if u in clique_set and v in clique_set:
                    to_remove.add((u, v))

            if to_remove:
                cliques.append(best_clique)
                remaining_edges.difference_update(to_remove)
            else:
                e = next(iter(remaining_edges))
                cliques.append(list(e))
                remaining_edges.discard(e)

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
