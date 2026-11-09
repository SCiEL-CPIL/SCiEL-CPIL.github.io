"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _rnd

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s kill

    def generate_edges(n, p, seed):
        rng = _rnd.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        """Return list of maximal cliques (each as frozenset) using pivot-based BK with bitmasks."""
        cliques = []
        all_mask = (1 << n) - 1

        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            # Pivot: choose vertex in P|X with max degree in P
            PX = P | X
            if PX:
                # Pick pivot with max |adj[p] & P|
                best_p = -1
                best_cnt = -1
                pp = PX
                while pp:
                    lb = pp & (-pp)
                    p = lb.bit_length() - 1
                    cnt = (adj[p] & P).bit_count()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_p = p
                    pp ^= lb
                # Candidates: P \ adj[pivot]
                candidates = P & ~(adj[best_p])
            else:
                candidates = P

            cp = candidates
            while cp:
                lb = cp & (-cp)
                u = lb.bit_length() - 1
                cp ^= lb
                bk(R | lb, P & adj[u], X & adj[u])
                P ^= lb
                X |= lb

        bk(0, all_mask, 0)
        return cliques

    def filter_cliques(cliques, min_size=2):
        return [c for c in cliques if len(c) >= min_size]

    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        # Map each clique to index
        uncovered = set(range(len(cliques)))
        # For each vertex, which cliques it hits
        vertex_cliques = [[] for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(ci)

        hit = set()
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = -1
            best_cnt = -1
            for v in range(n):
                cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_cnt <= 0:
                break
            hit.add(best_v)
            for ci in vertex_cliques[best_v]:
                uncovered.discard(ci)
        return hit

    def is_valid(hit, cliques):
        for c in cliques:
            if not any(v in hit for v in c):
                return False
        return True

    def local_search(hit, cliques, n, time_limit):
        """Try deletions and 2-opt swaps to reduce size."""
        hit = set(hit)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            # Try deletions
            for v in list(hit):
                trial = hit - {v}
                if is_valid(trial, cliques):
                    hit = trial
                    improved = True
                    break
            # Try 2-opt swaps: remove v, add w (w not in hit)
            if not improved:
                for v in list(hit):
                    if time.time() > t0 + time_limit:
                        break
                    trial = hit - {v}
                    for w in range(n):
                        if w in hit:
                            continue
                        trial2 = trial | {w}
                        if len(trial2) < len(hit):
                            if is_valid(trial2, cliques):
                                hit = trial2
                                improved = True
                                break
                    if improved:
                        break
        return hit

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = deadline - time.time()
        if remaining < 5:
            # Not enough time; return all vertices as fallback
            results.append(list(range(n)))
            continue

        edges = generate_edges(n, p, seed)
        adj = build_adjacency(n, edges)
        cliques = bron_kerbosch(adj, n)
        cliques = filter_cliques(cliques, min_size=2)

        if not cliques:
            results.append([])
            continue

        # Greedy initial solution
        hit = greedy_hitting_set(cliques, n)

        # Local search with time budget
        ls_budget = min(remaining * 0.6, 30.0)
        hit = local_search(hit, cliques, n, ls_budget)

        # Verify validity
        if not is_valid(hit, cliques):
            # Fallback: all vertices in any clique
            all_v = set()
            for c in cliques:
                all_v |= c
            hit = all_v

        results.append(sorted(hit))

    return results
# EVOLVE-BLOCK-END
