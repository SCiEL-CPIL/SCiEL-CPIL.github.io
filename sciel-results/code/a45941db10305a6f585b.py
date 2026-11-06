"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def build_graph(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, min_size=2):
        """Enumerate all maximal cliques of size >= min_size using BK with pivot."""
        n = len(adj)
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= min_size:
                    cliques.append(frozenset(R))
                return
            # Pick pivot u from P ∪ X to minimize branching
            union = P | X
            if union:
                # Choose pivot with max degree in P
                best_u = max(union, key=lambda u: len(P & adj[u]))
                candidates = P - adj[best_u]
            else:
                candidates = P

            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return cliques

    def greedy_hit_set(cliques, n):
        """Greedy set cover: pick vertices to hit all cliques."""
        if not cliques:
            return set()
        # Map each vertex to the set of clique indices it covers
        # For efficiency, track uncovered cliques
        uncovered = set(range(len(cliques)))
        if not uncovered:
            return set()
        # Precompute for each vertex, which cliques it belongs to
        vertex_to_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].add(i)

        selected = set()
        while uncovered:
            # Pick vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                cnt = len(vertex_to_cliques[v] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        return selected

    def local_improve(hit_set, cliques, n, max_iters=500):
        """Try to remove vertices from hit_set while still hitting all cliques."""
        hit_set = set(hit_set)
        if not cliques:
            return hit_set
        # Check if hit_set is valid
        for c in cliques:
            if not (hit_set & c):
                break
        else:
            # Valid, try to remove
            for v in list(hit_set):
                trial = hit_set - {v}
                if not trial:
                    continue
                valid = True
                for c in cliques:
                    if not (trial & c):
                        valid = False
                        break
                if valid:
                    hit_set = trial
        return hit_set

    def improve_by_swapping(hit_set, cliques, n, max_rounds=100):
        """Try swapping: remove one vertex, add another to maintain coverage."""
        hit_set = set(hit_set)
        if not cliques:
            return hit_set
        for _ in range(max_rounds):
            improved = False
            for v in list(hit_set):
                # Try removing v
                without_v = hit_set - {v}
                # Find cliques now uncovered
                uncovered = [c for c in cliques if not (without_v & c)]
                if not uncovered:
                    hit_set = without_v
                    improved = True
                    break
                # Try adding a single vertex w not in without_v to cover all uncovered
                for w in range(n):
                    if w in without_v:
                        continue
                    trial = without_v | {w}
                    if len(trial) < len(hit_set):
                        # Check validity
                        valid = True
                        for c in cliques:
                            if not (trial & c):
                                valid = False
                                break
                        if valid:
                            hit_set = trial
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return hit_set

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Return best effort: all touched vertices
            adj = build_graph(n, p, seed)
            touched = {x for u in range(n) for x in adj[u]}
            results.append(sorted(touched))
            continue

        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj, min_size=2)

        if not cliques:
            results.append([])
            continue

        # Greedy hit set
        hit = greedy_hit_set(cliques, n)

        # Local improvement
        if time.time() < DEADLINE:
            hit = local_improve(hit, cliques, n)
            hit = improve_by_swapping(hit, cliques, n)

        # Try multiple greedy orderings for better result
        best_hit = hit
        if time.time() < DEADLINE:
            rng2 = Random(42)
            for attempt in range(5):
                if time.time() > DEADLINE:
                    break
                # Perturb: randomize tie-breaking in greedy
                perm = list(range(n))
                rng2.shuffle(perm)
                # Redo greedy with different ordering for tie-breaking
                uncovered = set(range(len(cliques)))
                vertex_to_cliques = {v: set() for v in range(n)}
                for i, c in enumerate(cliques):
                    for v in c:
                        vertex_to_cliques[v].add(i)
                selected = set()
                while uncovered:
                    best_v = None
                    best_count = -1
                    for v in perm:
                        if v in selected:
                            continue
                        cnt = len(vertex_to_cliques[v] & uncovered)
                        if cnt > best_count:
                            best_count = cnt
                            best_v = v
                    if best_v is None or best_count <= 0:
                        break
                    selected.add(best_v)
                    uncovered -= vertex_to_cliques[best_v]
                selected = local_improve(selected, cliques, n)
                if len(selected) < len(best_hit):
                    best_hit = selected

        results.append(sorted(best_hit))

    return results
# EVOLVE-BLOCK-END
