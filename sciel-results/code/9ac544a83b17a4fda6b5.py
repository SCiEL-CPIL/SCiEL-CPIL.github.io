import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

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

        # Track remaining edges as a set for O(1) lookup
        remaining = set()
        for u, v in edges:
            remaining.add((min(u, v), max(u, v)))

        cliques = []
        time_limit_inner = deadline - 10  # leave margin for cleanup

        def bron_kerbosch_max_clique(adj, n, time_limit):
            """Find a maximum clique using Bron-Kerbosch with pivot.
            Returns the clique as a list of vertices."""
            best_clique = []
            best_size = 0
            full_mask = (1 << n) - 1

            def bk(R, P, X):
                nonlocal best_clique, best_size
                if time.time() > time_limit:
                    return
                if P == 0 and X == 0:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return
                if P == 0:
                    return  # not maximal, don't update

                # Choose pivot: vertex in P∪X with max neighbors in P
                pivot_set = P | X
                pivot = -1
                pivot_deg = -1
                tmp = pivot_set
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(adj[v] & P)
                    if d > pivot_deg:
                        pivot_deg = d
                        pivot = v
                    tmp ^= lsb

                # Branch on P \ N(pivot)
                candidates = P & ~(adj[pivot] if pivot >= 0 else 0)
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    if time.time() > time_limit:
                        return
                    R.append(v)
                    bk(R, P & adj[v], X & adj[v])
                    R.pop()
                    P &= ~(1 << v)
                    X |= (1 << v)

            bk([], full_mask, 0)
            return best_clique

        def find_good_clique(adj, n, remaining, time_limit):
            """Find a large clique in the remaining-edge subgraph."""
            # Build subgraph of remaining edges
            sub_adj = [0] * n
            for u, v in remaining:
                sub_adj[u] |= (1 << v)
                sub_adj[v] |= (1 << u)

            # Try Bron-Kerbosch on the full vertex set but restricted to sub_adj
            # For efficiency, only consider vertices with degree > 0 in subgraph
            active_mask = 0
            for v in range(n):
                if sub_adj[v]:
                    active_mask |= (1 << v)

            if active_mask == 0:
                return None

            best_clique = []
            best_size = 0

            def bk(R, P, X):
                nonlocal best_clique, best_size
                if time.time() > time_limit:
                    return
                if P == 0 and X == 0:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return
                if P == 0:
                    return

                # Pivot
                pivot_set = P | X
                pivot = -1
                pivot_deg = -1
                tmp = pivot_set
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(sub_adj[v] & P)
                    if d > pivot_deg:
                        pivot_deg = d
                        pivot = v
                    tmp ^= lsb

                candidates = P & ~(sub_adj[pivot] if pivot >= 0 else 0)
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    if time.time() > time_limit:
                        return
                    R.append(v)
                    bk(R, P & sub_adj[v], X & sub_adj[v])
                    R.pop()
                    P &= ~(1 << v)
                    X |= (1 << v)

            bk([], active_mask, 0)

            if best_size >= 2:
                return best_clique

            # Fallback: find any edge
            if remaining:
                e = next(iter(remaining))
                return list(e)
            return None

        while remaining:
            if time.time() > time_limit_inner:
                # Emergency: pack remaining edges as pairs
                for e in list(remaining):
                    cliques.append(list(e))
                break

            clique = find_good_clique(adj, n, remaining, time_limit_inner)

            if clique is None or len(clique) < 2:
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue

            # Remove all edges within the clique from remaining
            cset = set(clique)
            to_remove = []
            for e in remaining:
                if e[0] in cset and e[1] in cset:
                    to_remove.append(e)

            if to_remove:
                cliques.append(clique)
                for e in to_remove:
                    remaining.discard(e)
            else:
                # Shouldn't happen, but fallback
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
