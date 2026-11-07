"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    _deadline = time.time() + 130.0

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v - k))
        
        m = n - k
        nbr_C = [set() for _ in range(k)]
        nbr_I = [set() for _ in range(m)]
        for u, vi in cross_edges:
            nbr_C[u].add(vi)
            nbr_I[vi].add(u)
        
        rem_nbr_C = [set(s) for s in nbr_C]
        rem_nbr_I = [set(s) for s in nbr_I]
        rem = set(cross_edges)
        
        cliques = []
        while rem:
            if time.time() > _deadline - 5:
                break
            best = None
            best_cnt = -1
            for u in range(k):
                if not rem_nbr_C[u]:
                    continue
                B = set(rem_nbr_C[u])
                A = set(u2 for u2 in range(k) if B <= rem_nbr_C[u2])
                # Expand B
                B2 = None
                for u2 in A:
                    if B2 is None:
                        B2 = set(rem_nbr_C[u2])
                    else:
                        B2 &= rem_nbr_C[u2]
                if B2:
                    B = B2
                    A = set(u2 for u2 in range(k) if B <= rem_nbr_C[u2])
                cnt = len(A) * len(B)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = (A, B)
            if best is None:
                break
            A, B = best
            cliques.append((A, B))
            for u in A:
                for vi in B:
                    rem.discard((u, vi))
                    rem_nbr_C[u].discard(vi)
                    rem_nbr_I[vi].discard(u)
        
        result = [list(range(k))]
        for A, B in cliques:
            result.append(list(A) + [v + k for v in sorted(B)])
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _deadline - 5:
            break
        results.append(solve_one(*spec))
    return results
# EVOLVE-BLOCK-END
