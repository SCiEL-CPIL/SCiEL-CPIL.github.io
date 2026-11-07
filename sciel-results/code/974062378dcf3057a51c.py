"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        # Generate the split graph
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # The clique K is vertices 0..k-1, independent set I is k..n-1
        # K-edges: all pairs within K
        # Cross-edges: pairs (u, v) with u < k <= v
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = [(u, v) for u in range(k) for v in range(k, n) if (u, v) in edge_set]

        # Strategy:
        # 1. One big clique on all K vertices covers all K-edges.
        # 2. Each cross-edge (u, v) must be covered. Since I is independent,
        #    any clique containing a cross-edge (u, v) can only contain K-vertices
        #    that are adjacent to v. But to avoid duplicating K-edges (already
        #    covered by the big K-clique), we cannot include two K-vertices in
        #    the same cross-clique if their edge is in the big K-clique.
        #    
        #    Wait - the constraint is that every edge must lie in EXACTLY ONE listed clique.
        #    So if we have the big K-clique covering all K-edges, then no other clique
        #    can contain a K-edge. That means each cross-clique can contain at most
        #    one K-vertex plus some I-vertices. But I-vertices form an independent set,
        #    so a clique with one K-vertex u and I-vertex v is just {u, v} if u-v is an edge.
        #    
        #    Actually, can we have a clique {u, v, w} where u is in K and v, w are in I?
        #    No, because v and w are both in I, which is independent, so (v,w) is not an edge.
        #    So any clique with I-vertices can have at most one I-vertex.
        #    
        #    Therefore, each cross-edge (u, v) must be its own 2-clique {u, v}.
        #    
        #    Total cliques: 1 (for K) + len(cross_edges)

        cliques = [list(range(k))]  # The big K-clique
        for (u, v) in cross_edges:
            cliques.append([u, v])

        # Verify: every clique has at least 2 distinct vertices
        for c in cliques:
            assert len(c) >= 2, f"Clique too small: {c}"
            assert len(set(c)) == len(c), f"Clique has duplicates: {c}"

        # Verify: every edge is in exactly one clique
        edge_count = {}
        for c in cliques:
            c = sorted(c)
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = (c[i], c[j])
                    edge_count[e] = edge_count.get(e, 0) + 1

        for e, cnt in edge_count.items():
            assert cnt == 1, f"Edge {e} appears {cnt} times"
        assert len(edge_count) == len(edge_set), f"Missing edges: {len(edge_set) - len(edge_count)}"

        return cliques

    deadline = time.time() + 140.0  # Safety margin before 150s limit
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            break
        results.append(solve_one(*spec))

    return results
# EVOLVE-BLOCK-END
