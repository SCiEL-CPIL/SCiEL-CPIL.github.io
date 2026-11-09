"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        # Build adjacency
        rng = Random(seed)
        # Cross edges: for each v in [k, n), which u in [0, k) is adjacent?
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)

        # Build cross-edge adjacency: cross_adj[v] = set of u in [0,k) adjacent to v (v in [k,n))
        cross_adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if u in adj[v]:
                    cross_adj[v].add(u)

        # Independent vertices are I = {k, k+1, ..., n-1}
        I = list(range(k, n))

        # Group independent vertices by their neighborhood in the clique part
        # Two independent vertices with identical neighborhoods form a clique with any common neighbor
        # Actually, independent vertices are not adjacent to each other.
        # So cliques can contain at most one independent vertex plus some clique vertices.
        # A clique containing independent vertex v and clique vertices S must have S ⊆ N(v) ∩ C,
        # and S must be a clique in C (but C is a clique, so any subset works).
        # The edge (v, u) for u in S is covered by this clique.
        # We need to cover all cross edges (v, u) where v in I, u in C, and (v,u) is an edge.
        # Each clique covers cross edges from one independent vertex v to a subset S ⊆ N(v).
        # To minimize number of cliques, for each v, we want to cover N(v) with as few cliques as possible.
        # But each clique contains exactly one v (since I vertices are not adjacent to each other).
        # So for each v, we need to partition N(v) into subsets, each becoming a clique {v} ∪ S.
        # But wait - we also need to cover edges within C. Those are covered by cliques within C.
        # The clique C itself is a clique of size k. We can use {C} as one clique to cover all edges within C.
        # But then cross edges need separate cliques.
        # Alternatively, we can split C into parts and combine with I vertices.
        # Key insight: each listed clique can contain vertices from C and at most one vertex from I.
        # So the total number of cliques = (cliques covering C-edges) + (cliques covering cross edges).
        # But a clique {v} ∪ S where v in I, S ⊆ C covers both cross edges (v, s) for s in S,
        # AND if S has ≥ 2 elements, it covers edges within S (which are C-edges).
        # So we can "amortize" C-edge coverage by combining C vertices with I vertices.
        #
        # Strategy: For each independent vertex v, its neighborhood N(v) ⊆ C must be covered.
        # We can group multiple independent vertices that share neighbors.
        # But two independent vertices v1, v2 cannot be in the same clique (they're not adjacent).
        # So each clique has at most one I vertex.
        #
        # Let's think differently. Total edges = C-edges + cross edges.
        # C-edges: all pairs in C. These must be in cliques.
        # Cross edges: (v, u) for v in I, u in N(v).
        #
        # A clique can be:
        # - A subset of C (covers C-edges within that subset)
        # - {v} ∪ S where v in I, S ⊆ C (covers cross edges (v,s) for s in S, and C-edges within S)
        #
        # To minimize total cliques, we want to maximize the "efficiency" of each clique.
        # A clique {v} ∪ S with |S| = s covers s cross edges and C(s,2) C-edges.
        # A clique T ⊆ C with |T| = t covers C(t,2) C-edges.
        #
        # The C-edges that are NOT covered by any {v}∪S clique must be covered by pure C-cliques.
        #
        # Let's use a greedy approach:
        # 1. Start with the full clique C as one candidate.
        # 2. For each v in I, we need to cover N(v). If we include v in a clique with some subset S ⊆ N(v),
        #    we cover those cross edges and the C-edges within S.
        #
        # Alternative approach: 
        # - First, cover all cross edges. For each v in I, create cliques to cover N(v).
        # - Then, cover remaining C-edges.
        #
        # But we can be smarter. Let's consider the structure:
        # Each v in I has a neighborhood N(v) ⊆ C. 
        # If we assign each v to a "group" with a specific subset of C, we reduce C-edges that need separate coverage.
        #
        # Simplest good approach:
        # 1. Use one clique for all of C: {0, 1, ..., k-1}. This covers all C-edges.
        # 2. For each v in I, cover N(v) with cliques {v} ∪ {u} for each u in N(v). That's |N(v)| cliques per v.
        #    Total: 1 + sum(|N(v)| for v in I)
        #
        # Better approach:
        # 1. For each v in I, if N(v) is non-empty, create one clique {v} ∪ N(v). This covers all cross edges for v
        #    and the C-edges within N(v).
        # 2. The C-edges NOT within any N(v) for v in I need to be covered.
        #    Let U = C \ (union of all N(v) for v in I). Edges within U and between U and (C \ U) that aren't covered...
        #    Wait, edges within N(v) are covered by clique {v} ∪ N(v). But edges between different N(v1) and N(v2) 
        #    or between N(v) and U may not be covered.
        #
        # Let me think again. After creating cliques {v} ∪ N(v) for each v in I with N(v) non-empty:
        # - Cross edges: all covered (each (v,u) is in clique {v} ∪ N(v)).
        # - C-edges: an edge (a,b) in C is covered if there exists v in I such that a,b ∈ N(v).
        #   Otherwise, it's uncovered.
        #
        # So we need to cover the remaining C-edges. The remaining edges are those (a,b) such that no v has {a,b} ⊆ N(v).
        #
        # This is getting complex. Let me use a simpler but still good heuristic:
        #
        # Approach: Greedy set cover for C-edges, using cliques that may include I vertices.
        #
        # Actually, let me just use the straightforward approach that's likely decent:
        # 1. For each v in I, if N(v) has size ≥ 2, make clique {v} ∪ N(v).
        # 2. For remaining cross edges (where N(v) had size 0 or 1), handle separately.
        # 3. Cover remaining C-edges.
        #
        # Let me implement a clean version:

        # Build list of cross edges
        cross_edges = []
        for v in I:
            for u in sorted(cross_adj[v]):
                cross_edges.append((v, u))

        # C-edges
        c_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]

        # Total edges to cover
        all_edges = set()
        for u, v in c_edges:
            all_edges.add((min(u,v), max(u,v)))
        for v, u in cross_edges:
            all_edges.add((min(u,v), max(u,v)))

        covered = set()
        cliques = []

        # Step 1: For each independent vertex v, if |N(v)| >= 2, create clique {v} + N(v)
        for v in I:
            if len(cross_adj[v]) >= 2:
                clique = [v] + sorted(cross_adj[v])
                cliques.append(clique)
                # Mark all edges in this clique as covered
                clique_set = set(clique)
                for a in clique:
                    for b in clique:
                        if a < b:
                            covered.add((a, b))

        # Step 2: Handle remaining cross edges (for v with |N(v)| < 2, or edges not yet covered)
        # For each remaining cross edge (v, u), create clique [v, u]
        for v in I:
            for u in sorted(cross_adj[v]):
                e = (min(v,u), max(v,u))
                if e not in covered:
                    cliques.append([v, u])
                    covered.add(e)

        # Step 3: Cover remaining C-edges
        # The remaining C-edges are those not covered by any clique yet
        # Use greedy: find largest subset of C that forms a clique of uncovered edges
        # Since C is a clique, any subset is a clique. We just need to cover the edges.
        # Greedy: repeatedly pick the C-vertex with the most uncovered C-edges, form a clique with it
        # and all its neighbors via uncovered C-edges.
        
        # Build uncovered C-edge set
        uncov_c = set()
        for u, v in c_edges:
            e = (u, v)
            if e not in covered:
                uncov_c.add(e)

        if uncov_c:
            # Greedy edge cover on the C-clique (which is complete)
            # Just repeatedly take pairs or larger groups
            # Since C is a clique, we can cover all remaining C-edges efficiently.
            # The vertices involved in uncov_c:
            involved = set()
            for u, v in uncov_c:
                involved.add(u)
                involved.add(v)
            involved = sorted(involved)
            
            # Use a greedy approach: while there are uncovered edges,
            # pick a vertex with max degree in uncov_c, take it with all its uncov neighbors
            while uncov_c:
                # Find vertex with max uncovered degree
                deg = {}
                for u, v in uncov_c:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                max_v = max(deg, key=lambda x: deg[x])
                # Get all neighbors of max_v via uncov_c
                neighbors = set()
                remaining = set()
                for u, v in uncov_c:
                    if u == max_v:
                        neighbors.add(v)
                    elif v == max_v:
                        neighbors.add(u)
                    else:
                        remaining.add((u, v))
                # Clique: {max_v} + neighbors
                if len(neighbors) >= 1:
                    clique = [max_v] + sorted(neighbors)
                    cliques.append(clique)
                    # Mark edges covered
                    for a in clique:
                        for b in clique:
                            if a < b:
                                covered.add((a, b))
                    uncov_c -= {(a, b) for a in clique for b in clique if a < b}
                else:
                    # Should not happen since max_v has degree >= 1
                    break

        # Verify all edges are covered
        # (We trust the logic, but let's ensure validity)
        
        # Return cliques
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_one(*spec))

    return results
# EVOLVE-BLOCK-END
