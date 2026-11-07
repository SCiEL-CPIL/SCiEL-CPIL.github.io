"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        edges += cross_edges
        return edges

    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Determine neighbors in C for each vertex in I
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs

        # Strategy: We have two types of cliques:
        # 1. Pure C-cliques (subsets of C) covering C-C edges
        # 2. Mixed cliques {v} ∪ S where v ∈ I, S ⊆ N_C(v), covering C-I edges and some C-C edges
        #
        # Key insight: If we use one big clique C, we cover all C-C edges in 1 clique,
        # and each I-vertex needs 1 clique for its edges: total = 1 + (n-k)
        #
        # But we can do better: if an I-vertex v has neighbors S ⊆ C, we can put v with S
        # in a clique, which covers C-I edges (v,u) for u∈S AND C-C edges (a,b) for a,b∈S.
        # This means we DON'T need to cover those C-C edges in the pure C-clique.
        #
        # Optimal approach: 
        # - For each I-vertex v, create clique {v} ∪ N_C(v). This covers all edges of v.
        # - The C-C edges NOT covered by any mixed clique must be covered by pure C-cliques.
        # - A C-C edge (a,b) is covered by a mixed clique if there exists v∈I with a,b ∈ N_C(v).
        #
        # So: remaining C-C edges = all C-C edges minus those covered by at least one mixed clique.
        # We need to cover remaining C-C edges with pure C-cliques (minimum clique cover of a subgraph of C).
        #
        # Since C is a complete graph, the remaining C-C edges form a subgraph. 
        # Minimum clique cover of a general graph is NP-hard, but for small k (≤30) we can use
        # greedy or exact methods.
        #
        # Actually, let's think more carefully. The mixed cliques {v}∪N_C(v) for all v∈I
        # cover C-C edge (a,b) if ∃v: a,b ∈ N_C(v).
        # 
        # Alternative: we don't have to use {v}∪N_C(v) for every v. We could split v's
        # neighbors into multiple cliques. But that uses MORE cliques for v's edges.
        # So using one clique per I-vertex is optimal for covering I-vertex edges.
        #
        # Therefore: 
        # - Create clique {v}∪N_C(v) for each v∈I with |N_C(v)|≥2 (if |N_C(v)|<2, use edge cliques)
        # - Cover remaining C-C edges with minimum clique cover

        # Build set of C-C edges covered by mixed cliques
        covered_cc = set()
        for v in range(k, n):
            nbrs = neighbors[v]
            if len(nbrs) >= 2:
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        a, b = nbrs[i], nbrs[j]
                        e = (a, b) if a < b else (b, a)
                        covered_cc.add(e)

        # All C-C edges
        all_cc = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_cc.add((u, v))

        # Remaining C-C edges to cover
        remaining_cc = all_cc - covered_cc

        # Build cliques
        cliques = []

        # Mixed cliques for I-vertices
        for v in range(k, n):
            nbrs = neighbors[v]
            if len(nbrs) >= 2:
                cliques.append(list(nbrs) + [v])
            elif len(nbrs) == 1:
                cliques.append([nbrs[0], v])
            # if len(nbrs) == 0: no edges, no clique needed

        # Cover remaining C-C edges with pure C-cliques
        # Greedy: repeatedly find largest clique in remaining graph
        if remaining_cc:
            # Build adjacency for remaining C-C subgraph
            rem_adj = {u: set() for u in range(k)}
            for a, b in remaining_cc:
                rem_adj[a].add(b)
                rem_adj[b].add(a)

            rem_edges = set(remaining_cc)
            while rem_edges:
                # Find largest clique in remaining subgraph using greedy
                # Start with vertex of highest degree
                best_start = None
                best_deg = -1
                for u in rem_adj:
                    if len(rem_adj[u]) > best_deg:
                        best_deg = len(rem_adj[u])
                        best_start = u

                if best_start is None:
                    break

                # Greedy clique: start with best_start, add vertices connected to all current members
                clique = [best_start]
                candidates = set(rem_adj[best_start])
                while candidates:
                    # Pick vertex with highest degree among candidates
                    best_v = None
                    best_cd = -1
                    for c in candidates:
                        cd = len(rem_adj[c])
                        if cd > best_cd:
                            best_cd = cd
                            best_v = c
                    if best_v is None:
                        break
                    # Check if best_v is connected to all in clique
                    connected = all(best_v in rem_adj[x] for x in clique)
                    if connected:
                        clique.append(best_v)
                        candidates = candidates & rem_adj[best_v]
                    else:
                        candidates.discard(best_v)

                if len(clique) >= 2:
                    cliques.append(clique)
                    # Remove all edges within this clique from remaining
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            a, b = clique[i], clique[j]
                            e = (a, b) if a < b else (b, a)
                            rem_edges.discard(e)
                else:
                    # Shouldn't happen if rem_edges is non-empty, but safety
                    break

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        result.append(solve_split(*spec))

    return result
# EVOLVE-BLOCK-END
