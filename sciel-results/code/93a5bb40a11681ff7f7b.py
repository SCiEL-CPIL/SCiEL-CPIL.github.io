"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        """
        Find a clique partition for a split graph.
        Strategy:
        1. Use the full clique K (first k vertices) to cover all edges within K.
        2. For each vertex v in I (independent set), find its neighbors in K.
           Group these neighbors into cliques (which are just subsets of K, so any subset works).
           For each such subset S, create a clique S ∪ {v}.
        To minimize the number of cliques, we want to cover the neighbor set of each v
        with as few subsets as possible. Since any subset of K is a clique, we can cover
        all neighbors of v with a single clique: N(v) ∪ {v}.
        This gives:
        - 1 clique for K (if k >= 2)
        - For each v in I with at least one neighbor in K, one clique N(v) ∪ {v}
        
        This is valid because:
        - Edges within K are covered by the K clique.
        - Edges between K and I: edge (u, v) where u in K, v in I is covered by clique N(v) ∪ {v}
          since u is in N(v).
        - No edges within I (since I is independent).
        - Each edge is in exactly one clique:
          * Edges within K: only in the K clique.
          * Edges (u, v) with u in K, v in I: only in the clique for v (since u is in N(v)
            and no other clique contains both u and v).
        
        Total cliques: 1 + (number of vertices in I with at least one neighbor in K)
        """
        # Generate the graph
        rng = Random(seed)
        # Build adjacency for cross edges
        # For each v in I (v >= k), determine which u in K (u < k) are connected
        neighbors_of_i = []  # neighbors_of_i[v - k] = list of neighbors in K
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors_of_i.append(nbrs)
        
        cliques = []
        
        # Clique for K (only if k >= 2)
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each vertex in I with neighbors, add a clique
        for v_idx, nbrs in enumerate(neighbors_of_i):
            if len(nbrs) >= 1:
                v = k + v_idx
                # The clique is nbrs + [v]
                # Must have at least 2 vertices: nbrs has >= 1, plus v, so >= 2
                cliques.append(nbrs + [v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            break
        results.append(clique_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
