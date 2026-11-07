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
        Find a minimum clique partition for a split graph.
        K = {0, ..., k-1} is a clique.
        I = {k, ..., n-1} is an independent set.
        
        Strategy:
        1. The clique K itself is one clique (covers all K-K edges).
        2. For each vertex v in I, let N_K(v) be its neighbors in K.
           Each edge (u, v) where u in K, v in I must be covered by a clique.
           Since I is independent, any clique containing v can only contain
           vertices from N_K(v) ∪ {v}.
           So we need to cover the bipartite edges between K and I.
           
        For each v in I, the edges from v to K must be covered by cliques that
        contain v and some subset of N_K(v). Since each such clique is {v} ∪ S
        where S ⊆ N_K(v) and S is a clique in K (which it always is since K is
        a clique), we just need to cover the edges.
        
        But wait - a clique {v} ∪ S covers all edges from v to vertices in S.
        So for vertex v, we need to cover all edges from v to N_K(v).
        We can do this with one clique: {v} ∪ N_K(v). This is valid since K is
        a clique, so N_K(v) is a subset of K which is a clique.
        
        So the partition is:
        - One clique: K = {0, ..., k-1}
        - For each v in I: one clique {v} ∪ N_K(v)
        
        But we can do better! If two vertices v1, v2 in I have the same
        neighborhood N_K(v1) = N_K(v2) = S, then the clique {v1, v2} ∪ S
        is valid (since S is a clique in K, and v1, v2 are not adjacent to each
        other... wait, they're in I which is independent, so v1 and v2 are NOT
        adjacent. So {v1, v2} ∪ S is NOT a clique.
        
        Hmm, so we can't combine vertices from I into the same clique.
        
        Actually, let me reconsider. A clique in a split graph can contain:
        - Some vertices from K and some from I
        - But no two vertices from I (since I is independent)
        
        So each clique contains at most one vertex from I.
        
        Therefore, for each v in I, we need at least one clique containing v
        to cover its edges to K. The minimal such clique is {v} ∪ N_K(v).
        
        But wait - could we use multiple cliques for a single v? Yes, but that
        would increase the count. So one clique per v in I is optimal.
        
        Total cliques = 1 (for K) + |I| = 1 + (n - k)
        
        But we can also consider: do we need the clique K separately?
        The edges within K are only covered by the clique K itself (since no
        other clique contains two vertices from K... actually, the cliques
        {v} ∪ N_K(v) do contain vertices from K).
        
        Wait, the clique {v} ∪ N_K(v) contains vertices from K. So those K
        vertices appear in multiple cliques. But the edges between two K
        vertices u1, u2 are only covered if there's a clique containing both
        u1 and u2. The clique K contains all of them. The cliques {v} ∪ N_K(v)
        contain subsets of K, but not necessarily all pairs.
        
        So we still need the clique K to cover all K-K edges.
        
        Therefore, the minimum number of cliques is 1 + (n - k).
        
        Let me verify: each edge must be in exactly one clique.
        - K-K edges: only in clique K. ✓
        - K-I edges: edge (u, v) where u in K, v in I is in clique {v} ∪ N_K(v)
          if u in N_K(v). It's not in any other clique because no other clique
          contains v. ✓
        - I-I edges: none exist (I is independent). ✓
        
        So the partition is valid and has 1 + (n - k) cliques.
        """
        rng = Random(seed)
        # Generate the graph to get neighborhoods
        adj_K = [set() for _ in range(k)]  # neighbors in K for each vertex in I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj_K[v - k].add(u)
        
        # Build the partition
        cliques = []
        # Clique for K
        cliques.append(list(range(k)))
        # One clique per vertex in I
        for i, v in enumerate(range(k, n)):
            clique = [v] + sorted(adj_K[i])
            if len(clique) >= 2:
                cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END
