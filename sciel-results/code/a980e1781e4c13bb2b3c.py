"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split_clique_partition(n, k, p, seed):
        """
        Find minimum clique partition of a split graph.
        
        Split graph structure:
        - Clique C = {0, ..., k-1}
        - Independent set I = {k, ..., n-1}
        - Cross edges: each (c, i) present with probability p
        
        Key insight: The clique C itself is one clique. The challenge is covering
        all cross edges (c, i) with cliques that include vertices from both C and I.
        
        Since I is independent, any clique can contain at most one vertex from I.
        So each clique covering cross edges has the form {i} ∪ S_i where S_i ⊆ C.
        
        We need to cover all cross edges. Each cross edge (c, i) must be in some clique
        containing both c and i. Since a clique with i can only include vertices from C
        that are all adjacent to i, we can form cliques of the form {i} ∪ N_C(i).
        
        But we want to minimize the number of cliques. The key optimization:
        - We can merge multiple i's into the same "type" of clique if they have the same neighborhood in C.
        - Actually, no: each clique can only have one vertex from I (since I is independent).
        
        So the problem reduces to: for each i in I, we need at least one clique containing i.
        The clique containing i can be {i} ∪ S where S ⊆ N_C(i). To minimize total cliques,
        we want each such clique to cover as many cross edges as possible, but since each clique
        only has one vertex from I, the cross edges covered by clique {i} ∪ S are exactly {(c, i) : c in S}.
        
        Wait - we need to cover ALL cross edges. Each cross edge (c, i) must appear in exactly one clique.
        So for each i, the set of c's that are connected to i must be partitioned among the cliques that contain i.
        But we can only have cliques where all c's in the clique are adjacent to i.
        
        Actually, let me reconsider. We need:
        1. Every edge is in exactly one clique.
        2. Every clique is a valid clique (all pairs adjacent).
        
        The clique C = {0,...,k-1} covers all edges within C.
        For cross edges: each edge (c, i) must be in a clique. That clique must contain both c and i,
        and all vertices in the clique must be mutually adjacent. Since i is only adjacent to its neighbors in C,
        any clique containing i can only include vertices from N_C(i) ∪ {i}.
        
        So for each i, we need to cover all edges (c, i) for c in N_C(i). We can do this with one clique:
        {i} ∪ N_C(i). This covers all cross edges incident to i.
        
        But wait - can we do better? Can we share cliques? No, because each clique can only contain one vertex from I.
        So we need at least one clique per i that has at least one cross edge.
        
        However, we could potentially use multiple cliques for a single i if N_C(i) is large, but that would
        increase the count. The minimum is one clique per i with cross edges, plus the main clique C.
        
        But actually, we can be smarter. If two vertices i1, i2 in I have the same neighborhood in C,
        we still need separate cliques for each (since a clique can't contain both i1 and i2).
        
        So the minimum number of cliques is:
        1 (for C) + |{i in I : N_C(i) is non-empty}|
        
        But wait, we also need to make sure each listed clique has at least 2 vertices.
        The clique {i} ∪ N_C(i) has at least 2 vertices iff N_C(i) is non-empty.
        
        So the answer is: 1 + number of i in I with at least one neighbor in C.
        
        Let me verify: is this always a valid clique partition?
        - Clique 1: C = {0, ..., k-1} - valid, covers all C-C edges.
        - For each i with N_C(i) non-empty: clique {i} ∪ N_C(i) - valid clique, covers all cross edges incident to i.
        - Each cross edge (c, i) is in exactly one clique: the clique for i.
        - No edge is in two cliques: C-C edges only in C, cross edges only in their respective i-cliques.
        - No edges within I (I is independent).
        
        This is optimal because each i with cross edges needs its own clique (can't share with other i's).
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        cross_edges = set()
        for u, v in edges:
            if u < k and v >= k:
                cross_edges.add((u, v))
            elif v < k and u >= k:
                cross_edges.add((v, u))
        
        # For each i in I, find its neighbors in C
        neighbors_in_C = {i: set() for i in range(k, n)}
        for c, i in cross_edges:
            neighbors_in_C[i].add(c)
        
        # Build the clique partition
        cliques = []
        
        # Main clique C
        cliques.append(list(range(k)))
        
        # For each i with neighbors in C, add clique {i} ∪ N_C(i)
        for i in range(k, n):
            if neighbors_in_C[i]:
                clique = [i] + sorted(neighbors_in_C[i])
                cliques.append(clique)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return trivial solution if we're running out of time
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        cliques = solve_split_clique_partition(*spec)
        results.append(cliques)
    
    # Ensure all types are correct: ints must be Python int, sequences must be list or tuple
    final_results = []
    for cliques in results:
        clean_cliques = []
        for clique in cliques:
            clean_clique = [int(v) for v in clique]
            clean_cliques.append(clean_clique)
        final_results.append(clean_cliques)
    
    return final_results

# EVOLVE-BLOCK-END
