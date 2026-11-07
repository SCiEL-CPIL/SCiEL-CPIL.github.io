"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def get_cliques_for_instance(n, k, p, seed):
        """
        For a split graph with clique C = {0..k-1} and independent set I = {k..n-1},
        we need a clique partition.
        
        Strategy:
        1. The clique C itself is a clique. But we can't just list it as one clique 
           because edges between C and I must be covered by cliques that include 
           vertices from both sides.
        
        Actually, in a split graph, any clique is either:
        - A subset of C (the clique part)
        - A vertex from I plus a subset of C that is adjacent to it
        
        Since I is an independent set, no clique can contain two vertices from I.
        So every clique is either:
        - A subset of C (size >= 2)
        - {i} ∪ S where i ∈ I and S ⊆ N(i) ∩ C, and S ∪ {i} forms a clique (which it does if S ⊆ N(i) since C is a clique)
        
        Wait, for {i} ∪ S to be a clique, we need all pairs in S to be adjacent (true, since C is clique) 
        and i to be adjacent to all in S (true by definition of S ⊆ N(i)).
        
        So the problem reduces to:
        - Cover all edges in C∩C with cliques that are subsets of C
        - Cover all edges between C and I with cliques of the form {i} ∪ S where S ⊆ N(i)
        
        But note: an edge (u,v) with u,v ∈ C can be covered by:
        - A clique that is a subset of C containing both u and v
        - OR a clique {i} ∪ S where i ∈ I, u,v ∈ S ⊆ N(i), and i is adjacent to both u and v
        
        An edge (u,i) with u ∈ C, i ∈ I can only be covered by a clique {i} ∪ S where u ∈ S.
        
        Key insight: Each vertex i ∈ I has a neighborhood N(i) ⊆ C. The edges from i to N(i) 
        must be covered. We can cover them all with one clique {i} ∪ N(i) if N(i) is non-empty.
        But then the edges within N(i) are also covered by this clique.
        
        However, edges within C that are not in any N(i) for i ∈ I must be covered by cliques 
        that are subsets of C.
        
        Let's think about it differently. We want to minimize the number of cliques.
        
        Approach:
        1. For each i ∈ I, if N(i) is non-empty, create clique {i} ∪ N(i). This covers all edges 
           incident to i, plus all edges within N(i).
        2. After doing this for all i ∈ I, some edges within C may still be uncovered. 
           Specifically, an edge (u,v) with u,v ∈ C is uncovered if there is no i ∈ I such that 
           both u and v are in N(i).
        3. For the remaining uncovered edges within C, we need to cover them with cliques that 
           are subsets of C.
        
        But wait, this greedy approach may not be optimal. Let me reconsider.
        
        Actually, every edge (u,i) with i ∈ I MUST be in a clique that contains i. Since I is 
        independent, that clique must be of the form {i} ∪ S where S ⊆ N(i). To minimize the 
        number of cliques, we should make S as large as possible, i.e., S = N(i). So for each 
        i ∈ I with N(i) ≠ ∅, we have a "forced" clique {i} ∪ N(i).
        
        But we could also choose to split the edges from i into multiple cliques. For example, 
        if N(i) = {u, v, w}, we could use {i, u, v} and {i, w} instead of {i, u, v, w}. But 
        that uses 2 cliques instead of 1, so it's worse unless it helps cover other edges better.
        
        Hmm, but the edges within N(i) are covered by {i} ∪ N(i). If we split, we might not 
        cover all edges within N(i) in those cliques. For example, {i, u, v} covers (u,v) but 
        {i, w} does not cover (u,w) or (v,w). So we'd need additional cliques for those.
        
        So it seems optimal to use {i} ∪ N(i) for each i ∈ I. Let's verify with a small example.
        
        Actually, I think the key is: the cliques {i} ∪ N(i) for i ∈ I cover all edges incident 
        to I, and also cover some edges within C. The remaining edges within C are those not 
        contained in any N(i) for i ∈ I. We need to cover those with cliques that are subsets of C.
        
        To minimize the number of additional cliques, we want to cover the remaining edges in C 
        with as few cliques as possible. Since C is a clique, any subset of C of size ≥ 2 is 
        a valid clique. So we just need to partition the remaining edges into as few cliques as 
        possible.
        
        But since C is a clique, we can cover all remaining edges with one clique: the set of 
        all vertices in C that are incident to any remaining edge. Wait, but that clique would 
        include vertices that might not be needed. Actually, since C is a clique, the entire set 
        of vertices incident to remaining edges forms a clique, so we can cover all remaining 
        edges with one clique.
        
        But wait, that's not right either. If the remaining edges form a subgraph of C, and C 
        is a clique, then the vertex set of that subgraph induces a clique in C. So yes, one 
        clique suffices for all remaining edges.
        
        So the algorithm is:
        1. For each i ∈ I with N(i) ≠ ∅, add clique {i} ∪ N(i).
        2. Find all edges (u,v) with u,v ∈ C that are not covered by any clique from step 1.
        3. If there are remaining edges, add one clique consisting of all vertices in C that 
           are incident to remaining edges.
        
        Let's implement this and see.
        """
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)
        
        # Step 1: For each i in I, add clique {i} ∪ N(i) if N(i) is non-empty
        cliques = []
        covered_edges = set()
        
        for i in range(k, n):
            neighbors_in_C = adj[i] & set(range(k))
            if neighbors_in_C:
                clique = [i] + sorted(neighbors_in_C)
                cliques.append(clique)
                # Mark all edges in this clique as covered
                for a in range(len(clique)):
                    for b in range(a + 1, len(clique)):
                        u, v = clique[a], clique[b]
                        if u < v:
                            covered_edges.add((u, v))
                        else:
                            covered_edges.add((v, u))
        
        # Step 2: Find remaining edges in C
        remaining = []
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_edges:
                    remaining.append((u, v))
        
        # Step 3: If there are remaining edges, add one clique
        if remaining:
            vertices_in_remaining = set()
            for u, v in remaining:
                vertices_in_remaining.add(u)
                vertices_in_remaining.add(v)
            if len(vertices_in_remaining) >= 2:
                cliques.append(sorted(vertices_in_remaining))
        
        # Convert to lists of ints
        result = []
        for clique in cliques:
            result.append([int(v) for v in clique])
        
        return result

    # Process the three instances
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if time.time() > deadline:
            break
        result.append(get_cliques_for_instance(*spec))
    
    return result
# EVOLVE-BLOCK-END
