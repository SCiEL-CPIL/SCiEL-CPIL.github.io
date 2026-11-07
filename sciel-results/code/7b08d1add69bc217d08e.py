"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    """
    Solve the minimum clique partition problem for three fixed split graphs.
    
    A split graph has vertices partitioned into a clique K (size k) and an independent set I (size n-k).
    Cross edges between K and I exist with probability p.
    
    Key insight: In a split graph, every maximal clique contains the entire clique K, plus a subset of I 
    that is adjacent to all vertices in K. Wait, that's not quite right.
    
    Actually, in a split graph:
    - K is a clique
    - I is an independent set
    - Edges only exist between K and K (all), I and I (none), and K and I (some)
    
    A clique in a split graph can be:
    1. A subset of K (any subset of size >= 2)
    2. A vertex from I plus a subset of K that is adjacent to that vertex
    
    But wait, for a clique partition, we need to cover all edges with cliques such that each edge is in exactly one clique.
    
    Let's think about this more carefully. The edges are:
    - All edges within K: C(k,2) edges
    - Some edges between K and I: each vertex in I has some neighbors in K
    
    For the edges within K: we need to partition the complete graph K_k into cliques. The minimum number of cliques to partition K_k is C(k,2) if we use only edges as cliques, but we can do better by using larger cliques. Actually, the minimum clique partition of a complete graph K_k is 1 (just use K_k itself). But wait, we might not want to use K_k as a clique if it interferes with covering cross edges.
    
    Actually, let's reconsider. A clique partition means we partition the edge set into cliques. Each clique is a set of vertices where every pair is connected. The cliques in the partition must be such that every edge belongs to exactly one clique.
    
    For a split graph:
    - Edges within K must be covered by cliques that are subsets of K
    - Edges between K and I must be covered by cliques that contain one vertex from I and some vertices from K that are adjacent to it
    
    Note: A clique containing a vertex from I can only contain that vertex from I (since I is independent) and some vertices from K.
    
    So the problem decomposes:
    1. Partition the edges of K_k into cliques (subsets of K of size >= 2)
    2. For each vertex v in I, partition the edges from v to its neighbors in N(v) into cliques. Each such clique is {v} ∪ S where S ⊆ N(v) and S is a clique in K (which it always is since K is a clique).
    
    Wait, but the cliques in the partition must be such that no edge is in two cliques. Since edges within K and edges between K and I are disjoint, the two parts can be handled independently!
    
    So the total number of cliques = (min cliques to partition K_k) + (sum over v in I of min cliques to partition the star centered at v with leaves N(v))
    
    For the star part: The edges from v to N(v) form a star. To partition a star with d edges into cliques, each clique can contain at most... well, a clique {v} ∪ S where S ⊆ N(v) covers the edges from v to each vertex in S. So one clique {v, u1, u2, ..., um} covers edges (v,u1), (v,u2), ..., (v,um). 
    
    To minimize the number of cliques covering the star, we want each clique to cover as many edges as possible. Since {v} ∪ S covers |S| edges (the edges from v to each vertex in S), and S can be any subset of N(v), we can cover all d edges with just 1 clique: {v} ∪ N(v). 
    
    Wait, that's true! So for each vertex v in I, we need exactly 1 clique to cover all its incident edges: {v} ∪ N(v).
    
    But wait, we need each clique to have at least 2 vertices. If N(v) is empty, then v has no edges, so we don't need any clique for v. If N(v) is non-empty, then {v} ∪ N(v) has at least 2 vertices.
    
    So the cross edges contribute exactly |{v in I : N(v) ≠ ∅}| cliques.
    
    Now for the edges within K: We need to partition the complete graph K_k into cliques. The minimum number of cliques to partition K_k is 1 (just use K_k itself).
    
    So the total is 1 + |{v in I : N(v) ≠ ∅}|.
    
    Wait, but is this always valid? Let's check:
    - Clique 1: K (all k vertices). This covers all C(k,2) edges within K.
    - For each v in I with N(v) ≠ ∅: clique {v} ∪ N(v). This covers all edges from v to its neighbors.
    
    These cliques are edge-disjoint because:
    - K covers only edges within K
    - {v} ∪ N(v) covers only edges from v to K (specifically to N(v))
    - Different v's in I give different cliques, and their edge sets are disjoint (since each edge (v, u) is only in the clique for v)
    
    And every edge is covered:
    - Edges within K: covered by K
    - Edges (v, u) where v in I, u in K: if u in N(v), covered by {v} ∪ N(v); if u not in N(v), there's no edge
    
    So the answer is 1 + (number of vertices in I with at least one neighbor in K).
    
    Let me verify with the seed program. The seed program returns edges as cliques (each edge is a 2-vertex clique). So the seed's answer is the total number of edges.
    
    For (30, 10, 0.7, 301): k=10, n=30, so |I|=20. Expected cross edges = 20*10*0.7 = 140. Plus C(10,2)=45 edges within K. Total edges ≈ 185.
    My formula: 1 + (number of vertices in I with at least one neighbor). With p=0.7, probability a vertex in I has no neighbor = (1-0.7)^10 = 0.3^10 ≈ 0.000059. So expected number of isolated vertices in I ≈ 20 * 0.000059 ≈ 0.0012. So expected answer ≈ 1 + 20 = 21.
    
    That's a huge improvement from ~185 to ~21!
    
    Let me code this up.
    """
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build adjacency for vertices in I to K
        # Vertex i in I (i from k to n-1) has neighbors in K (0 to k-1)
        neighbors_in_I = []
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors_in_I.append((v, nbrs))
        
        # Count vertices in I with at least one neighbor
        non_isolated = sum(1 for v, nbrs in neighbors_in_I if len(nbrs) > 0)
        
        # Build the cliques
        cliques = []
        
        # Clique 1: the entire K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each vertex in I with neighbors, add clique {v} ∪ N(v)
        for v, nbrs in neighbors_in_I:
            if len(nbrs) > 0:
                clique = [v] + nbrs
                cliques.append(clique)
        
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        result.append(instance(*spec))
    
    return result

# EVOLVE-BLOCK-END
