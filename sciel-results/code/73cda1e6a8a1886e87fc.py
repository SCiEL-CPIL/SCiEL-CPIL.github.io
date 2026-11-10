import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a K3,3-free graph on 20 vertices maximizing the number of edges.
    
    Strategy:
    1. Use a deterministic construction based on projective planes or finite geometry 
       to get a high-quality initial graph. Specifically, we can model the vertices
       as points and lines in a finite geometry, or use a greedy heuristic with 
       a strong initial seed.
       
    2. A more robust approach for n=20: 
       - Start with a random graph or an empty graph.
       - Use Local Search (Simulated Annealing or Hill Climbing) to maximize edges 
         while maintaining the K3,3-free property.
       - The move is to add an edge if it doesn't create a K3,3, or to swap edges 
         (remove one, add another) to increase density.
       
    3. Since n=20 is small, we can afford a more expensive check for K3,3.
       A graph is K3,3-free if and only if for every pair of disjoint sets A, B of size 3,
       not all 9 edges between them are present.
       
       However, checking all pairs of triples is O(n^6) which is 20^6 ~ 64M, which is 
       borderline in Python if done naively many times. 
       
       Better: Use the property that K3,3 is a minor or use a fast subgraph isomorphism 
       check. But for n=20, we can precompute common neighborhoods.
       
       Actually, the condition |N(u) ∩ N(v)| <= 2 for all pairs (u,v) is sufficient 
       but not necessary. The current program uses this. 
       
       Let's try a different strategy: 
       **Construct a graph from a known dense K3,3-free graph structure.**
       
       The extremal number ex(n, K3,3) is known to be O(n^(5/3)). For n=20, 
       n^(5/3) ≈ 20^1.666 ≈ 100. 
       
       Let's look at the seed score 1.4444. The seed program (p3) returned K2,18 which 
       has 18 edges. 18 * 1.4444 ≈ 26. So the current best is around 26 edges.
       
       Wait, the score is value/seed. Seed value is 18. 
       Current score 1.4444 -> 18 * 1.4444 = 26 edges.
       
       Can we get more?
       
       Let's try a greedy algorithm that does NOT enforce |N(u) ∩ N(v)| <= 2, 
       but instead checks for actual K3,3 subgraphs. This allows for denser graphs.
       
       How to check if adding edge (u,v) creates a K3,3?
       A new K3,3 must use the new edge (u,v).
       So (u,v) is part of the K3,3.
       Case 1: u and v are in the same part of the bipartition.
               Then they share 3 neighbors in the other part.
               So we need to check if |N(u) ∩ N(v)| >= 3. If so, and if there exist 
               3 common neighbors w1, w2, w3 such that they are not connected to each other 
               (wait, in K3,3 the vertices in the same part are not connected).
               Actually, if u and v are in the same part, they are NOT connected to each other 
               in the K3,3 subgraph. But we are ADDING the edge (u,v). So u and v CANNOT be 
               in the same part of the K3,3 if the edge (u,v) is used as an edge in the K3,3.
               
       Case 2: u and v are in different parts of the bipartition.
               Let Part A contain u and Part B contain v.
               Then there must be 2 other vertices a1, a2 in Part A and 2 other vertices 
               b1, b2 in Part B such that:
               - u is connected to b1, b2, v
               - a1, a2 are connected to b1, b2, v
               - u, a1, a2 are connected to b1, b2, v? No.
               The edges are all between A and B.
               So:
               u (in A) is connected to v (in B), b1 (in B), b2 (in B).
               a1 (in A) is connected to v (in B), b1 (in B), b2 (in B).
               a2 (in A) is connected to v (in B), b1 (in B), b2 (in B).
               
               So, for a fixed edge (u,v), a K3,3 containing it exists if:
               There exist a1, a2 such that:
               1. a1, a2 are not u, v, and are distinct.
               2. a1, a2 are NOT connected to u or v? No, in K3,3 vertices in the same part 
                  are not connected. So a1, a2 must not be connected to u? 
                  Wait, u is in A, a1 is in A. So u and a1 are not connected in the K3,3.
                  But in the graph, they might be connected. The subgraph just needs the 
                  9 edges. It doesn't forbid other edges.
                  
               So the condition is:
               There exist a1, a2 in V \ {u, v} and b1, b2 in V \ {u, v} such that:
               - {u, a1, a2} are in Part A
               - {v, b1, b2} are in Part B
               - Edges present: 
                 (u,v), (u,b1), (u,b2),
                 (a1,v), (a1,b1), (a1,b2),
                 (a2,v), (a2,b1), (a2,b2)
               
               This is a specific subgraph pattern.
               
               To check if adding (u,v) creates a K3,3, we can check:
               For all pairs (a1, a2) in V \ {u, v}:
                 If a1 and a2 are both connected to v (in the current graph), 
                 then we need to check if there exist b1, b2 such that:
                 - b1, b2 are connected to u, a1, a2.
                 - Specifically, b1, b2 must be in N(u) ∩ N(a1) ∩ N(a2).
                 - And b1, b2 must be distinct and not u, v, a1, a2.
                 
               So:
               Let C = N(u) ∩ N(a1) ∩ N(a2).
               If |C| >= 2, then we can pick b1, b2 from C.
               But we also need b1, b2 to be connected to v.
               Wait, a1 and a2 are connected to v by assumption.
               Do b1, b2 need to be connected to v?
               In the K3,3, b1 and b2 are in Part B. v is in Part B.
               So b1 and b2 are NOT connected to v in the K3,3 subgraph.
               But in the graph, they might be.
               
               So the condition is simply:
               There exist a1, a2 such that:
               1. a1, a2 are connected to v.
               2. |N(u) ∩ N(a1) ∩ N(a2) \ {v}| >= 2.
               
               If this holds, then we have a K3,3.
               
               So the check for adding (u,v) is:
               For all pairs (a1, a2) in N(v) \ {u}:
                 if len(N(u) & N(a1) & N(a2) - {v}) >= 2:
                   return False
               
               This is O(n^2) per edge addition, with set operations.
               n=20, so 20^2 = 400 checks. Each check is a set intersection of size <= 20.
               Very fast.
               
               Let's implement this greedy approach with the correct K3,3 check.
    """
    n = 20
    deadline = time.time() + 145
    
    # Multiple random restarts to find a better solution
    best_edges = []
    best_count = 0
    
    # Number of restarts
    num_restarts = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        adj = [set() for _ in range(n)]
        edges = []
        
        # Randomly shuffle edge order
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 2:
                break
                
            # Check if adding (u, v) creates a K3,3
            valid = True
            
            # We need to check if there exist a1, a2 in N(v) \ {u} such that
            # |N(u) ∩ N(a1) ∩ N(a2) \ {v}| >= 2
            
            # To speed up, we can precompute common neighbors?
            # For now, direct check.
            for a1 in adj[v]:
                if a1 == u:
                    continue
                for a2 in adj[v]:
                    if a2 <= a1: # avoid duplicates and self
                        continue
                    # Check common neighbors of u, a1, a2
                    # common = adj[u] & adj[a1] & adj[a2]
                    # We need at least 2 common neighbors that are not v
                    # Note: v is in adj[u] (after adding), adj[a1], adj[a2]
                    # But we are checking BEFORE adding.
                    # So v is in adj[a1] and adj[a2].
                    # Is v in adj[u]? No, not yet.
                    # So N(u) does not contain v.
                    # So N(u) ∩ N(a1) ∩ N(a2) will not contain v.
                    # So we just need len(adj[u] & adj[a1] & adj[a2]) >= 2
                    
                    common = adj[u] & adj[a1] & adj[a2]
                    if len(common) >= 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    return best_edges
# EVOLVE-BLOCK-END
