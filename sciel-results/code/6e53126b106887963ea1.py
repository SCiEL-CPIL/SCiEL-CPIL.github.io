import time
import random
import sys

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        # Use integer bitmasks for adjacency for fast intersection checks
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def is_clique(vertices, adj):
        # Check if all pairs in vertices are connected in adj
        # vertices is a list of ints
        if len(vertices) <= 1:
            return True
        for i in range(len(vertices)):
            u = vertices[i]
            # Check if u is connected to all other vertices in the list
            # We can check if adj[u] contains all other vertices
            # A fast way: for each v in vertices[i+1:], check if v is in adj[u]
            for j in range(i + 1, len(vertices)):
                v = vertices[j]
                if not (adj[u] & (1 << v)):
                    return False
        return True

    def clique_partition_ils(adj, n, time_limit):
        """
        Finds a clique partition minimizing the number of cliques.
        Returns a list of cliques (each a list of vertices).
        """
        start_time = time.time()
        
        # 1. Initial Solution: Each edge is a clique.
        # This is valid but has the maximum number of cliques.
        # We can do a bit better by merging edges that form triangles, etc.
        # But let's start with the trivial partition and let ILS merge them.
        
        # Actually, a better initialization: 
        # Greedily merge edges into cliques.
        # Start with each edge as a clique.
        # Then try to merge cliques.
        
        # Let's represent the partition as a list of sets of vertices.
        # To make it hashable and comparable, we use tuples of sorted vertices.
        
        # Initialize: list of (u, v) for all edges
        cliques = []
        for u in range(n):
            mask = adj[u]
            while mask:
                # Extract lowest set bit
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                mask ^= lsb
                if v > u: # Avoid duplicates
                    cliques.append([u, v])
        
        if not cliques:
            return []

        # Convert to a structure that allows fast lookup and merging.
        # We need to track which edges are covered by which clique.
        # But for merging, we just need to check if union is a clique.
        
        # State: list of cliques (lists of vertices)
        # We need to be able to quickly check if two cliques can be merged.
        # Two cliques C1 and C2 can be merged if C1 U C2 is a clique.
        # This requires all vertices in C1 to be connected to all vertices in C2.
        
        def can_merge(c1, c2):
            # c1, c2 are lists of vertices
            for u in c1:
                for v in c2:
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        # To speed up can_merge, we can precompute or use bitmasks.
        # Let's represent each clique by its set of vertices.
        
        # ILS Loop
        best_cliques = [list(c) for c in cliques]
        best_len = len(best_cliques)
        current_cliques = [list(c) for c in cliques]
        current_len = len(current_cliques)
        
        # We need a way to quickly find mergeable pairs.
        # Instead of checking all pairs, we can maintain a list of candidate pairs.
        # Or, since n is small (32, 64), checking all pairs O(K^2 * |C|^2) might be okay if K is small.
        # Initially K is ~n^2/2, which is large. Merging will reduce K quickly.
        
        # Let's use a more efficient representation:
        # Map each vertex to the clique index it belongs to? No, a vertex can be in multiple cliques?
        # NO. The task says "every edge must lie in exactly one listed clique".
        # This implies the cliques are edge-disjoint.
        # It does NOT imply vertex-disjoint. A vertex can be in multiple cliques, as long as the edges are distinct.
        # However, if two cliques share a vertex, say u, and C1={u,a,b}, C2={u,c,d}, they can only be merged if {a,b,c,d} form a clique with u.
        
        # Wait, if C1 and C2 share a vertex u, and we merge them, the new clique is C1 U C2.
        # The edges in C1 U C2 are all pairs in C1 U C2.
        # The edges in C1 are pairs in C1.
        # The edges in C2 are pairs in C2.
        # If C1 and C2 share only u, the edges in C1 U C2 include edges between C1\{u} and C2\{u}.
        # These edges were NOT in C1 or C2.
        # So merging two cliques that share a vertex adds NEW edges to the covered set!
        # But the task requires that every edge in the graph is covered EXACTLY ONCE.
        # If we merge C1 and C2, we cover edges(C1 U C2).
        # If edges(C1 U C2) is a superset of edges(C1) U edges(C2), then we are covering extra edges.
        # Those extra edges must not have been covered by any other clique.
        # And they must exist in the graph.
        
        # This complicates things significantly.
        # "Every edge must lie in exactly one listed clique."
        # This is a partition of the EDGE SET.
        # So, if we have a partition {C1, C2, ..., Ck}, then for any edge e, there is exactly one Ci such that e is an edge of Ci.
        # Note: An edge e=(u,v) is an edge of a clique C if u in C and v in C.
        # So, if u in C1 and v in C1, then (u,v) is covered by C1.
        # If u in C1 and v in C2, and C1 != C2, then (u,v) is NOT covered by C1 (unless v in C1) and NOT covered by C2 (unless u in C2).
        # If (u,v) is in the graph, it MUST be covered by some clique.
        # So, if u in C1 and v in C2, and C1 != C2, then (u,v) cannot be in the graph, OR it must be covered by another clique C3 containing both u and v.
        # But if C3 contains u and v, then (u,v) is covered by C3.
        # But wait, if C1 contains u and C2 contains v, and C1 != C2, does that prevent (u,v) from being in the graph?
        # No. It just means (u,v) is not covered by C1 or C2. It must be covered by some other clique.
        
        # This means cliques CAN share vertices.
        # Example: Graph is a triangle (0,1,2).
        # Partition: C1={0,1}, C2={1,2}, C3={0,2}.
        # Edges: (0,1) in C1, (1,2) in C2, (0,2) in C3.
        # Each edge is in exactly one clique. Valid.
        # Can we merge?
        # If we merge C1 and C2, we get {0,1,2}. This is a clique.
        # The edges in {0,1,2} are (0,1), (1,2), (0,2).
        # If we replace C1, C2 with {0,1,2}, we now cover (0,1), (1,2), (0,2) with this single clique.
        # But (0,2) was previously covered by C3.
        # So now (0,2) is covered by BOTH {0,1,2} and C3. Invalid.
        # So we can only merge C1 and C2 if the new clique {0,1,2} does NOT contain any edges that are covered by other cliques.
        # i.e., for any pair (u,v) in C1 U C2, if (u,v) is covered by some other clique Cj (j not 1,2), then we cannot merge.
        
        # This is a very complex constraint.
        # Let's re-read the task carefully.
        # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        # Yes, edge partition.
        
        # This makes "merging" two cliques dangerous because the union clique covers edges that might be covered by other cliques.
        # Specifically, if C1 and C2 are disjoint (no common vertices), then edges(C1 U C2) = edges(C1) U edges(C2) U edges_between(C1, C2).
        # The edges_between(C1, C2) are new edges. If any of these exist in the graph, they must be covered by some other clique.
        # If they are not covered by any other clique, then we can merge C1 and C2, provided that C1 U C2 is a clique (which implies edges_between exist in graph).
        # Wait, if C1 U C2 is a clique, then ALL pairs between C1 and C2 are edges in the graph.
        # So if we merge, we cover those edges.
        # If those edges were already covered by other cliques, it's invalid.
        # So we can merge C1 and C2 if:
        # 1. C1 U C2 is a clique.
        # 2. No edge between C1 and C2 is covered by any other clique Cj (j != 1, 2).
        
        # This is hard to check efficiently.
        
        # Alternative Strategy:
        # Instead of merging existing cliques, let's build the partition from scratch using a different method.
        # Or, use a different local search move:
        # "Swap" an edge from one clique to another? No, edges are fixed in cliques.
        
        # Let's try a different approach: **Greedy with Backtracking / Local Search on Vertices**.
        
        # Actually, the most effective strategy for edge clique partition is often related to **Line Graph Clique Cover**.
        # The edges of G are vertices of L(G).
        # A clique in G corresponds to a clique in L(G) (specifically, a star or a triangle in G corresponds to a clique in L(G)).
        # Wait, a clique in G of size k corresponds to a clique of size k(k-1)/2 in L(G).
        # The problem is to partition the vertices of L(G) into cliques, such that each clique in L(G) corresponds to a clique in G.
        # Not every clique in L(G) corresponds to a clique in G. Only those that are "stars" or "triangles" (or larger cliques in G).
        # Actually, a set of edges in G forms a clique in G if and only if they are all incident to a common vertex (star) OR they form a triangle (or larger).
        # Wait, if edges form a clique in G, they must be mutually adjacent in L(G).
        # In L(G), two edges are adjacent if they share a vertex in G.
        # A set of edges E' is a clique in L(G) if every pair of edges in E' shares a vertex in G.
        # This is equivalent to E' being a star in G OR E' being a triangle in G (if |E'|=3).
        # If |E'| > 3, can it be a clique in L(G) without being a star or triangle?
        # Example: K4 in G. Edges: 6. In L(K4), all 6 edges are pairwise adjacent?
        # Edge (1,2) and (3,4) do NOT share a vertex in K4. So they are not adjacent in L(K4).
        # So the 6 edges of K4 do NOT form a clique in L(K4).
        # The cliques in L(G) are exactly the stars and triangles of G.
        # So the problem is to partition the edges of G into stars and triangles such that every edge is in exactly one star/triangle.
        # Wait, a clique in G of size k > 3 contains many stars and triangles.
        # If we list a clique C of size 4 as a single item in the partition, it covers all 6 edges.
        # Is that allowed? "Every listed set must be a clique". Yes.
        # "Every edge must lie in exactly one listed clique". Yes.
        # So we are partitioning the edge set into cliques of G.
        
        # So my previous "merge" logic was correct but complex.
        
        # Let's try a simpler heuristic:
        # 1. Find a large clique in G.
        # 2. Remove all its edges.
        # 3. Repeat.
        # This is the current program's strategy!
        # The current program does this but with a poor greedy clique finder.
        
        # The key to improvement is a **Better Clique Finder**.
        # The current program uses a simple greedy expansion.
        # We should use **Bron-Kerbosch with pivot** to find the **Maximum Clique** (or near-maximum) in the remaining graph.
        # Removing the maximum clique is likely to minimize the number of cliques.
        
        # Let's implement Bron-Kerbosch with pivot to find the largest clique in the remaining graph.
        
        def bron_kerbosch_max_clique(adj, n):
            """
            Finds the maximum clique in the graph defined by adj.
            Returns the clique as a list of vertices.
            """
            best_clique = []
            
            # We need to handle the graph dynamically (edges removed).
            # So we pass the current adjacency matrix (bitmasks).
            
            def bk(R, P, X):
                nonlocal best_clique
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                if len(R) + P.bit_count() <= len(best_clique):
                    return # Pruning
                
                # Pivot selection
                # Choose u from P U X to minimize |P \ N(u)|
                # For simplicity, pick a random vertex from P U X or the one with max degree in P
                if P | X:
                    # Pick pivot u
                    # Let's just pick the first vertex in P for now, or random
                    # Better: pick u that maximizes |P & ~N(u)| ? No, minimizes |P & N(u)|?
                    # Standard pivot: choose u in P U X to minimize |P \ N(u)|.
                    # This is equivalent to maximizing |P & N(u)|? No.
                    # P \ N(u) are the vertices in P NOT connected to u.
                    # We want to minimize this set.
                    # So we want u to be connected to as many vertices in P as possible.
                    
                    # Let's try a few candidates for pivot
                    candidates = []
                    temp = P | X
                    while temp:
                        lsb = temp & (-temp)
                        u = lsb.bit_length() - 1
                        temp ^= lsb
                        # Calculate |P \ N(u)|
                        # N(u) is adj[u]
                        # P \ N(u) = P & ~adj[u]
                        # We want to minimize this count
                        # But we need to be careful with bitmasks beyond n
                        # adj[u] only has bits for 0..n-1
                        not_n_u = (~adj[u]) & ((1 << n) - 1)
                        count = (P & not_n_u).bit_count()
                        candidates.append((count, u))
                    
                    if candidates:
                        candidates.sort()
                        pivot = candidates[0][1]
                    else:
                        return
                else:
                    return
                
                # Recurse on P \ N(pivot)
                # Wait, the standard algorithm recurses on P \ N(u) for pivot u.
                # But we need to ensure we cover all vertices.
                # The standard BK finds all maximal cliques.
                # To find the MAXIMAL clique, we can stop early if we find a large one?
                # No, we need to explore.
                
                # Actually, for Maximum Clique, we can use a branch and bound.
                # Let's use a simpler approach:
                # Greedy expansion from each vertex, but with backtracking?
                
                # Given time constraints, let's use a simpler heuristic for finding a large clique:
                # 1. Pick a vertex with high degree.
                # 2. Greedily add vertices that are connected to all current clique members.
                # 3. Try this from multiple starting vertices.
                
                # This is what the current program does.
                # To improve, we can randomize the order and try many times.
                
                pass # Placeholder

            # Let's implement a randomized greedy clique finder with many restarts.
            def find_large_clique(adj, n, num_restarts=100):
                best = []
                for _ in range(num_restarts):
                    # Pick a random starting vertex
                    if not any(adj[u] for u in range(n)):
                        break
                    start = random.randint(0, n-1)
                    if not adj[start]:
                        continue
                    
                    clique = {start}
                    candidates = adj[start]
                    
                    while candidates:
                        # Pick a candidate with high degree in the subgraph induced by candidates
                        # Or just random
                        # Let's pick the one with highest degree in the original graph
                        # To be faster, just pick random
                        lsb = candidates & (-candidates)
                        v = lsb.bit_length() - 1
                        candidates ^= lsb
                        
                        # Check if v is connected to all in clique
                        # Since candidates are neighbors of start, and we update candidates by intersecting with neighbors of v
                        # This greedy method ensures clique property
                        clique.add(v)
                        # Update candidates: must be neighbors of v
                        candidates &= adj[v]
                    
                    if len(clique) > len(best):
                        best = list(clique)
                return best

            # Use this in the main loop
            return find_large_clique(adj, n, 50)

        # Main ILS Loop for Clique Partition
        # 1. Start with empty partition.
        # 2. While edges remain:
        #    a. Find a large clique in the remaining graph.
        #    b. Add it to partition.
        #    c. Remove its edges.
        # 3. Local Search: Try to replace a clique with two smaller ones? No, we want to MINIMIZE count.
        #    So we want to MERGE cliques.
        #    But merging is hard.
        #    Alternative: Try to find a clique that covers edges from MULTIPLE existing cliques?
        #    This is equivalent to repartitioning.
        
        # Let's stick to the "Find Max Clique, Remove" strategy but with a very good Max Clique finder.
        # The current program's weakness is the clique finder.
        
        adj_remaining = [adj[u] for u in range(n)]
        cliques = []
        
        while True:
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if adj_remaining[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find a large clique
            clique = find_large_clique(adj_remaining, n, 20)
            
            if not clique or len(clique) < 2:
                # Fallback: take a single edge
                # Find any edge
                for u in range(n):
                    if adj_remaining[u]:
                        lsb = adj_remaining[u] & (-adj_remaining[u])
                        v = lsb.bit_length() - 1
                        clique = [u, v]
                        break
            
            if not clique:
                break
                
            cliques.append(clique)
            
            # Remove edges of clique
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i+1, len(clique)):
                    v = clique[j]
                    adj_remaining[u] &= ~(1 << v)
                    adj_remaining[v] &= ~(1 << u)
                    
        return cliques

    # Generate graphs
    adj1 = generate_graph(32, 0.6, 601)
    adj2 = generate_graph(64, 0.55, 602)
    
    start_time = time.time()
    deadline = start_time + 140
    
    best_result_1 = None
    best_result_2 = None
    best_len_1 = float('inf')
    best_len_2 = float('inf')
    
    # Run multiple times with different random seeds for the clique finder
    seed_counter = 0
    while time.time() < deadline:
        # Instance 1
        random.seed(seed_counter)
        cliques_1 = clique_partition_ils(adj1, 32, time.time())
        if len(cliques_1) < best_len_1:
            best_len_1 = len(cliques_1)
            best_result_1 = cliques_1
            
        # Instance 2
        random.seed(seed_counter + 1000)
        cliques_2 = clique_partition_ils(adj2, 64, time.time())
        if len(cliques_2) < best_len_2:
            best_len_2 = len(cliques_2)
            best_result_2 = cliques_2
            
        seed_counter += 1
        
        # If we've found a small number, we can stop earlier?
        # Theoretical lower bound: M / (k*(k-1)/2) where M is edges, k is max clique size.
        # For n=32, p=0.6, M ~ 32*31/2 * 0.6 ~ 297.
        # Max clique in G(32, 0.6) is around 7-8?
        # 7*6/2 = 21 edges. 297/21 ~ 14 cliques.
        # If we get below 15, it's good.
        
    if best_result_1 is None:
        best_result_1 = []
    if best_result_2 is None:
        best_result_2 = []
        
    return [best_result_1, best_result_2]

