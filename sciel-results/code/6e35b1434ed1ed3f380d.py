"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        """
        Strategy:
        1. For each i in I, form clique K_i = {i} U N(i).
           This covers all cross edges incident to i.
           It also covers C-C edges within N(i).
        2. Identify C-C edges not covered by any K_i.
        3. Cover these remaining C-C edges with a minimal clique cover 
           of the subgraph induced by these edges on C.
        """
        # Identify C and I
        C = set(range(k))
        I = set(range(k, n))
        
        # Build adjacency for cross edges
        # N(i) for i in I
        N = {i: set() for i in I}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif u >= k and v < k:
                N[u].add(v)
        
        cliques = []
        
        # 1. Create cliques for I vertices
        covered_cc_edges = set()
        for i in I:
            if N[i]:
                # Clique is {i} U N(i)
                clique_vertices = [i] + sorted(list(N[i]))
                cliques.append(clique_vertices)
                
                # Mark C-C edges within N(i) as covered
                neigh = list(N[i])
                for a_idx in range(len(neigh)):
                    for b_idx in range(a_idx + 1, len(neigh)):
                        u, v = neigh[a_idx], neigh[b_idx]
                        if u > v: u, v = v, u
                        covered_cc_edges.add((u, v))
        
        # 2. Identify remaining C-C edges
        all_cc_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_cc_edges.add((u, v))
        
        remaining_edges = all_cc_edges - covered_cc_edges
        
        # 3. Cover remaining edges with cliques
        # The remaining edges form a graph G_rem on vertices C.
        # We need to cover its edges with cliques.
        # Greedy approach:
        while remaining_edges:
            # Pick an arbitrary edge
            # To be deterministic, pick the smallest one
            edge = min(remaining_edges)
            u, v = edge
            
            # Find a maximal clique in G_rem containing u and v
            # Start with {u, v}
            clique = {u, v}
            
            # Try to add other vertices from C
            # We can iterate through all vertices in C
            for w in C:
                if w in clique:
                    continue
                # Check if w can be added to clique
                # w must be connected to all vertices in clique via edges in remaining_edges
                can_add = True
                for x in clique:
                    e1, e2 = (x, w) if x < w else (w, x)
                    if (e1, e2) not in remaining_edges:
                        can_add = False
                        break
                if can_add:
                    clique.add(w)
            
            # Add this clique to the partition
            cliques.append(sorted(list(clique)))
            
            # Remove all edges within this clique from remaining_edges
            clique_list = list(clique)
            for a_idx in range(len(clique_list)):
                for b_idx in range(a_idx + 1, len(clique_list)):
                    e1, e2 = clique_list[a_idx], clique_list[b_idx]
                    if e1 > e2: e1, e2 = e2, e1
                    if (e1, e2) in remaining_edges:
                        remaining_edges.remove((e1, e2))
                        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
