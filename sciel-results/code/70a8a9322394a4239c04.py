"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(edges, n, k):
        """
        Valid clique partition for split graphs using maximum clique extraction.
        
        Strategy:
        1. One clique for the entire K (covers all edges in K).
        2. For cross edges, repeatedly find a maximum clique in the remaining
           bipartite graph and remove it, until no edges remain.
        
        This groups cross-edges into larger cliques, reducing total count.
        """
        cliques = []
        
        # 1. The clique K
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. Collect cross edges
        cross_edges = set()
        for u, v in edges:
            if u < k and v >= k:
                cross_edges.add((u, v))
        
        # Build adjacency for cross graph: left side = [0, k), right side = [k, n)
        # left_adj[u] = set of right vertices connected to u
        # right_adj[v] = set of left vertices connected to v
        left_adj = defaultdict(set)
        right_adj = defaultdict(set)
        for u, v in cross_edges:
            left_adj[u].add(v)
            right_adj[v].add(u)
        
        # Repeatedly find a maximum clique in the remaining bipartite graph
        # A clique in a bipartite graph is at most size 2 (one from each side)
        # Wait - that's not right. In a split graph, the cross edges form a bipartite graph.
        # But a clique in the full graph can include multiple left vertices and one right vertex
        # if all those left vertices are connected to that right vertex.
        # Or multiple right vertices and one left vertex if all those right vertices are connected to that left vertex.
        # Actually, since left vertices form a clique and right vertices form an independent set,
        # a clique in the full graph is: a subset of left vertices + a subset of right vertices
        # where every left-right pair in the subset is an edge.
        # Since right vertices are independent, we can have at most... wait, we CAN have multiple right vertices
        # as long as they're all connected to all left vertices in the clique.
        
        # So a clique = L_subset ∪ R_subset where L_subset ⊆ [0,k), R_subset ⊆ [k,n),
        # and for all u in L_subset, v in R_subset: (u,v) is an edge.
        
        # Greedy approach: repeatedly find a large clique and remove its edges.
        # A clique is determined by choosing L_subset and R_subset such that
        # R_subset ⊆ intersection of neighbors of all vertices in L_subset.
        
        while cross_edges:
            # Try to find a good clique by picking a left vertex with high degree
            # or a right vertex with high degree
            best_clique = None
            best_size = 0
            
            # Option 1: Pick a left vertex u, include all right vertices connected to u
            # that are also connected to other left vertices we want to include
            # Simple approach: for each left vertex u, the clique {u} ∪ N(u) is valid
            # But we can do better: pick a set of left vertices that share common neighbors
            
            # Greedy: find a left vertex with max degree, then find common neighbors
            max_left = None
            max_left_deg = 0
            for u in list(left_adj.keys()):
                if len(left_adj[u]) > max_left_deg:
                    max_left_deg = len(left_adj[u])
                    max_left = u
            
            if max_left is not None and max_left_deg > 0:
                # Start with {max_left} ∪ N(max_left)
                # Can we add more left vertices? A left vertex u can be added if
                # N(u) ⊇ N(max_left) (i.e., u is connected to all right vertices in N(max_left))
                # Actually, for the clique to remain valid, all left vertices must be connected
                # to all right vertices in the clique.
                # So if we have R = N(max_left), we can add u if N(u) ⊇ R.
                
                R = set(left_adj[max_left])
                L = {max_left}
                
                # Try to add other left vertices whose neighbor set contains R
                for u in list(left_adj.keys()):
                    if u != max_left and left_adj[u].issuperset(R):
                        L.add(u)
                
                clique = list(L) + list(R)
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            # Option 2: Pick a right vertex v, include all left vertices connected to v
            max_right = None
            max_right_deg = 0
            for v in list(right_adj.keys()):
                if len(right_adj[v]) > max_right_deg:
                    max_right_deg = len(right_adj[v])
                    max_right = v
            
            if max_right is not None and max_right_deg > 0:
                L = set(right_adj[max_right])
                R = {max_right}
                
                # Try to add other right vertices whose neighbor set contains L
                for v in list(right_adj.keys()):
                    if v != max_right and right_adj[v].issuperset(L):
                        R.add(v)
                
                clique = list(L) + list(R)
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or best_size < 2:
                # Fallback: take one edge
                u, v = next(iter(cross_edges))
                best_clique = [u, v]
                best_size = 2
            
            # Add the clique
            cliques.append(best_clique)
            
            # Remove all edges within this clique from cross_edges
            L = [x for x in best_clique if x < k]
            R = [x for x in best_clique if x >= k]
            for u in L:
                for v in R:
                    cross_edges.discard((u, v))
            
            # Update adjacency
            for u in L:
                for v in R:
                    if v in left_adj.get(u, set()):
                        left_adj[u].discard(v)
                    if u in right_adj.get(v, set()):
                        right_adj[v].discard(u)
            
            # Clean up empty adjacencies
            for u in list(left_adj.keys()):
                if not left_adj[u]:
                    del left_adj[u]
            for v in list(right_adj.keys()):
                if not right_adj[v]:
                    del right_adj[v]
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
