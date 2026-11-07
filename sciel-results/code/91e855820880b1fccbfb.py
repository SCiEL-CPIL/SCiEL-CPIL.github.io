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

    def clique_partition(edges, n, k):
        """
        Clique partition for split graphs using a smarter grouping strategy.
        
        Key insight: Instead of creating separate cliques for each cross edge,
        we can group multiple independent vertices that share the same neighborhood 
        in K into a single clique with that neighborhood.
        
        For a subset S of independent vertices where all have the same neighbor set N in K,
        {N} ∪ S forms a clique (since N is a clique in K, and each v in S is adjacent to 
        all vertices in N, and S is independent so no edges within S to worry about).
        
        This covers all edges between N and S with one clique instead of |N|*|S| edges.
        """
        # Build adjacency: for each vertex in I, find its neighbors in K
        neighbors_in_k = {}  # v -> frozenset of neighbors in K
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                # Check if (u, v) is an edge (u < v since u < k <= v)
                if (u, v) in set(edges) or (v, u) in set(edges):
                    nbrs.add(u)
            neighbors_in_k[v] = frozenset(nbrs)
        
        # Group independent vertices by their neighbor set in K
        from collections import defaultdict
        groups = defaultdict(list)
        for v, nbrs in neighbors_in_k.items():
            if len(nbrs) >= 1:  # Only vertices with at least one neighbor
                groups[nbrs].append(v)
        
        cliques = []
        
        # For each group, create a clique with the shared neighborhood plus the independent vertices
        # This covers all cross edges for that group
        for nbrs, independent_verts in groups.items():
            if len(nbrs) == 1:
                # Single neighbor u: clique is {u} ∪ independent_verts
                u = list(nbrs)[0]
                cliques.append([u] + independent_verts)
            elif len(nbrs) >= 2:
                # Multiple neighbors: clique is nbrs ∪ independent_verts
                cliques.append(list(nbrs) + independent_verts)
        
        # Now we need to cover edges within K that are NOT covered by the above cliques.
        # An edge (u, w) in K is covered if there exists some group where both u and w 
        # are in the neighborhood set.
        
        # Build a set of K-edges that are covered
        covered_k_edges = set()
        for nbrs, independent_verts in groups.items():
            nbr_list = list(nbrs)
            for i in range(len(nbr_list)):
                for j in range(i + 1, len(nbr_list)):
                    covered_k_edges.add((min(nbr_list[i], nbr_list[j]), 
                                        max(nbr_list[i], nbr_list[j])))
        
        # Find K-edges that are NOT covered
        # All K-edges are (u, v) for 0 <= u < v < k
        all_k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        uncovered_k_edges = [(u, v) for u, v in all_k_edges if (u, v) not in covered_k_edges]
        
        # For uncovered K-edges, we need to add cliques to cover them.
        # Simplest: each uncovered edge as its own clique.
        for u, v in uncovered_k_edges:
            cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        cliques = clique_partition_with_set(edges, edge_set, n, k)
        results.append(cliques)
        
    return results

def clique_partition_with_set(edges, edge_set, n, k):
    """
    Clique partition using edge_set for O(1) lookups.
    """
    # Build adjacency: for each vertex in I, find its neighbors in K
    neighbors_in_k = {}
    for v in range(k, n):
        nbrs = set()
        for u in range(k):
            if (u, v) in edge_set:
                nbrs.add(u)
        neighbors_in_k[v] = frozenset(nbrs)
    
    # Group independent vertices by their neighbor set in K
    from collections import defaultdict
    groups = defaultdict(list)
    for v, nbrs in neighbors_in_k.items():
        if len(nbrs) >= 1:
            groups[nbrs].append(v)
    
    cliques = []
    
    # For each group, create a clique
    for nbrs, independent_verts in groups.items():
        if len(nbrs) == 1:
            u = list(nbrs)[0]
            cliques.append([u] + independent_verts)
        elif len(nbrs) >= 2:
            cliques.append(list(nbrs) + independent_verts)
    
    # Find uncovered K-edges
    covered_k_edges = set()
    for nbrs, independent_verts in groups.items():
        nbr_list = list(nbrs)
        for i in range(len(nbr_list)):
            for j in range(i + 1, len(nbr_list)):
                covered_k_edges.add((min(nbr_list[i], nbr_list[j]), 
                                    max(nbr_list[i], nbr_list[j])))
    
    all_k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
    uncovered_k_edges = [(u, v) for u, v in all_k_edges if (u, v) not in covered_k_edges]
    
    for u, v in uncovered_k_edges:
        cliques.append([u, v])
    
    return cliques
# EVOLVE-BLOCK-END
