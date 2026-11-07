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
        1. Generate all cross-edges between K and I.
        2. Group I-vertices by their neighborhood in K.
        3. For each group with identical neighborhoods, if the neighborhood has size >= 2,
           the union of the neighborhood and the group forms a clique (since K is a clique
           and all I-vertices in the group have the same neighbors in K, and I is independent).
           Wait, that's not right. The I-vertices are independent, so they can't be in the same clique.
           
        Let me reconsider. In a split graph:
        - K is a clique
        - I is an independent set
        - Edges only within K and between K and I
        
        A valid clique can contain:
        - Any subset of K (since K is a clique)
        - At most one vertex from I (since I is independent)
        
        So a clique is either:
        - A subset of K (size >= 2)
        - A subset of K plus one vertex from I (where that I-vertex is adjacent to all vertices in the K-subset)
        
        The edges to cover are:
        - All edges within K
        - All cross-edges between K and I
        
        Each edge must be in exactly one clique.
        
        Approach:
        1. For cross-edges: each cross-edge (u, v) where u in K, v in I must be covered by a clique containing both u and v. Since a clique can contain at most one I-vertex, each cross-edge must be in a clique that contains that specific I-vertex.
        
        2. For a given I-vertex v with neighborhood N(v) in K, the cross-edges from v to N(v) must all be covered. They can be covered by:
           - One clique containing v and all of N(v) (if |N(v)| >= 2, this is a valid clique since N(v) is a subset of K)
           - Or multiple cliques, each containing v and a subset of N(v)
           
        Using one clique per I-vertex (containing v and its entire neighborhood N(v)) covers all cross-edges from v in one clique. But this clique also covers edges within N(v), which are K-edges. Those K-edges must NOT be covered by any other clique.
        
        So if we use clique {v} ∪ N(v) for each v in I, then all K-edges within any N(v) are covered. The remaining K-edges (those not in any N(v)) must be covered by cliques that are subsets of K only.
        
        This is a good approach! Let's implement it:
        1. For each v in I, create clique {v} ∪ N(v) (if |N(v)| >= 1, but we need at least 2 vertices total, so if |N(v)| >= 1 then |{v} ∪ N(v)| >= 2)
        2. Track which K-edges are covered by these cliques.
        3. Cover the remaining K-edges with additional cliques (subsets of K).
        
        For step 3, we need to partition the remaining K-edges into cliques. Since K is a complete graph, any subset of K is a clique. We just need to cover the uncovered edges with as few cliques as possible, where each clique is a subset of K and covers all edges within it.
        
        Actually, if we have a set of uncovered edges in K, we need to cover them with cliques (subsets of K). Each clique covers all edges within its vertex set. We want to minimize the number of cliques.
        
        This is equivalent to: given a graph H on vertex set K (with edges being the uncovered K-edges), cover the edges of H with the minimum number of cliques. This is the clique cover number of H, which equals the chromatic number of the complement of H.
        
        For small k (10, 20, 30), we can use a greedy approach or even brute force for small cases.
        
        Let's implement:
        1. Generate cross-edges and group by I-vertex.
        2. For each I-vertex with non-empty neighborhood, add clique {v} ∪ N(v).
        3. Determine uncovered K-edges.
        4. Cover uncovered K-edges with minimum number of cliques (greedy or exact for small k).
        """
        rng = Random(seed)
        
        # Build neighborhood for each I-vertex
        neighborhoods = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighborhoods[v] = set(nbrs)
        
        cliques = []
        
        # Step 1: Add cliques for each I-vertex with non-empty neighborhood
        covered_k_edges = set()
        for v in range(k, n):
            nbrs = neighborhoods[v]
            if len(nbrs) >= 1:
                clique = sorted(list(nbrs) + [v])
                cliques.append(clique)
                # Mark all K-edges within nbrs as covered
                nbrs_list = sorted(nbrs)
                for i in range(len(nbrs_list)):
                    for j in range(i + 1, len(nbrs_list)):
                        covered_k_edges.add((nbrs_list[i], nbrs_list[j]))
        
        # Step 2: Determine uncovered K-edges
        all_k_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                all_k_edges.add((i, j))
        
        uncovered_k_edges = all_k_edges - covered_k_edges
        
        # Step 3: Cover uncovered K-edges with cliques (subsets of K)
        # Greedy: repeatedly find the largest clique that covers the most uncovered edges
        # Since K is complete, any subset is a clique. We want to maximize edges covered.
        # For a subset S of K, it covers all edges within S that are in uncovered_k_edges.
        
        remaining = set(uncovered_k_edges)
        
        while remaining:
            # Find the subset S of K that maximizes |{edges in S that are in remaining}|
            # Greedy: try all pairs, then extend
            best_clique = None
            best_count = 0
            
            # Try all single edges first
            for e in remaining:
                if best_count == 0:
                    best_clique = list(e)
                    best_count = 1
            
            # Try to extend by adding vertices
            # For each candidate clique, try adding vertices that connect to all existing vertices
            if best_clique is not None:
                # Greedy extension
                current = set(best_clique)
                changed = True
                while changed:
                    changed = False
                    for u in range(k):
                        if u not in current:
                            # Check if u is connected to all vertices in current (in the remaining graph)
                            # i.e., (u, v) is in remaining for all v in current
                            can_add = True
                            for v in current:
                                if v < u:
                                    if (v, u) not in remaining:
                                        can_add = False
                                        break
                                else:
                                    if (u, v) not in remaining:
                                        can_add = False
                                        break
                            if can_add:
                                current.add(u)
                                changed = True
                                break
                if len(current) > len(best_clique):
                    best_clique = sorted(current)
                    # Count edges
                    best_count = 0
                    for i in range(len(best_clique)):
                        for j in range(i + 1, len(best_clique)):
                            a, b = best_clique[i], best_clique[j]
                            if a < b:
                                if (a, b) in remaining:
                                    best_count += 1
                            else:
                                if (b, a) in remaining:
                                    best_count += 1
            
            if best_clique is None or best_count == 0:
                break
            
            cliques.append(best_clique)
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    if a < b:
                        remaining.discard((a, b))
                    else:
                        remaining.discard((b, a))
        
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
