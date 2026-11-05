import time
import sys
import random
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        if not edges_set:
            return []
        
        # 1. Compute Spectral Centrality
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges_set:
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Use eigsh to find the eigenvector corresponding to the largest eigenvalue
        # This vector approximates eigenvector centrality.
        # For dense graphs, the largest eigenvalue is well-separated.
        try:
            # k=1 to get the top eigenvector
            # which='LM' for Largest Magnitude
            # sigma=0 is not needed for LM, but we can use it if we want algebraic largest.
            # For a graph with positive weights, the largest eigenvalue is positive.
            # eigsh requires k < n.
            if n <= 2:
                # Fallback for very small n
                cent = np.ones(n)
            else:
                # Convert to sparse for efficiency if n is large, but n=64 is small.
                # Dense eigsh might be slower, but for 64x64 it's fine.
                # Actually, scipy.sparse.linalg.eigsh works on sparse matrices.
                A_sp = csr_matrix(adj)
                # Use which='LM' to get the largest magnitude eigenvalue
                # For connected graphs, this is the Perron-Frobenius eigenvalue.
                eigenvalues, eigenvectors = eigsh(A_sp, k=1, which='LM')
                # Take the absolute value of the eigenvector to get centrality
                # The sign is arbitrary.
                cent = np.abs(eigenvectors[:, 0])
                # Normalize
                cent = cent / np.linalg.norm(cent)
        except Exception:
            # Fallback to degree centrality if spectral fails
            degrees = adj.sum(axis=1)
            cent = degrees / (np.sum(degrees) + 1e-9)

        # 2. Iterative Maximum Clique Extraction
        # We maintain the set of uncovered edges.
        # At each step:
        #   a. Find the vertex v with the highest centrality among vertices incident to uncovered edges.
        #   b. Find the maximum clique in the subgraph induced by the uncovered edges, 
        #      that MUST contain v.
        #   c. Remove all edges in this clique from the uncovered set.
        #   d. Add the clique to the result.
        
        uncovered = set(edges_set)
        cliques = []
        
        # Precompute adjacency sets for the original graph for fast checks
        # But we need to check against 'uncovered'.
        # We will maintain an adjacency set for uncovered edges.
        unc_adj = {i: set() for i in range(n)}
        for u, v in uncovered:
            unc_adj[u].add(v)
            unc_adj[v].add(u)
            
        # Helper to check if time is up
        def time_up():
            return time.time() > DEADLINE

        while uncovered:
            if time_up():
                # Fallback: cover remaining edges individually
                for u, v in uncovered:
                    cliques.append([u, v])
                break
            
            # Find active vertices (incident to uncovered edges)
            active_vertices = set()
            for u, v in uncovered:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
                
            # Select seed vertex: highest centrality among active vertices
            # We use the precomputed centrality.
            # If we want to recompute centrality dynamically, it's expensive.
            # Static centrality is a good heuristic for "core" vertices.
            seed_vertex = max(active_vertices, key=lambda x: cent[x])
            
            # Find Maximum Clique containing seed_vertex in the uncovered subgraph.
            # The subgraph is defined by 'uncovered' edges.
            # Vertices in the clique must be pairwise connected in 'uncovered'.
            
            # Candidates for the clique: neighbors of seed_vertex in 'uncovered'
            candidates = list(unc_adj[seed_vertex])
            
            # If no candidates, the clique is just {seed_vertex}? 
            # No, a clique must have at least 2 vertices (edges).
            # If seed_vertex has no uncovered neighbors, we can't form a clique with it?
            # Wait, if seed_vertex is active, it has at least one uncovered neighbor.
            
            # We want to find the maximum clique C such that seed_vertex in C.
            # This is equivalent to finding the maximum clique in the subgraph induced by 
            # {seed_vertex} U candidates, with the constraint that seed_vertex is included.
            # Which is equivalent to finding a maximum clique in the subgraph induced by candidates
            # that is also a clique with seed_vertex (which is true by definition of candidates),
            # and then adding seed_vertex.
            # So, find max clique in G[candidates].
            
            # Use Bron-Kerbosch with pivot for maximum clique.
            # n is small (<=64), so recursion depth is manageable.
            # We need to find the MAXIMUM size clique, not just one maximal.
            # Standard Bron-Kerbosch enumerates maximal cliques.
            # We can modify it to track the best one.
            
            best_clique = set()
            
            # Convert candidates to a set for fast intersection
            cand_set = set(candidates)
            
            # We need to check if a set of vertices forms a clique in the UNCOVERED graph.
            def is_clique_in_unc(verts):
                if not verts:
                    return True
                first = next(iter(verts))
                for v in verts:
                    if v == first: continue
                    if v not in unc_adj[first]:
                        return False
                return True
                
            # Bron-Kerbosch to find maximum clique in G[cand_set]
            # P: set of candidates
            # X: set of excluded candidates (for maximal clique enumeration)
            # R: current clique
            
            def bron_kerbosch_max(P, X, R):
                nonlocal best_clique
                if time_up():
                    return
                
                if not P and not X:
                    # Found a maximal clique
                    if len(R) > len(best_clique):
                        best_clique = set(R)
                    return
                
                # Pivot selection: choose u from P U X with max neighbors in P
                # This reduces the number of recursive calls.
                pivot_candidates = P | X
                if pivot_candidates:
                    # Choose pivot with max degree in P
                    pivot = max(pivot_candidates, key=lambda u: len(P & unc_adj[u]))
                    # Candidates are P \ N(pivot)
                    candidates_for_recursion = P - unc_adj[pivot]
                else:
                    candidates_for_recursion = P
                    
                for v in list(candidates_for_recursion):
                    bron_kerbosch_max(P & unc_adj[v], X & unc_adj[v], R | {v})
                    P.remove(v)
                    X.add(v)
                    
            # Initial call
            # P = candidates
            # X = empty
            # R = empty
            # Note: We are looking for clique in G[candidates].
            # The edges are from 'uncovered'.
            # unc_adj is built from 'uncovered'.
            
            bron_kerbosch_max(cand_set, set(), set())
            
            # If no clique found (shouldn't happen if seed is active), fallback to edge
            if not best_clique:
                # Pick any neighbor
                if unc_adj[seed_vertex]:
                    v = next(iter(unc_adj[seed_vertex]))
                    current_clique = {seed_vertex, v}
                else:
                    # This shouldn't happen if seed is active
                    break
            else:
                current_clique = best_clique | {seed_vertex}
            
            # Remove edges of current_clique from uncovered
            clique_list = list(current_clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in uncovered:
                        uncovered.remove((a, b))
                        # Update unc_adj
                        unc_adj[a].remove(b)
                        unc_adj[b].remove(a)
            
            cliques.append(clique_list)
            
        return cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
