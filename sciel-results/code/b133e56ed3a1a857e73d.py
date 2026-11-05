"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from collections import defaultdict

def construct():
    deadline = time.time() + 130.0
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert edges to a set of frozensets
        edge_set = set()
        for u, v in edges:
            edge_set.add(frozenset((u, v)))
        
        remaining_edges = set(edge_set)
        cliques = []
        
        # Precompute eigenvector centrality for initial ordering
        # Build adjacency matrix
        A = np.zeros((n, n))
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Compute eigenvector centrality using power iteration
        try:
            eigenvalues, eigenvectors = np.linalg.eig(A)
            # Take the eigenvector corresponding to the largest eigenvalue
            idx = np.argmax(eigenvalues.real)
            centrality = np.abs(eigenvectors[:, idx].real)
            # Normalize
            centrality = centrality / (centrality.sum() + 1e-10)
        except:
            centrality = np.array([len(adj[i]) for i in range(n)], dtype=float)
            centrality = centrality / (centrality.sum() + 1e-10)
        
        # Sort vertices by centrality descending
        vertex_order = np.argsort(-centrality)
        
        def find_max_clique_in_subgraph(vertices, adj_sub):
            """Find a large clique using branch and bound on the given vertex set."""
            if not vertices:
                return []
            
            # Convert to list for easier manipulation
            verts = list(vertices)
            
            # If small, try all combinations or use a heuristic
            if len(verts) <= 1:
                return verts
            
            # Use a greedy approach with backtracking for small sets
            # For larger sets, use a heuristic
            
            best_clique = []
            
            # Greedy expansion
            def expand_clique(current_clique, candidates):
                nonlocal best_clique
                if not candidates:
                    if len(current_clique) > len(best_clique):
                        best_clique = current_clique[:]
                    return
                
                # Try each candidate
                for i, v in enumerate(candidates):
                    new_clique = current_clique + [v]
                    # Find candidates that connect to v
                    new_candidates = [u for u in candidates[i+1:] if u in adj_sub[v]]
                    expand_clique(new_clique, new_candidates)
                
                # Also consider not taking any more candidates
                if len(current_clique) > len(best_clique):
                    best_clique = current_clique[:]
            
            # Limit the search to avoid exponential blowup
            if len(verts) <= 12:
                expand_clique([], verts)
            else:
                # Greedy heuristic for larger sets
                current_clique = []
                candidates = verts[:]
                
                while candidates:
                    # Pick vertex with highest centrality that connects to all in current_clique
                    best_v = None
                    best_c = -1
                    for v in candidates:
                        if all(v in adj_sub[u] for u in current_clique):
                            if centrality[v] > best_c:
                                best_c = centrality[v]
                                best_v = v
                    if best_v is None:
                        break
                    current_clique.append(best_v)
                    candidates = [u for u in candidates if u != best_v and u in adj_sub[best_v]]
                
                best_clique = current_clique
            
            return best_clique
        
        while remaining_edges:
            if time.time() > deadline - 15:
                # Safety margin: cover remaining edges individually
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Find vertex with highest degree in remaining graph
            deg = {v: 0 for v in range(n)}
            for e in remaining_edges:
                u, v = e
                deg[u] += 1
                deg[v] += 1
            
            if not any(d > 0 for d in deg.values()):
                break
            
            # Start from highest centrality vertex that still has edges
            start_vertex = None
            for v in vertex_order:
                if deg[v] > 0:
                    start_vertex = v
                    break
            if start_vertex is None:
                start_vertex = max(deg, key=lambda x: deg[x])
            
            # Build local adjacency for remaining graph
            local_adj = [set() for _ in range(n)]
            for e in remaining_edges:
                u, v = e
                local_adj[u].add(v)
                local_adj[v].add(u)
            
            # Find a large clique containing start_vertex
            # Get neighbors of start_vertex in remaining graph
            neighbors = list(local_adj[start_vertex])
            
            if not neighbors:
                # Should not happen if deg > 0
                e = next(iter(remaining_edges))
                clique = list(e)
            else:
                # Use branch and bound on neighbors + start_vertex
                clique = find_max_clique_in_subgraph([start_vertex] + neighbors, local_adj)
                
                # Ensure start_vertex is in the clique
                if start_vertex not in clique:
                    clique = [start_vertex] + clique
                    # Verify it's a clique
                    valid = True
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            if clique[j] not in local_adj[clique[i]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        # Fallback to just start_vertex and its best neighbor
                        clique = [start_vertex]
                        for nb in neighbors:
                            if nb in local_adj[start_vertex]:
                                clique.append(nb)
                                break
            
            if len(clique) < 2:
                e = next(iter(remaining_edges))
                clique = list(e)
            
            cliques.append(clique)
            
            # Remove edges in this clique from remaining_edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = frozenset((clique[i], clique[j]))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
        
        return cliques
    
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
