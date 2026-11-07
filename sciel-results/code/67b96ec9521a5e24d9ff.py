import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets for fast lookup
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
        
        cliques = []
        remaining_edges = set(edge_set)
        
        # Step 1: Handle independent set vertices
        # For each v in I, if it has neighbors in K, form clique N_K(v) U {v}
        for v in range(k, n):
            if time.time() > deadline:
                # Fallback: cover remaining edges individually
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            neighbors_in_k = [u for u in range(k) if (min(u, v), max(u, v)) in remaining_edges]
            if neighbors_in_k:
                clique = neighbors_in_k + [v]
                cliques.append(clique)
                # Remove all edges in this clique from remaining_edges
                # Edges are (u, v) for u in neighbors_in_k
                # And (u, w) for u, w in neighbors_in_k
                for u in neighbors_in_k:
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
                for i in range(len(neighbors_in_k)):
                    for j in range(i + 1, len(neighbors_in_k)):
                        u, w = neighbors_in_k[i], neighbors_in_k[j]
                        e = (u, w) if u < w else (w, u)
                        remaining_edges.discard(e)
        
        # Step 2: Handle remaining edges in K
        # These are edges (u, w) with u, w < k
        # Build a graph H on vertices 0..k-1 with edges in remaining_edges
        if remaining_edges:
            # Active vertices in K
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            # Greedy maximal clique extraction on the residual graph
            # We will repeatedly find a maximal clique, remove its edges, and repeat.
            while remaining_edges:
                if time.time() > deadline:
                    # Fallback: cover remaining edges individually
                    for e in remaining_edges:
                        cliques.append([e[0], e[1]])
                    break
                
                # Find a maximal clique in the graph defined by remaining_edges
                # Heuristic: Start with the vertex with highest degree in the residual graph
                # and greedily add neighbors.
                
                # Calculate degrees in residual graph
                deg = {v: 0 for v in active_vertices}
                for u, v in remaining_edges:
                    deg[u] += 1
                    deg[v] += 1
                
                # Pick a starting vertex (highest degree)
                if not deg:
                    break
                start_v = max(deg, key=lambda x: deg[x])
                
                clique = [start_v]
                candidates = set(adj[start_v]) & active_vertices
                # We need to ensure candidates are connected to all in clique
                # Since we start with 1, candidates are just neighbors.
                
                # Greedy expansion
                # We want to find a large clique.
                # Sort candidates by degree in the induced subgraph of candidates?
                # Or just iterate.
                
                # A simple greedy:
                # While candidates is not empty:
                #   Pick v in candidates with highest degree in candidates
                #   Add v to clique
                #   Remove v from candidates
                #   Remove from candidates any vertex not connected to v
                
                while candidates:
                    # Find vertex in candidates with highest degree within candidates
                    # This is O(|candidates|^2) which is fine for k<=30
                    best_c = None
                    best_deg = -1
                    for c in candidates:
                        d = 0
                        for other in candidates:
                            if c != other and (min(c, other), max(c, other)) in remaining_edges:
                                d += 1
                        if d > best_deg:
                            best_deg = d
                            best_c = c
                    if best_c is None:
                        break
                    clique.append(best_c)
                    candidates.discard(best_c)
                    # Filter candidates: must be connected to best_c
                    # And already connected to all others in clique (implied by previous steps if we maintain intersection)
                    # Actually, since we are in a clique, all current candidates are connected to all in clique.
                    # Adding best_c requires new candidates to be connected to best_c.
                    candidates = {c for c in candidates if (min(c, best_c), max(c, best_c)) in remaining_edges}
                
                if len(clique) >= 2:
                    cliques.append(clique)
                    # Remove edges of this clique from remaining_edges
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            u, v = clique[i], clique[j]
                            e = (u, v) if u < v else (v, u)
                            remaining_edges.discard(e)
                else:
                    # Should not happen if remaining_edges is not empty, but just in case
                    # If clique is size 1, we didn't cover any edge.
                    # This can happen if the graph is not connected or our logic failed.
                    # Force pick an edge
                    if remaining_edges:
                        u, v = next(iter(remaining_edges))
                        cliques.append([u, v])
                        remaining_edges.discard((u, v) if u < v else (v, u))
                    else:
                        break

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
