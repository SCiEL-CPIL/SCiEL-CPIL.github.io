"""Program using global greedy maximum clique partitioning."""
import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        r = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if r.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(edges, n):
        if not edges:
            return []
        
        # Build adjacency sets
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        
        while edges:
            # Compute degrees
            degrees = {u: len(adj[u]) for u in adj if adj[u]}
            if not degrees:
                break
                
            # Sort vertices by degree descending
            sorted_vertices = sorted(degrees.keys(), key=lambda x: -degrees[x])
            
            best_clique = []
            
            # Try to find a large clique starting from the top few high-degree vertices
            # We try starting from top 5 to increase chance of finding a larger clique
            for start_v in sorted_vertices[:5]:
                if not adj[start_v]:
                    continue
                
                current_clique = [start_v]
                # Candidates are neighbors of start_v
                candidates = set(adj[start_v])
                
                while candidates:
                    # Pick candidate with highest degree in the candidate set
                    # To be efficient, we pick the one with max global degree as a proxy,
                    # or iterate to find max intersection.
                    # Given small N, iterating is fine.
                    best_v = None
                    max_score = -1
                    
                    for v in candidates:
                        # Score: number of neighbors in candidates
                        # This ensures we pick vertices that keep the candidate set dense
                        score = len(adj[v] & candidates)
                        if score > max_score:
                            max_score = score
                            best_v = v
                            
                    if best_v is None:
                        break
                    
                    # Check if best_v connects to all in current_clique
                    is_connected = True
                    for u in current_clique:
                        if (min(u, best_v), max(u, best_v)) not in edges:
                            is_connected = False
                            break
                    
                    if is_connected:
                        current_clique.append(best_v)
                        candidates.remove(best_v)
                        # Restrict candidates to neighbors of best_v
                        candidates = candidates & adj[best_v]
                    else:
                        candidates.remove(best_v)
                        
                if len(current_clique) > len(best_clique):
                    best_clique = current_clique
            
            if not best_clique:
                # Fallback: pick any edge
                u, v = next(iter(edges))
                best_clique = [u, v]
                
            if len(best_clique) >= 2:
                cliques.append(best_clique)
                
                # Remove all edges within this clique
                clique_edges = set()
                for i in range(len(best_clique)):
                    for j in range(i + 1, len(best_clique)):
                        a, b = best_clique[i], best_clique[j]
                        e = (min(a, b), max(a, b))
                        if e in edges:
                            clique_edges.add(e)
                            
                edges -= clique_edges
                
                # Update adjacency lists
                for e in clique_edges:
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                # Clean up empty adj lists to maintain 'degrees' accuracy in next loop
                for u in list(adj.keys()):
                    if not adj[u]:
                        del adj[u]
            else:
                break

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = greedy_clique_partition(edges, n)
        result.append(cliques)

    return result
