"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def get_maximal_clique(edges_set, n, k):
        """
        Find a large clique in the graph defined by edges_set.
        edges_set is a set of tuples (u, v) with u < v.
        n is total vertices, k is size of clique part C = {0..k-1}.
        """
        if not edges_set:
            return []
        
        # Build adjacency lists for the remaining graph
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            
        best_clique = []
        
        # Heuristic: Try to find a clique anchored at each vertex, or just a few high-degree ones
        # To be fast, we'll try a greedy expansion from each vertex, but only keep the best.
        # Given n<=90, this is O(n * E) which is fine.
        
        # Sort vertices by degree descending to prioritize high-degree nodes
        vertices = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        for start in vertices:
            if not adj[start]:
                continue
                
            # Greedy expansion
            clique = {start}
            candidates = set(adj[start])
            
            # We want to find a maximal clique containing 'start'
            # Simple greedy: add vertices that are connected to all current clique members
            # This doesn't guarantee maximum, but is fast.
            # To improve, we can do a simple local search or just accept this.
            # Given the structure, let's try a slightly smarter greedy:
            # Pick next vertex with highest degree within candidates.
            
            current_clique = [start]
            candidates = list(adj[start])
            
            while candidates:
                # Pick candidate with highest degree in the induced subgraph of candidates
                # Approximation: pick one with highest degree in global adj
                best_cand = max(candidates, key=lambda x: len(adj[x]))
                
                # Check if best_cand is connected to all in current_clique
                is_connected = all(best_cand in adj[v] for v in current_clique)
                
                if is_connected:
                    current_clique.append(best_cand)
                    # Remove best_cand and its non-neighbors from candidates?
                    # For a clique, all new vertices must be connected to all existing.
                    # So candidates must be neighbors of best_cand AND all previous.
                    new_candidates = []
                    for c in candidates:
                        if c == best_cand:
                            continue
                        if all(c in adj[v] for v in current_clique):
                            new_candidates.append(c)
                    candidates = new_candidates
                else:
                    # Remove best_cand from candidates, try next best
                    candidates.remove(best_cand)
                    
            if len(current_clique) > len(best_clique):
                best_clique = current_clique
                
        return best_clique

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
            
        # Use a set for O(1) removal
        remaining_edges = set(edges)
        cliques = []
        
        # To ensure determinism and good performance, we might need to be careful.
        # The greedy above picks the "best" found by trying all starts.
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return what we have? No, must be valid.
                # If we run out of time, we should have finished.
                # For n=90, this loop should run < 100 times.
                pass
                
            clique = get_maximal_clique(remaining_edges, n, k)
            
            if not clique:
                # Should not happen if remaining_edges is not empty
                break
            
            # Validate clique (safety)
            # Check all pairs in clique are in remaining_edges
            valid = True
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u,v), max(u,v))
                    if e not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback: just take an edge
                u, v = next(iter(remaining_edges))
                clique = [u, v]
            
            cliques.append(clique)
            
            # Remove edges covered by this clique
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u,v), max(u,v))
                    remaining_edges.discard(e)
                    
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
