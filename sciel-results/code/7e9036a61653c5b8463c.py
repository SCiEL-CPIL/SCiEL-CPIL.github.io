"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit

    def random_graph_edges(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_max_clique(adj, n, time_limit):
        """
        Find a maximal clique in the graph. Uses a simple greedy approach:
        start from each vertex, try to extend to a clique by adding neighbors
        in order of decreasing degree within current candidate set.
        Returns the largest clique found.
        """
        if time.time() > time_limit:
            return None
        
        best_clique = []
        
        for start in range(n):
            if time.time() > time_limit:
                break
            
            # Try to build a clique starting from 'start'
            clique = [start]
            candidates = set(adj[start])
            
            while candidates:
                if time.time() > time_limit:
                    break
                
                # Pick candidate with max degree within candidates
                best_cand = None
                best_deg = -1
                for c in candidates:
                    # Count how many of c's neighbors are in candidates
                    deg = len(adj[c] & candidates)
                    if deg > best_deg:
                        best_deg = deg
                        best_cand = c
                
                if best_cand is None:
                    break
                
                # Check if best_cand connects to all current clique members
                valid = True
                for member in clique:
                    if best_cand not in adj[member]:
                        valid = False
                        break
                
                if not valid:
                    # Remove best_cand from candidates and try next
                    candidates.discard(best_cand)
                    continue
                
                clique.append(best_cand)
                candidates.discard(best_cand)
                # Update candidates: keep only those connected to all clique members
                new_candidates = set()
                for c in candidates:
                    if all(c in adj[m] for m in clique):
                        new_candidates.add(c)
                candidates = new_candidates
            
            if len(clique) > len(best_clique):
                best_clique = clique
        
        return best_clique if best_clique else None

    def greedy_max_clique_partition(edges, n, adj, time_limit):
        """
        Repeatedly extract a maximal clique from the remaining edge graph,
        remove its edges, and repeat until no edges remain.
        """
        edge_set = set(frozenset(e) for e in edges)
        uncovered = set(edge_set)
        
        if not uncovered:
            return []
        
        used_cliques = []
        
        while uncovered:
            if time.time() > time_limit:
                break
            
            # Build adjacency of remaining edges
            rem_adj = [set() for _ in range(n)]
            for e in uncovered:
                u, v = min(e), max(e)
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Find maximal clique in remaining graph
            clique = find_max_clique(rem_adj, n, time_limit)
            
            if clique is None or len(clique) < 2:
                # No more cliques of size >= 2; remaining edges are isolated
                for e in uncovered:
                    u, v = min(e), max(e)
                    used_cliques.append([u, v])
                break
            
            if len(clique) >= 2:
                used_cliques.append(list(clique))
                # Remove all edges in this clique
                for pair in combinations(clique, 2):
                    edge_set = frozenset(pair)
                    uncovered.discard(edge_set)
        
        # Any remaining uncovered edges as 2-vertex cliques
        for e in uncovered:
            u, v = min(e), max(e)
            used_cliques.append([u, v])
        
        return used_cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph_edges(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        result = greedy_max_clique_partition(edges, n, adj, time_limit)
        return result

    # Solve both instances
    results = []
    remaining_time = DEADLINE - time.time()
    time_per_instance = remaining_time / 2
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        t_start = time.time()
        t_limit = t_start + time_per_instance
        result = solve_instance(n, p, seed, t_limit)
        results.append(result)
    
    return results

# EVOLVE-BLOCK-END
