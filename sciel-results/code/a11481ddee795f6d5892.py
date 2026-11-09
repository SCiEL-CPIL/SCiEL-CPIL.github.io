"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def find_greedy_clique_partition(edges, adj, n, time_limit):
        """Greedy: repeatedly pick a large clique from remaining edges, remove it."""
        remaining = set(edges)
        cliques = []
        
        while remaining and time.time() < time_limit:
            # Find a good clique: try to grow a clique from a random edge
            best_clique = None
            best_size = 1
            
            # Sample some starting edges to find a good clique
            edge_list = list(remaining)
            if not edge_list:
                break
            
            # Try multiple starting points
            n_tries = min(50, len(edge_list))
            indices = random.sample(range(len(edge_list)), n_tries) if len(edge_list) > n_tries else range(len(edge_list))
            
            for idx in indices:
                u, v = edge_list[idx]
                # Greedily grow clique starting from {u, v}
                clique = [u, v]
                # Common neighbors of all in clique
                common = adj[u] & adj[v]
                # Only consider vertices that have edges to all in clique
                # and are connected via remaining edges
                for w in sorted(common):
                    if w in clique:
                        continue
                    # Check if w connects to all in clique via remaining edges
                    can_add = True
                    for c in clique:
                        if (min(c, w), max(c, w)) not in remaining:
                            can_add = False
                            break
                    if can_add:
                        clique.append(w)
                        common &= adj[w]
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or len(best_clique) < 2:
                # Fallback: just take one edge
                u, v = edge_list[0]
                best_clique = [u, v]
            
            # Remove all edges within best_clique from remaining
            clique_set = set(best_clique)
            to_remove = set()
            for a in best_clique:
                for b in best_clique:
                    if a < b:
                        e = (a, b)
                        if e in remaining:
                            to_remove.add(e)
            
            remaining -= to_remove
            cliques.append([v for v in best_clique])
        
        return cliques

    def solve_instance(n, p, seed):
        edges, adj = instance(n, p, seed)
        if not edges:
            return []
        
        time_limit = min(deadline, time.time() + 40)
        cliques = find_greedy_clique_partition(edges, adj, n, time_limit)
        
        # Verify and fix: ensure every edge is covered exactly once
        # Build edge-to-clique mapping
        edge_count = {}
        for clique in cliques:
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (min(a, b), max(a, b))
                    edge_count[e] = edge_count.get(e, 0) + 1
        
        # Find edges not covered or covered multiple times
        all_edges = set(edges)
        covered = set(edge_count.keys())
        
        # Edges in multiple cliques - need to fix
        # Edges not covered - need to add as single-edge cliques
        uncovered = all_edges - covered
        for e in uncovered:
            cliques.append(list(e))
        
        return cliques

    result1 = solve_instance(32, 0.6, 601)
    if time.time() > deadline - 20:
        # Not enough time for second, return best effort
        result2 = solve_instance(64, 0.55, 602)
    else:
        result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
