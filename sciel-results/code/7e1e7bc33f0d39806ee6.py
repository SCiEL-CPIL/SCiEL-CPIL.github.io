"""Edge clique partition using greedy maximal clique extraction with time guard."""
import time
import sys
from random import Random

def _random_graph(n, p, seed):
    """Generate the same random graph as the verifier."""
    rng = Random(seed)
    edges = []
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return edges, adj

def _find_maximal_cliques(adj, n):
    """Find all maximal cliques using a simple backtracking approach."""
    cliques = []
    
    def expand(clique, candidates):
        if not candidates:
            if len(clique) >= 2:
                cliques.append(frozenset(clique))
            return
        # Choose vertex with highest degree in candidates
        v = max(candidates, key=lambda x: len(adj[x] & candidates))
        # Include v
        expand(clique + [v], candidates & adj[v] - {v})
        # Exclude v
        expand(clique, candidates - {v})
    
    # Start with empty clique, all vertices as candidates
    expand([], set(range(n)))
    return cliques

def _solve_instance(n, p, seed, deadline):
    """Solve one instance: partition edges into minimum number of cliques."""
    edges, adj = _random_graph(n, p, seed)
    
    if not edges:
        return []
    
    # Find all maximal cliques
    start = time.time()
    cliques = _find_maximal_cliques(adj, n)
    
    if time.time() > deadline:
        # Fallback: just use individual edges
        return [list(e) for e in edges]
    
    # Greedy: repeatedly pick the clique that covers the most uncovered edges
    edge_set = set(edges)
    uncovered = set(edges)
    result = []
    
    while uncovered:
        if time.time() > deadline:
            # Fallback for remaining edges
            result.extend([list(e) for e in uncovered])
            break
        
        best_clique = None
        best_count = 0
        for c in cliques:
            c_set = frozenset(c)
            # Count how many uncovered edges are within this clique
            count = 0
            for e in uncovered:
                if e[0] in c_set and e[1] in c_set:
                    count += 1
            if count > best_count:
                best_count = count
                best_clique = c
        
        if best_clique is None or best_count == 0:
            # No clique covers any uncovered edge; use single edges
            result.extend([list(e) for e in uncovered])
            break
        
        # Add this clique to result
        result.append(list(best_clique))
        # Remove covered edges
        c_set = frozenset(best_clique)
        new_uncovered = set()
        for e in uncovered:
            if not (e[0] in c_set and e[1] in c_set):
                new_uncovered.add(e)
        uncovered = new_uncovered
    
    return result

def construct():
    deadline = time.time() + 115  # Leave 5s safety margin before 120s limit
    
    # Instance 1: n=32, p=0.6, seed=601
    result1 = _solve_instance(32, 0.6, 601, deadline)
    
    # Instance 2: n=64, p=0.55, seed=602
    result2 = _solve_instance(64, 0.55, 602, deadline)
    
    return [result1, result2]
