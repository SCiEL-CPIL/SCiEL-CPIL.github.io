"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _generate_edges(n, p, seed):
    """Generate edges using the same RNG sequence as the verifier."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed, deadline):
    """Find a clique partition of the random graph."""
    edges = _generate_edges(n, p, seed)
    edge_set = set(edges)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all maximal cliques using Bron-Kerbosch
    maximal_cliques = []
    
    def bronk(R, P, X):
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) >= 2:
                maximal_cliques.append(frozenset(R))
        # Pivot selection: choose vertex in P ∪ X with max degree in P
        if P or X:
            # Pick pivot from P ∪ X that maximizes |P ∩ N(u)|
            pivot = None
            max_count = -1
            for u in P:
                count = len(P & adj[u])
                if count > max_count:
                    max_count = count
                    pivot = u
            for u in X:
                count = len(P & adj[u])
                if count > max_count:
                    max_count = count
                    pivot = u
            
            if pivot is not None:
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                if time.time() > deadline:
                    return
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
    
    bronk(set(), set(range(n)), set())
    
    # Greedy set cover: prioritize cliques with highest ratio of newly covered edges to size
    # Each clique of size k covers k*(k-1)/2 edges
    # We want to minimize total number of cliques
    
    # Filter to cliques that are subsets of the graph (all edges present)
    valid_cliques = []
    for clique in maximal_cliques:
        clique_list = list(clique)
        if len(clique_list) < 2:
            continue
        # Verify all edges in clique exist
        valid = True
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = clique_list[i], clique_list[j]
                if min(a,b) > max(a,b):
                    a, b = b, a
                if (a, b) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            valid_cliques.append(clique)
    
    # Also include all edges as size-2 cliques
    for u, v in edges:
        valid_cliques.append(frozenset([u, v]))
    
    # Greedy set cover
    remaining_edges = set(edge_set)
    chosen_cliques = []
    
    # Sort cliques by size descending for initial preference
    # But use greedy: at each step, pick clique covering most remaining edges
    clique_list = list(valid_cliques)
    
    while remaining_edges:
        if time.time() > deadline:
            break
        best_clique = None
        best_count = 0
        for clique in clique_list:
            if len(clique) < 2:
                continue
            # Count edges in this clique that are still remaining
            count = 0
            clique_sorted = sorted(clique)
            for i in range(len(clique_sorted)):
                for j in range(i + 1, len(clique_sorted)):
                    e = (clique_sorted[i], clique_sorted[j])
                    if e in remaining_edges:
                        count += 1
            if count > best_count:
                best_count = count
                best_clique = clique
            elif count == best_count and count > 0:
                # Prefer larger cliques
                if best_clique is not None and len(clique) > len(best_clique):
                    best_clique = clique
        
        if best_clique is None or best_count == 0:
            # Should not happen if graph is valid, but safety
            break
        
        chosen_cliques.append(sorted(best_clique))
        # Remove covered edges
        clique_sorted = sorted(best_clique)
        for i in range(len(clique_sorted)):
            for j in range(i + 1, len(clique_sorted)):
                e = (clique_sorted[i], clique_sorted[j])
                remaining_edges.discard(e)
        
        # Remove clique from list
        try:
            clique_list.remove(best_clique)
        except ValueError:
            pass
    
    # If any edges remain, add them as individual edge cliques
    if remaining_edges:
        for e in remaining_edges:
            chosen_cliques.append(sorted(e))
    
    return chosen_cliques

def construct():
    deadline = time.time() + 130  # Leave margin before 150s kill
    
    result1 = _solve_instance(32, 0.6, 601, deadline)
    
    if time.time() > deadline:
        # Fallback: return edge-only partition
        edges1 = _generate_edges(32, 0.6, 601)
        result1 = [sorted(e) for e in edges1]
    
    result2 = _solve_instance(64, 0.55, 602, deadline)
    
    if time.time() > deadline:
        edges2 = _generate_edges(64, 0.55, 602)
        result2 = [sorted(e) for e in edges2]
    
    return [result1, result2]
# EVOLVE-BLOCK-END
