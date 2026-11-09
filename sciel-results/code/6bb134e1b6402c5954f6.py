"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import Counter

def _generate_edges(n, p, seed):
    """Generate edges for a random graph using the same RNG as the verifier."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    """Build adjacency list and bitmasks."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Bitmasks for fast intersection
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    return adj, adj_mask

def _bron_kerbosch_pivot(R, P, X, adj, adj_mask, cliques):
    """Bron-Kerbosch with pivot. Only records cliques of size >= 2."""
    if not P and not X:
        if len(R) >= 2:
            cliques.append(list(R))
        return
    
    # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
    union_px = P | X
    if not union_px:
        if len(R) >= 2:
            cliques.append(list(R))
        return
    
    # Find pivot with max degree in P
    pivot = None
    max_count = -1
    for u in union_px:
        count = len(P & adj[u])
        if count > max_count:
            max_count = count
            pivot = u
        if count == len(P):  # Can't do better
            break
    
    # Recurse on vertices in P \ N(pivot)
    candidates = P - adj[pivot] if pivot is not None else P
    
    for v in list(candidates):
        new_R = R | {v}
        new_P = P & adj[v]
        new_X = X & adj[v]
        P.remove(v)
        X.add(v)
        _bron_kerbosch_pivot(new_R, new_P, new_X, adj, adj_mask, cliques)
        P.add(v)
        X.remove(v)

def _enumerate_maximal_cliques(n, edges):
    """Enumerate all maximal cliques of size >= 2."""
    adj, adj_mask = _build_adjacency(n, edges)
    cliques = []
    _bron_kerbosch_pivot(set(), set(range(n)), set(), adj, adj_mask, cliques)
    return cliques

def _greedy_hitting_set(cliques, n):
    """Greedy set cover: repeatedly pick the vertex covering the most uncovered cliques."""
    # Count how many cliques each vertex is in
    vertex_clique_count = Counter()
    for clique in cliques:
        for v in clique:
            vertex_clique_count[v] += 1
    
    # Track which cliques are covered
    uncovered = set(range(len(cliques)))
    selected = set()
    
    # For each vertex, track which cliques it covers
    vertex_cliques = {v: set() for v in vertex_clique_count}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    while uncovered:
        # Find vertex covering most uncovered cliques
        best_v = None
        best_count = -1
        for v in vertex_clique_count:
            count = len(vertex_cliques[v] & uncovered)
            if count > best_count:
                best_count = count
                best_v = v
            elif count == best_count and best_v is not None:
                # Tie-break: prefer vertex with fewer total cliques (less "wasteful")
                if vertex_clique_count[v] < vertex_clique_count[best_v]:
                    best_v = v
        
        if best_v is None or best_count <= 0:
            break
        
        selected.add(best_v)
        uncovered -= vertex_cliques[best_v]
    
    return selected

def _local_search(selected, cliques, n, time_limit=10.0):
    """Local search: try to remove vertices and swap to reduce size."""
    start_time = time.time()
    selected = set(selected)
    
    # Precompute for each vertex which cliques it covers
    def get_covered_cliques(v):
        return [i for i, clique in enumerate(cliques) if v in clique]
    
    # Try removals
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for v in list(selected):
            # Check if removing v still covers all cliques
            # v is redundant if every clique containing v also contains another selected vertex
            redundant = True
            for clique in cliques:
                if v in clique:
                    if not any(u in selected for u in clique if u != v):
                        redundant = False
                        break
            if redundant:
                selected.remove(v)
                improved = True
    
    # Try swaps: remove one, add another that helps
    for _ in range(3):  # A few rounds of swaps
        if time.time() - start_time > time_limit:
            break
        swapped = False
        for v in list(selected):
            # Try removing v
            candidate = selected - {v}
            # Find cliques that would be uncovered
            uncovered_cliques = [i for i, clique in enumerate(cliques) 
                                 if v in clique and not any(u in candidate for u in clique)]
            if not uncovered_cliques:
                # v is redundant, already handled
                continue
            # Find a vertex w not in selected that covers all uncovered_cliques
            for w in range(n):
                if w in candidate:
                    continue
                if all(w in cliques[i] for i in uncovered_cliques):
                    # Check if w doesn't create new issues (it won't, since we're adding)
                    candidate.add(w)
                    # Verify all cliques are still covered
                    valid = True
                    for clique in cliques:
                        if not any(u in candidate for u in clique):
                            valid = False
                            break
                    if valid:
                        selected = candidate
                        swapped = True
                        break
                if time.time() - start_time > time_limit:
                    break
            if swapped or time.time() - start_time > time_limit:
                break
        if not swapped:
            break
    
    return selected

def construct():
    deadline = time.time() + 120  # Safety margin before 150s kill
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Not enough time, return what we have
            break
        
        edges = _generate_edges(n, p, seed)
        cliques = _enumerate_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy hitting set
        selected = _greedy_hitting_set(cliques, n)
        
        # Local search
        remaining_time = max(1.0, deadline - time.time() - 5)
        selected = _local_search(selected, cliques, n, time_limit=remaining_time)
        
        results.append(sorted(selected))
    
    # If we ran out of time, fill in with trivial solutions
    while len(results) < 3:
        n, p, seed = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))[len(results)]
        edges = _generate_edges(n, p, seed)
        touched = set()
        for u, v in edges:
            touched.add(u)
            touched.add(v)
        results.append(sorted(touched))
    
    return results
# EVOLVE-BLOCK-END
