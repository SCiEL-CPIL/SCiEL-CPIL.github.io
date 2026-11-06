"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random

def _bron_kerbosch_maximal_cliques(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    adj_sets = [set(adj[i]) for i in range(n)]
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X to minimize |P \ N(u)|
        upx = P | X
        if upx:
            # Pick pivot with max degree in P
            best_u = max(upx, key=lambda u: len(P & adj_sets[u]))
            candidates = P - adj_sets[best_u]
        else:
            candidates = P
        
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj_sets[v]
            new_X = X & adj_sets[v]
            bk(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}
    
    bk(frozenset(), set(range(n)), frozenset())
    return cliques

def _build_graph(n, p, seed):
    """Build adjacency list from random_graph(n, p, seed)."""
    rng = _random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _greedy_hitting_set(cliques, n):
    """Greedy hitting set: pick vertex covering most uncovered cliques."""
    # Map each vertex to the set of clique indices it covers
    vertex_cliques = [set() for _ in range(n)]
    for i, c in enumerate(cliques):
        for v in c:
            vertex_cliques[v].add(i)
    
    uncovered = set(range(len(cliques)))
    selected = set()
    
    while uncovered:
        # Find vertex covering most uncovered cliques
        best_v = -1
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = len(uncovered & vertex_cliques[v])
            if count > best_count:
                best_count = count
                best_v = v
        if best_count <= 0:
            break
        selected.add(best_v)
        uncovered -= vertex_cliques[best_v]
    
    return selected

def _try_remove_vertex(selected, cliques, n):
    """Try to remove vertices from selected while maintaining feasibility."""
    selected = set(selected)
    improved = True
    while improved:
        improved = False
        for v in list(selected):
            # Check if removing v still hits all cliques
            feasible = True
            for c in cliques:
                if not (c & (selected - {v})):
                    feasible = False
                    break
            if feasible:
                selected.discard(v)
                improved = True
    return selected

def _local_search(selected, cliques, n, time_limit):
    """Local search: try removals and swaps to minimize size."""
    start_time = time.time()
    selected = set(selected)
    
    # Phase 1: repeated removal
    selected = _try_remove_vertex(selected, cliques, n)
    
    # Phase 2: try swaps (remove v, add u) that enable further removals
    # For now, just do iterative removal which is the main gain
    while time.time() - start_time < time_limit:
        prev_size = len(selected)
        selected = _try_remove_vertex(selected, cliques, n)
        if len(selected) == prev_size:
            break
    
    return selected

def _solve_instance(n, p, seed, time_limit):
    adj = _build_graph(n, p, seed)
    all_cliques = _bron_kerbosch_maximal_cliques(adj, n)
    # Filter to cliques of size >= 2
    cliques = [c for c in all_cliques if len(c) >= 2]
    
    if not cliques:
        return []
    
    # Greedy hitting set
    selected = _greedy_hitting_set(cliques, n)
    
    # Local search
    selected = _local_search(selected, cliques, n, time_limit)
    
    return sorted(selected)

def construct():
    start_time = time.time()
    total_time_limit = 110.0  # Leave margin before 120s
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for i, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        if remaining < 5:
            # Fallback: return all non-isolated vertices
            adj = _build_graph(n, p, seed)
            touched = set()
            for u in range(n):
                if adj[u]:
                    touched.add(u)
            results.append(sorted(touched))
            continue
        
        per_instance_time = min(remaining / (len(instances) - i), 30.0)
        result = _solve_instance(n, p, seed, per_instance_time)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
