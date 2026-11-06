"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _bron_kerbosch(R, P, X, adj, cliques_out):
    """Recursive Bron-Kerbosch with pivot to find maximal cliques."""
    if not P and not X:
        if len(R) >= 2:
            cliques_out.append(frozenset(R))
        return
    
    # Choose pivot to minimize branching
    best_pivot = None
    best_count = -1
    candidates = list(P) + list(X)
    for v in candidates:
        cnt = len(P & adj[v])
        if cnt > best_count:
            best_count = cnt
            best_pivot = v
    
    if best_pivot is None:
        candidates_to_iter = list(P)
    else:
        candidates_to_iter = list(P - adj[best_pivot])
    
    for v in candidates_to_iter:
        new_R = R | {v}
        new_P = P & adj[v]
        new_X = X & adj[v]
        _bron_kerbosch(new_R, new_P, new_X, adj, cliques_out)
        P = P - {v}
        X = X | {v}

def _greedy_hitting_set(cliques, n):
    """Greedy set cover: repeatedly pick vertex hitting most uncovered cliques."""
    uncovered = [set(c) for c in cliques]
    if not uncovered:
        return set()
    
    # Build vertex -> list of clique indices it covers
    vertex_cliques = {v: [] for v in range(n)}
    for i, c in enumerate(uncovered):
        for v in c:
            vertex_cliques[v].append(i)
    
    covered = [False] * len(uncovered)
    selected = set()
    remaining = len(uncovered)
    
    while remaining > 0:
        best_v = None
        best_gain = -1
        for v in range(n):
            gain = 0
            for ci in vertex_cliques[v]:
                if not covered[ci]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v is None or best_gain == 0:
            break
        selected.add(best_v)
        for ci in vertex_cliques[best_v]:
            if not covered[ci]:
                covered[ci] = True
                remaining -= 1
    return selected

def _local_search_improve(selected, cliques, n, max_iter=5000):
    """Local search: try removing, adding, swapping to reduce size."""
    selected = set(selected)
    
    def is_valid(s):
        for c in cliques:
            if not c & s:
                return False
        return True
    
    if not is_valid(selected):
        return selected
    
    best = set(selected)
    for _ in range(max_iter):
        improved = False
        
        # Try removing a vertex
        for v in list(best):
            candidate = best - {v}
            if is_valid(candidate):
                best = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add one
        for v_remove in list(best):
            for v_add in range(n):
                if v_add in best:
                    continue
                candidate = (best - {v_remove}) | {v_add}
                if len(candidate) < len(best) and is_valid(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _solve_instance(n, p, seed, time_limit=30):
    """Solve one instance: build graph, find maximal cliques, find hitting set."""
    start_time = time.time()
    
    # Build graph
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    # Find all maximal cliques of size >= 2
    cliques = []
    _bron_kerbosch(set(), set(range(n)), set(), adj, cliques)
    
    if not cliques:
        return []
    
    # Greedy hitting set
    solution = _greedy_hitting_set(cliques, n)
    
    # Local search improvement
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.5:
        solution = _local_search_improve(solution, cliques, n, max_iter=10000)
    
    # Verify
    for c in cliques:
        if not c & solution:
            # Fallback: add all vertices in uncovered cliques
            solution = solution | set(c)
    
    return sorted(solution)

def construct():
    total_deadline = 130  # seconds
    start = time.time()
    results = []
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for i, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start
        remaining = total_deadline - elapsed
        if remaining < 5:
            # Not enough time; return what we have plus empty lists
            while len(results) < 3:
                results.append([])
            return results
        
        per_instance_limit = min(remaining - 2, 40)
        sol = _solve_instance(n, p, seed, time_limit=per_instance_limit)
        results.append(sol)
    
    return results
# EVOLVE-BLOCK-END
