"""Clique-transversal solver: minimize vertices meeting every maximal clique (size>=2)."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency bitmasks from the same PRNG stream as the verifier."""
    rng = random.Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _bron_kerbosch(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        U = P | X
        if not U:
            return
        # Pivot: vertex in P∪X with max neighbors in P
        pivot = max(U, key=lambda u: bin(P & adj[u]).count('1'))
        candidates = P & ~adj[pivot]
        while candidates:
            # Pick lowest set bit
            v = (candidates & -candidates).bit_length() - 1
            candidates &= candidates - 1
            R.add(v)
            bk(R, P & adj[v], X & adj[v])
            R.remove(v)
            P &= ~(1 << v)
            X |= (1 << v)
    
    bk(set(), (1 << n) - 1, 0)
    return cliques

def _greedy_set_cover(cliques, n):
    """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
    if not cliques:
        return []
    
    m = len(cliques)
    # vertex -> bitmask of cliques it covers
    vertex_cliques = [0] * n
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v] |= (1 << i)
    
    uncovered = (1 << m) - 1
    selected = []
    
    while uncovered:
        best_v = -1
        best_count = 0
        for v in range(n):
            if not (uncovered >> v) & 0:  # placeholder, we check differently
                pass
            cnt = bin(uncovered & vertex_cliques[v]).count('1')
            if cnt > best_count:
                best_count = cnt
                best_v = v
        if best_v < 0 or best_count == 0:
            break
        selected.append(best_v)
        uncovered &= ~vertex_cliques[best_v]
    
    return sorted(selected)

def _try_remove_vertices(cliques, selected, n):
    """Try to remove vertices from selected while maintaining coverage."""
    m = len(cliques)
    # For each vertex in selected, compute which cliques it covers
    # Coverage: for each clique, which selected vertices are in it
    selected_set = set(selected)
    
    # Build clique -> set of selected vertices in it
    clique_covers = []
    for clique in cliques:
        covers = set(selected_set) & set(clique)
        clique_covers.append(covers)
    
    current = set(selected)
    changed = True
    while changed:
        changed = False
        for v in list(current):
            # Check if removing v leaves all cliques covered
            ok = True
            for covers in clique_covers:
                if v in covers and len(covers) == 1:
                    ok = False
                    break
            if ok:
                current.remove(v)
                # Update clique_covers
                for i, covers in enumerate(clique_covers):
                    if v in covers:
                        clique_covers[i] = covers - {v}
                changed = True
    
    return sorted(current)

def _local_search(cliques, selected, n, time_budget=2.0):
    """Local search: try removals and 1-for-1 swaps to reduce size."""
    start = time.time()
    current = set(selected)
    m = len(cliques)
    
    def is_valid(sol):
        sol_set = set(sol)
        for clique in cliques:
            if not (sol_set & set(clique)):
                return False
        return True
    
    # First, try removals
    current = set(_try_remove_vertices(cliques, list(current), n))
    
    # Try swaps: remove v, add u
    for _ in range(200):
        if time.time() - start > time_budget:
            break
        improved = False
        for v in list(current):
            trial = current - {v}
            # Check if trial is valid; if not, find a vertex u to add
            # Find uncovered cliques
            uncovered_cliques = []
            for i, clique in enumerate(cliques):
                if not (trial & set(clique)):
                    uncovered_cliques.append(clique)
            if not uncovered_cliques:
                current = trial
                improved = True
                break
            # Find u that covers most uncovered cliques
            best_u = -1
            best_cnt = 0
            for u in range(n):
                if u in trial:
                    continue
                cnt = sum(1 for c in uncovered_cliques if u in c)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_u = u
            if best_u >= 0 and best_cnt > 0:
                trial2 = trial | {best_u}
                if len(trial2) < len(current):
                    if is_valid(trial2):
                        current = trial2
                        improved = True
                        break
        if not improved:
            break
    
    return sorted(current)

def _solve_instance(adj, time_budget=10.0):
    """Solve one instance: enumerate cliques, greedy, then local search."""
    start = time.time()
    n = len(adj)
    cliques = _bron_kerbosch(adj, min_size=2)
    
    if not cliques:
        return []
    
    # Greedy set cover
    sol = _greedy_set_cover(cliques, n)
    
    # Local search to improve
    remaining = time_budget - (time.time() - start)
    if remaining > 0.5:
        sol2 = _local_search(cliques, sol, n, time_budget=min(remaining, 5.0))
        if len(sol2) < len(sol):
            sol = sol2
    
    return sol

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 15:
            # Fallback: quick greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch(adj, min_size=2)
            if not cliques:
                results.append([])
            else:
                results.append(_greedy_set_cover(cliques, n))
            continue
        
        adj = _build_graph(n, p, seed)
        remaining = deadline - time.time() - 5
        sol = _solve_instance(adj, time_budget=min(remaining, 40.0))
        results.append(sol)
    
    return results

# EVOLVE-BLOCK-END
