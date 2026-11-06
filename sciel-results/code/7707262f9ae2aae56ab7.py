import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P union X to reduce recursion
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal problem for maximal cliques."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    num_cliques = len(cliques)
    
    # Precompute for each vertex, the list of cliques it covers
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Filter out vertices that don't cover any clique (isolated vertices in the context of cliques)
    # Actually, any vertex in a clique covers that clique. Isolated vertices (degree 0) are in no clique.
    # But we only need to consider vertices that are in at least one clique.
    relevant_vertices = [v for v in range(n) if vertex_cliques[v]]
    
    if not relevant_vertices:
        return []
    
    # Greedy Initialization
    S = set()
    is_covered = [False] * num_cliques
    remaining_cliques = list(range(num_cliques))
    
    # To speed up, we can use a priority queue or just scan.
    # Scanning is O(n * num_cliques) per step. n<=40, num_cliques can be large but usually manageable.
    # Let's optimize the inner loop.
    
    # We'll maintain a list of uncovered cliques.
    # For each vertex, we want to know how many uncovered cliques it covers.
    # Let's precompute the count of cliques per vertex (total).
    # Then update dynamically.
    
    # Actually, simple greedy scanning is fine for n=40.
    
    while remaining_cliques:
        best_v = -1
        best_score = -1
        for v in relevant_vertices:
            if v in S:
                continue
            count = 0
            for ci in vertex_cliques[v]:
                if not is_covered[ci]:
                    count += 1
            if count > best_score:
                best_score = count
                best_v = v
            elif count == best_score and count > 0:
                # Tie break: prefer vertex with higher degree in original graph?
                # Or lower index? Let's use lower index for determinism.
                if v < best_v:
                    best_v = v
        
        if best_v == -1 or best_score == 0:
            break
            
        S.add(best_v)
        for ci in vertex_cliques[best_v]:
            if not is_covered[ci]:
                is_covered[ci] = True
                remaining_cliques.remove(ci)
                
    best_solution = list(S)
    best_size = len(best_solution)
    
    # Local Search / Simulated Annealing
    current_S = set(best_solution)
    current_cover = [0] * num_cliques
    for v in current_S:
        for ci in vertex_cliques[v]:
            current_cover[ci] += 1
            
    def is_valid(cover):
        return min(cover) >= 1 if cover else True

    if not is_valid(current_cover):
         # Fallback
         pass

    start_time = time.time()
    current_size = len(current_S)
    
    # 1. Greedy Removals
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.4:
        improved = False
        for v in list(current_S):
            # Try removing v
            # Check if any clique becomes uncovered
            valid = True
            for ci in vertex_cliques[v]:
                if current_cover[ci] == 1:
                    valid = False
                    break
            if valid:
                current_S.remove(v)
                for ci in vertex_cliques[v]:
                    current_cover[ci] -= 1
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_S)
                break # Restart loop as cover counts changed
    
    # 2. Simulated Annealing / Random Walk
    if time.time() - start_time < time_limit * 0.8:
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        current_S = set(best_solution)
        current_cover = [0] * num_cliques
        for v in current_S:
            for ci in vertex_cliques[v]:
                current_cover[ci] += 1
                
        current_energy = len(current_S)
        best_energy = best_size
        best_S = set(best_solution)
        
        # We will use a random walk that allows swaps and additions/removals.
        # Since we want to minimize size, we primarily look for removals.
        # Swaps allow us to move to a different local minimum.
        
        while T > T_min and (time.time() - start_time) < time_limit:
            # Try to find a removal first
            removed = False
            if current_S:
                v = random.choice(list(current_S))
                valid = True
                for ci in vertex_cliques[v]:
                    if current_cover[ci] == 1:
                        valid = False
                        break
                if valid:
                    current_S.remove(v)
                    for ci in vertex_cliques[v]:
                        current_cover[ci] -= 1
                    current_energy -= 1
                    removed = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_S = set(current_S)
                        
            if not removed:
                # Try a swap
                if current_S:
                    v_in = random.choice(list(current_S))
                    candidates_out = [u for u in relevant_vertices if u not in current_S]
                    if candidates_out:
                        v_out = random.choice(candidates_out)
                        
                        # Check swap validity
                        # Remove v_in, add v_out
                        valid = True
                        for ci in vertex_cliques[v_in]:
                            if current_cover[ci] == 1:
                                # This clique would be uncovered by v_in removal
                                # Unless v_out also covers it
                                if v_out not in cliques[ci]: # Note: cliques[ci] is a frozenset of vertices
                                    valid = False
                                    break
                        
                        if valid:
                            # Perform swap
                            current_S.remove(v_in)
                            current_S.add(v_out)
                            for ci in vertex_cliques[v_in]:
                                current_cover[ci] -= 1
                            for ci in vertex_cliques[v_out]:
                                current_cover[ci] += 1
                            # Energy unchanged
                            
            T *= cooling
    
    return sorted(best_S)

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    time_per_instance = (deadline - time.time()) / len(instances)
    
    for n, p, seed in instances:
        if time.time() > deadline - 5:
            # Fallback: simple greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            S = set()
            is_covered = [False] * len(cliques)
            vertex_cliques = [[] for _ in range(n)]
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
                    
            relevant_vertices = [v for v in range(n) if vertex_cliques[v]]
            remaining = list(range(len(cliques)))
            while remaining:
                best_v = -1
                best_count = -1
                for v in relevant_vertices:
                    if v in S: continue
                    cnt = sum(1 for ci in vertex_cliques[v] if not is_covered[ci])
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v == -1: break
                S.add(best_v)
                for ci in vertex_cliques[best_v]:
                    if not is_covered[ci]:
                        is_covered[ci] = True
                        remaining.remove(ci)
            results.append(sorted(S))
            continue
            
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_per_instance - 1)
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
