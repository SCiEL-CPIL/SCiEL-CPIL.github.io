"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 145.0  # wall-clock limit, leave 5s margin

def _elapsed():
    return time.time() - _START_TIME

def _build_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
    n = len(adj)
    maximal_cliques = []
    
    def expand(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                maximal_cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        if P:
            pivot_pool = P | X
        else:
            pivot_pool = X
        
        if not pivot_pool:
            candidates = list(P)
        else:
            best_pivot = max(pivot_pool, key=lambda u: len(P & adj[u]))
            candidates = [v for v in P if best_pivot not in adj[v]]
        
        for v in candidates:
            if _elapsed() > DEADLINE - 5.0:
                return
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            expand(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
    
    expand(set(), set(range(n)), set())
    return maximal_cliques

def _solve_instance(n, p, seed, time_budget=40.0):
    """Solve one instance: find minimum vertex set meeting all maximal cliques of size >= 2."""
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch(adj)
    
    if not cliques:
        return []
    
    all_vertices = set(range(n))
    num_cliques = len(cliques)
    
    # Precompute: for each vertex, which cliques it belongs to
    vertex_cliques = {v: [] for v in all_vertices}
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    def _greedy_init(rng):
        """Greedy set cover with random tie-breaking."""
        covered = [0] * num_cliques
        selected = []
        remaining_cliques = num_cliques
        
        # Create a list of vertices to process, potentially shuffled
        vertices = list(all_vertices)
        rng.shuffle(vertices)
        
        while remaining_cliques > 0:
            if _elapsed() > time_budget + _START_TIME - 5.0:
                break
            best_v = None
            best_gain = -1
            # We need to pick the vertex that covers the most *new* cliques
            # To be efficient, we can just iterate. n is small.
            for v in vertices:
                gain = 0
                for ci in vertex_cliques[v]:
                    if covered[ci] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                elif gain == best_gain and gain > 0:
                    # Random tie break
                    if rng.random() < 0.5:
                        best_v = v
                        
            if best_v is None or best_gain == 0:
                # Should not happen if graph is connected enough, but safety
                break
                
            selected.append(best_v)
            for ci in vertex_cliques[best_v]:
                if covered[ci] == 0:
                    remaining_cliques -= 1
                covered[ci] += 1
            vertices.remove(best_v)
            
        return selected, covered

    def _local_search(selected, covered):
        """Local search with cover counts."""
        selected_set = set(selected)
        # Rebuild cover counts from selected_set to be safe
        covered = [0] * num_cliques
        for v in selected_set:
            for ci in vertex_cliques[v]:
                covered[ci] += 1
        
        improved = True
        while improved:
            if _elapsed() > time_budget + _START_TIME - 5.0:
                break
            improved = False
            
            # 1. Try to remove vertices (reduction)
            for v in list(selected_set):
                # Check if removing v leaves any clique uncovered
                # A clique is uncovered if covered[ci] == 1 and v is in it
                # Actually, if covered[ci] == 1, and v is in clique ci, then removing v makes covered[ci] 0.
                removable = True
                for ci in vertex_cliques[v]:
                    if covered[ci] == 1:
                        removable = False
                        break
                if removable:
                    selected_set.remove(v)
                    for ci in vertex_cliques[v]:
                        covered[ci] -= 1
                    improved = True
                    break # Restart loop as state changed
            
            if improved:
                continue
                
            # 2. Try swaps (exchange)
            # For each v in S, find cliques that would be lost (covered[ci] == 1)
            # Find a vertex u not in S that covers ALL those cliques.
            for v in list(selected_set):
                if _elapsed() > time_budget + _START_TIME - 5.0:
                    break
                lost_cliques = []
                for ci in vertex_cliques[v]:
                    if covered[ci] == 1:
                        lost_cliques.append(ci)
                
                if not lost_cliques:
                    continue # Already handled by reduction step, or redundant
                
                # Find candidate u
                # u must be in all cliques in lost_cliques
                # Intersect the sets of vertices in those cliques
                # Optimization: pick the smallest clique in lost_cliques to iterate
                min_clique = min(cliques[ci] for ci in lost_cliques)
                
                found_swap = False
                for u in min_clique:
                    if u in selected_set:
                        continue
                    # Check if u covers all lost_cliques
                    valid_swap = True
                    for ci in lost_cliques:
                        if u not in cliques[ci]:
                            valid_swap = False
                            break
                    if valid_swap:
                        # Perform swap
                        selected_set.remove(v)
                        for ci in vertex_cliques[v]:
                            covered[ci] -= 1
                        selected_set.add(u)
                        for ci in vertex_cliques[u]:
                            covered[ci] += 1
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break # Restart loop as state changed
                    
        return list(selected_set)

    # Run multiple restarts
    best_solution = []
    best_size = float('inf')
    
    rng = random.Random(seed)
    
    # Allocate time based on instance index or just fixed budget
    # Total time 145s. 3 instances. ~45s each.
    
    start_time = time.time()
    max_time_for_instance = 45.0
    
    while time.time() - start_time < max_time_for_instance:
        if _elapsed() > DEADLINE - 10.0:
            break
        
        sel, cov = _greedy_init(rng)
        sel = _local_search(sel, cov)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_solution = sel
            
        # If we found a very small solution, we might be close to optimal
        # Heuristic: if size is small, we can stop early? No, keep searching.
        
        # Diversification: random restarts handle this.
        
    return best_solution

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if _elapsed() > DEADLINE - 20.0:
            # Not enough time left; return empty lists as fallback (invalid but avoids timeout)
            # But we should try to return something valid if possible.
            # If we run out of time, we return the best we have.
            # If we haven't solved it, we return [] which is invalid.
            # Better to solve quickly.
            pass
            
        sol = _solve_instance(n, p, seed)
        results.append(sol)
        
    return results
# EVOLVE-BLOCK-END
