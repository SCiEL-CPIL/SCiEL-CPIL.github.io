import time
import random
import itertools
from collections import deque
import numpy as np

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

def _solve_instance_milp(adj, time_limit=10.0):
    """
    Solve the minimum clique hitting set problem using scipy.optimize.milp.
    Returns a list of vertices or None if it fails/times out.
    """
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
        
    num_cliques = len(cliques)
    
    # If the number of cliques is very large, MILP might be slow or memory heavy.
    if num_cliques > 5000:
        return None
        
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None

    # Variables: x_v for v in 0..n-1
    # Objective: Minimize sum(x_v)
    c = np.ones(n)
    
    # Constraints: For each clique i, sum_{v in C_i} x_v >= 1
    # This is equivalent to: -sum_{v in C_i} x_v <= -1
    A = np.zeros((num_cliques, n))
    for i, clique in enumerate(cliques):
        for v in clique:
            A[i, v] = -1
            
    b_lower = -np.ones(num_cliques)
    b_upper = np.zeros(num_cliques)
    
    # Bounds: 0 <= x_v <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(n)
    
    # Run MILP
    start_time = time.time()
    try:
        res = milp(c=c, constraints=LinearConstraint(A, b_lower, b_upper), 
                   integrality=integrality, bounds=bounds, 
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        
        if res.success and res.x is not None:
            # res.x is the optimal solution
            sol = [int(v) for v in res.x if v > 0.5]
            # Verify solution just in case
            sol_set = set(sol)
            for clique in cliques:
                if not (clique & sol_set):
                    return None # Invalid
            return sol
        else:
            return None
    except Exception:
        return None

def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal problem for maximal cliques."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    num_cliques = len(cliques)
    
    # Precompute vertex -> clique indices
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Bitmask representation
    clique_mask = [0] * n
    for v in range(n):
        mask = 0
        for i in vertex_cliques[v]:
            mask |= (1 << i)
        clique_mask[v] = mask
        
    # Sort vertices by coverage (number of cliques) descending
    vertex_freq = [len(vertex_cliques[v]) for v in range(n)]
    
    # Precompute for each clique: list of vertices in it, sorted by frequency descending
    clique_vertices = []
    for i, clique in enumerate(cliques):
        verts = sorted(clique, key=lambda v: vertex_freq[v], reverse=True)
        clique_vertices.append(verts)
        
    best_solution = None
    best_size = num_cliques + 1
    
    # Initial greedy solution
    def greedy_solve():
        S = set()
        uncovered = (1 << num_cliques) - 1
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in S:
                    continue
                count = bin(clique_mask[v] & uncovered).count('1')
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            S.add(best_v)
            uncovered &= ~clique_mask[best_v]
        return S if not uncovered else None
    
    S_greedy = greedy_solve()
    if S_greedy is not None:
        best_size = len(S_greedy)
        best_solution = set(S_greedy)
        
    # Try MILP first if feasible
    start_time = time.time()
    if num_cliques < 2000:
        milp_sol = _solve_instance_milp(adj, time_limit=min(10.0, time_limit/2))
        if milp_sol is not None and len(milp_sol) < best_size:
            best_size = len(milp_sol)
            best_solution = set(milp_sol)
            
    # B&B Search if MILP didn't find a better solution or wasn't used
    # Check if we have time left
    if time.time() - start_time < time_limit:
        def count_bits(x):
            return bin(x).count('1')
        
        def lower_bound(uncovered_mask, current_size):
            if not uncovered_mask:
                return 0
            max_cover = 0
            for v in range(n):
                c = count_bits(clique_mask[v] & uncovered_mask)
                if c > max_cover:
                    max_cover = c
            if max_cover == 0:
                return 1000
            return (count_bits(uncovered_mask) + max_cover - 1) // max_cover
        
        def dfs(current_size, uncovered_mask, path):
            nonlocal best_size, best_solution
            
            if time.time() - start_time > time_limit:
                return False
                
            if uncovered_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = set(path)
                return True
                
            if current_size + lower_bound(uncovered_mask, current_size) >= best_size:
                return False
                
            if current_size >= best_size:
                return False
                
            # Pick the clique with fewest candidate vertices
            best_clique_idx = -1
            best_num_candidates = 1000
            candidates_to_check = []
            um = uncovered_mask
            while um:
                bit = um & -um
                idx = bit.bit_length() - 1
                candidates_to_check.append(idx)
                um ^= bit
                if len(candidates_to_check) >= 10:
                    break
                
            for ci in candidates_to_check:
                verts = clique_vertices[ci]
                num_cand = len(verts)
                if num_cand < best_num_candidates:
                    best_num_candidates = num_cand
                    best_clique_idx = ci
                
            if best_clique_idx == -1:
                target_clique_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            else:
                target_clique_idx = best_clique_idx
                
            verts = clique_vertices[target_clique_idx]
            
            for v in verts:
                new_uncovered = uncovered_mask & ~clique_mask[v]
                
                path.append(v)
                dfs(current_size + 1, new_uncovered, path)
                path.pop()
                
                if best_size <= current_size:
                    return False
                    
        dfs(0, (1 << num_cliques) - 1, [])
        
    # Local search polish
    if best_solution is not None and time.time() - start_time < time_limit - 2:
        polished = set(best_solution)
        improved = True
        while improved:
            improved = False
            for v in list(polished):
                candidate = polished - {v}
                ok = True
                for i, clique in enumerate(cliques):
                    if not (clique & candidate):
                        ok = False
                        break
                if ok:
                    polished = candidate
                    improved = True
                    best_size = len(polished)
                    best_solution = set(polished)
                    
    if best_solution is not None:
        return sorted(best_solution)
    else:
        if S_greedy is not None:
            return sorted(S_greedy)
        return list(range(n))

def construct():
    deadline = time.time() + 145
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    time_weights = [0.25, 0.35, 0.40]
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time() - 3
        if remaining_time < 5:
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            S = set()
            is_covered = [False] * len(cliques)
            vertex_cliques = [[] for _ in range(n)]
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
                    
            remaining = list(range(len(cliques)))
            while remaining:
                best_v = -1
                best_count = -1
                for v in range(n):
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
            
        time_budget = remaining_time * time_weights[idx]
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_budget)
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
