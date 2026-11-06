import time
import random
from collections import deque
import sys

def _build_graph(n, p, seed):
    """Build adjacency lists from the same PRNG stream as the seed program."""
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

def _solve_instance(n, p, seed, time_budget):
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []

    num_cliques = len(cliques)
    
    # Bitmask for all cliques
    all_cliques_mask = (1 << num_cliques) - 1
    
    # Precompute for each vertex, the bitmask of cliques it covers
    vertex_covers = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_covers[v] |= bit

    # Helper to count bits
    def popcount(x):
        return x.bit_count()
        
    # 1. Greedy Initialization
    sol_mask = 0
    clique_counts = [0] * num_cliques
    remaining_cliques = all_cliques_mask
    
    while remaining_cliques:
        best_v = -1
        best_count = 0
        for v in range(n):
            if (sol_mask & (1 << v)) != 0:
                continue
            count = popcount(vertex_covers[v] & remaining_cliques)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v == -1 or best_count == 0:
            break
            
        sol_mask |= (1 << best_v)
        cover = vertex_covers[best_v]
        for i in range(num_cliques):
            if (cover & (1 << i)):
                clique_counts[i] += 1
                if clique_counts[i] == 1:
                    remaining_cliques &= ~(1 << i)
                    
    cur_sol_mask = sol_mask
    cur_clique_counts = clique_counts[:]
    cur_unhit = popcount(remaining_cliques)
    cur_size = popcount(cur_sol_mask)
    
    best_sol_mask = cur_sol_mask
    best_clique_counts = cur_clique_counts[:]
    best_size = cur_size
    best_unhit = cur_unhit
    
    start_time = time.time()
    
    def try_remove_and_update():
        """Try to remove any single vertex that keeps solution valid. Returns True if improvement made."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        for v in range(n):
            if (cur_sol_mask & (1 << v)) == 0:
                continue
            new_unhit = cur_unhit
            cover = vertex_covers[v]
            c = cover
            while c:
                i = (c & -c).bit_length() - 1
                c &= c - 1
                if cur_clique_counts[i] == 1:
                    new_unhit += 1
            
            if new_unhit == 0:
                # Commit removal
                cur_sol_mask &= ~(1 << v)
                cover = vertex_covers[v]
                c = cover
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    cur_clique_counts[i] -= 1
                cur_size -= 1
                cur_unhit = new_unhit
                if cur_size < best_size:
                    best_size = cur_size
                    best_sol_mask = cur_sol_mask
                    best_clique_counts = cur_clique_counts[:]
                return True
        return False

    def try_2swap():
        """Try 2-swap: remove 2 vertices, add 2 different vertices to get same or better size with more flexibility.
           Actually for minimization, we want to try removing 2 and adding 1 (net -1) if possible,
           or removing 2 and adding 2 (net 0) to enable further removals.
           Let's try: remove u,v, then greedily add back only what's needed.
           This is a perturbation+repair that can escape local minima."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        if cur_unhit != 0:
            return False
        
        vertices_in_sol = [v for v in range(n) if (cur_sol_mask & (1 << v))]
        if len(vertices_in_sol) < 2:
            return False
        
        # Try all pairs to remove, then repair
        # To limit time, only try a subset of pairs
        max_pairs_to_try = 50
        pair_count = 0
        for i_idx in range(len(vertices_in_sol)):
            for j_idx in range(i_idx + 1, len(vertices_in_sol)):
                pair_count += 1
                if pair_count > max_pairs_to_try:
                    break
                if time.time() - start_time > time_budget:
                    return False
                
                u = vertices_in_sol[i_idx]
                v = vertices_in_sol[j_idx]
                
                # Simulate removal of u and v
                trial_sol = cur_sol_mask & ~(1 << u) & ~(1 << v)
                trial_counts = cur_clique_counts[:]
                
                cover_u = vertex_covers[u]
                c = cover_u
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    trial_counts[i] -= 1
                    
                cover_v = vertex_covers[v]
                c = cover_v
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    trial_counts[i] -= 1
                    
                # Find unhit cliques
                remaining_mask = 0
                for i in range(num_cliques):
                    if trial_counts[i] == 0:
                        remaining_mask |= (1 << i)
                        
                if remaining_mask == 0:
                    # Both can be removed! Net improvement of -2
                    cur_sol_mask = trial_sol
                    cur_clique_counts = trial_counts
                    cur_size -= 2
                    cur_unhit = 0
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol_mask = cur_sol_mask
                        best_clique_counts = cur_clique_counts[:]
                    return True
                    
                # Greedily add back vertices to cover remaining cliques
                # Try to add fewer than 2 vertices
                added = 0
                while remaining_mask and added < 2 and time.time() - start_time < time_budget:
                    best_v = -1
                    best_count = 0
                    for w in range(n):
                        if (trial_sol & (1 << w)) != 0:
                            continue
                        count = popcount(vertex_covers[w] & remaining_mask)
                        if count > best_count:
                            best_count = count
                            best_v = w
                    if best_v == -1 or best_count == 0:
                        break
                    trial_sol |= (1 << best_v)
                    cover = vertex_covers[best_v]
                    c = cover
                    while c:
                        i = (c & -c).bit_length() - 1
                        c &= c - 1
                        trial_counts[i] += 1
                        if trial_counts[i] == 1:
                            remaining_mask &= ~(1 << i)
                    added += 1
                    
                trial_unhit = popcount(remaining_mask)
                trial_size = popcount(trial_sol)
                
                if trial_unhit == 0 and trial_size < cur_size:
                    cur_sol_mask = trial_sol
                    cur_clique_counts = trial_counts
                    cur_size = trial_size
                    cur_unhit = 0
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol_mask = cur_sol_mask
                        best_clique_counts = cur_clique_counts[:]
                    return True
            if pair_count > max_pairs_to_try:
                break
        return False

    def kick_and_repair():
        """Perturb by removing k random vertices, then repair with greedy."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        if cur_unhit != 0:
            return False
        
        vertices_in_sol = [v for v in range(n) if (cur_sol_mask & (1 << v))]
        if len(vertices_in_sol) < 3:
            return False
            
        k = random.randint(2, min(5, len(vertices_in_sol) - 1))
        to_remove = random.sample(vertices_in_sol, k)
        
        trial_sol = cur_sol_mask
        trial_counts = cur_clique_counts[:]
        for v in to_remove:
            trial_sol &= ~(1 << v)
            cover = vertex_covers[v]
            c = cover
            while c:
                i = (c & -c).bit_length() - 1
                c &= c - 1
                trial_counts[i] -= 1
                
        remaining_mask = 0
        for i in range(num_cliques):
            if trial_counts[i] == 0:
                remaining_mask |= (1 << i)
                
        while remaining_mask and time.time() - start_time < time_budget:
            best_v = -1
            best_count = 0
            for v in range(n):
                if (trial_sol & (1 << v)) != 0:
                    continue
                count = popcount(vertex_covers[v] & remaining_mask)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            trial_sol |= (1 << best_v)
            cover = vertex_covers[best_v]
            c = cover
            while c:
                i = (c & -c).bit_length() - 1
                c &= c - 1
                trial_counts[i] += 1
                if trial_counts[i] == 1:
                    remaining_mask &= ~(1 << i)
                    
        trial_unhit = popcount(remaining_mask)
        trial_size = popcount(trial_sol)
        
        if trial_unhit == 0 and trial_size < cur_size:
            cur_sol_mask = trial_sol
            cur_clique_counts = trial_counts
            cur_size = trial_size
            cur_unhit = 0
            if cur_size < best_size:
                best_size = cur_size
                best_sol_mask = cur_sol_mask
                best_clique_counts = cur_clique_counts[:]
            return True
        return False

    # Main local search loop
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        
        # Phase 1: Try single removals
        if try_remove_and_update():
            improved = True
            continue
            
        # Phase 2: Try 2-swap moves
        if try_2swap():
            improved = True
            continue
            
        # Phase 3: Kick and repair
        if kick_and_repair():
            improved = True
            continue

    result = [v for v in range(n) if (best_sol_mask & (1 << v))]
    return result

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time: give more to larger/harder instances
    # n=30,p=0.3: denser, more cliques, harder
    # n=35,p=0.25: medium
    # n=40,p=0.2: sparser but more vertices
    time_allocs = [45, 40, 45]  # Total 130s, leaving 15s buffer
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 8:
             # Fallback: quick greedy
             adj = _build_graph(n, p, seed)
             cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
             if not cliques:
                 results.append([])
                 continue
             sol = []
             for clique in cliques:
                 if clique & set(sol):
                     continue
                 v = min(clique)  # Pick smallest vertex for determinism
                 sol.append(v)
             results.append(sorted(set(sol)))
             continue
         
        alloc = min(time_allocs[i], remaining_time - 5)
        
        sol = _solve_instance(n, p, seed, alloc)
        results.append(sol)
        
    return results
