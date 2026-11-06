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

def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal problem for maximal cliques."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    # Precompute vertex -> clique indices mapping
    num_cliques = len(cliques)
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Convert to numpy for faster operations
    vc_arrays = [np.array(vc, dtype=np.int32) for vc in vertex_cliques]
    
    def is_valid(cover):
        return np.all(cover >= 1)
    
    def compute_cover(S):
        cover = np.zeros(num_cliques, dtype=np.int32)
        for v in S:
            cover[vc_arrays[v]] += 1
        return cover
    
    # Greedy Initialization
    S = set()
    is_covered = np.zeros(num_cliques, dtype=bool)
    remaining = num_cliques
    
    # Precompute uncovered count for each vertex
    uncovered_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        uncovered_count[v] = len(vc_arrays[v])
    
    while remaining > 0:
        # Find vertex covering most uncovered cliques
        best_v = -1
        best_score = -1
        for v in range(n):
            if v in S:
                continue
            # Count how many uncovered cliques v covers
            if len(vc_arrays[v]) == 0:
                continue
            score = np.sum(~is_covered[vc_arrays[v]])
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1 or best_score == 0:
            break
        
        S.add(best_v)
        for ci in vc_arrays[best_v]:
            if not is_covered[ci]:
                is_covered[ci] = True
                remaining -= 1
                # Update uncovered_count for other vertices in this clique
                for v in cliques[ci]:
                    if v not in S:
                        uncovered_count[v] -= 1
    
    # Local Search with optimized swap check
    current_S = set(S)
    current_cover = compute_cover(current_S)
    best_S = set(current_S)
    best_size = len(current_S)
    
    start_time = time.time()
    
    def try_remove(v, cover):
        """Check if removing v keeps solution valid. Returns new cover or None."""
        idxs = vc_arrays[v]
        if len(idxs) == 0:
            return cover.copy()
        new_cover = cover.copy()
        new_cover[idxs] -= 1
        if np.all(new_cover >= 1):
            return new_cover
        return None
    
    def try_swap(v_in, v_out, cover):
        """Check if swapping v_in for v_out keeps solution valid."""
        idxs_in = vc_arrays[v_in]
        idxs_out = vc_arrays[v_out]
        
        new_cover = cover.copy()
        if len(idxs_in) > 0:
            new_cover[idxs_in] -= 1
        if len(idxs_out) > 0:
            new_cover[idxs_out] += 1
        
        if np.all(new_cover >= 1):
            return new_cover
        return None
    
    # Phase 1: Greedy removals
    deadline_phase1 = start_time + time_limit * 0.25
    while time.time() < deadline_phase1:
        improved = False
        for v in list(current_S):
            new_cover = try_remove(v, current_cover)
            if new_cover is not None:
                current_S.remove(v)
                current_cover = new_cover
                best_S = set(current_S)
                best_size = len(current_S)
                improved = True
                break
        if not improved:
            break
    
    # Phase 2: Systematic swap search (single swaps)
    deadline_phase2 = start_time + time_limit * 0.5
    while time.time() < deadline_phase2:
        improved = False
        
        # Try all single swaps
        for v_in in list(current_S):
            for v_out in range(n):
                if v_out in current_S:
                    continue
                new_cover = try_swap(v_in, v_out, current_cover)
                if new_cover is not None:
                    current_S.remove(v_in)
                    current_S.add(v_out)
                    current_cover = new_cover
                    improved = True
                    break
            if improved:
                break
        
        # After swaps, try removals again
        for v in list(current_S):
            new_cover = try_remove(v, current_cover)
            if new_cover is not None:
                current_S.remove(v)
                current_cover = new_cover
                best_S = set(current_S)
                best_size = len(current_S)
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 3: 2-exchange local search (remove 2, add 1 or 2)
    deadline_phase3 = start_time + time_limit * 0.75
    while time.time() < deadline_phase3:
        improved = False
        
        # Try removing 2 vertices and adding 1
        s_list = list(current_S)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                v1, v2 = s_list[i], s_list[j]
                idxs1 = vc_arrays[v1]
                idxs2 = vc_arrays[v2]
                
                new_cover = current_cover.copy()
                if len(idxs1) > 0:
                    new_cover[idxs1] -= 1
                if len(idxs2) > 0:
                    new_cover[idxs2] -= 1
                
                # Check if any single vertex can fix it
                for v_new in range(n):
                    if v_new in current_S:
                        continue
                    idxs_new = vc_arrays[v_new]
                    if len(idxs_new) == 0:
                        continue
                    
                    test_cover = new_cover.copy()
                    test_cover[idxs_new] += 1
                    
                    if np.all(test_cover >= 1):
                        # Valid 2-for-1 swap
                        current_S.discard(v1)
                        current_S.discard(v2)
                        current_S.add(v_new)
                        current_cover = test_cover
                        best_S = set(current_S)
                        best_size = len(current_S)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Try removing 1 and adding 2 (might help in some cases)
        if not improved:
            for v_in in list(current_S):
                idxs_in = vc_arrays[v_in]
                
                new_cover = current_cover.copy()
                if len(idxs_in) > 0:
                    new_cover[idxs_in] -= 1
                
                # Find best pair to add
                best_pair = None
                best_gain = -1
                for v1 in range(n):
                    if v1 in current_S:
                        continue
                    idxs1 = vc_arrays[v1]
                    if len(idxs1) == 0:
                        continue
                    test1 = new_cover.copy()
                    test1[idxs1] += 1
                    gain1 = np.sum(test1 >= 1) - np.sum(new_cover >= 1)
                    if gain1 > 0:
                        for v2 in range(n):
                            if v2 in current_S or v2 == v1:
                                continue
                            idxs2 = vc_arrays[v2]
                            if len(idxs2) == 0:
                                continue
                            test2 = test1.copy()
                            test2[idxs2] += 1
                            if np.all(test2 >= 1):
                                # Valid 1-for-2 swap
                                current_S.remove(v_in)
                                current_S.add(v1)
                                current_S.add(v2)
                                current_cover = test2
                                best_S = set(current_S)
                                best_size = len(current_S)
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    # Phase 4: Random restarts with perturbation
    rng = random.Random(42)
    while time.time() < time_limit:
        # Start from best_S, perturb
        perturbed_S = set(best_S)
        k_remove = rng.randint(1, min(3, len(perturbed_S)))
        to_remove = rng.sample(list(perturbed_S), k_remove)
        for v in to_remove:
            perturbed_S.remove(v)
        
        # Repair: greedily add vertices to make valid
        cover = compute_cover(perturbed_S)
        max_repair = 100
        while not is_valid(cover) and max_repair > 0:
            max_repair -= 1
            uncovered_cliques = np.where(cover == 0)[0]
            if len(uncovered_cliques) == 0:
                break
            # Pick vertex covering most uncovered cliques
            best_v = -1
            best_score = -1
            for v in range(n):
                if v in perturbed_S:
                    continue
                idxs = vc_arrays[v]
                if len(idxs) == 0:
                    continue
                score = np.sum(cover[idxs] == 0)
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v == -1:
                break
            perturbed_S.add(best_v)
            cover[vc_arrays[best_v]] += 1
        
        if not is_valid(cover):
            continue
        
        # Local search on perturbed_S
        local_S = set(perturbed_S)
        local_cover = compute_cover(local_S)
        
        # Greedy removals
        for _ in range(30):
            improved = False
            for v in list(local_S):
                new_cover = try_remove(v, local_cover)
                if new_cover is not None:
                    local_S.remove(v)
                    local_cover = new_cover
                    improved = True
                    break
            if not improved:
                break
        
        # Single swaps
        for _ in range(20):
            improved = False
            for v_in in list(local_S):
                for v_out in range(n):
                    if v_out in local_S:
                        continue
                    new_cover = try_swap(v_in, v_out, local_cover)
                    if new_cover is not None:
                        local_S.remove(v_in)
                        local_S.add(v_out)
                        local_cover = new_cover
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        # 2-exchange
        for _ in range(5):
            improved = False
            s_list = list(local_S)
            for i in range(len(s_list)):
                for j in range(i + 1, len(s_list)):
                    v1, v2 = s_list[i], s_list[j]
                    idxs1 = vc_arrays[v1]
                    idxs2 = vc_arrays[v2]
                    
                    new_cover = local_cover.copy()
                    if len(idxs1) > 0:
                        new_cover[idxs1] -= 1
                    if len(idxs2) > 0:
                        new_cover[idxs2] -= 1
                    
                    for v_new in range(n):
                        if v_new in local_S:
                            continue
                        idxs_new = vc_arrays[v_new]
                        if len(idxs_new) == 0:
                            continue
                        
                        test_cover = new_cover.copy()
                        test_cover[idxs_new] += 1
                        
                        if np.all(test_cover >= 1):
                            local_S.discard(v1)
                            local_S.discard(v2)
                            local_S.add(v_new)
                            local_cover = test_cover
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        if len(local_S) < best_size:
            best_S = set(local_S)
            best_size = len(local_S)
    
    return sorted(best_S)

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time: first instance gets less, later ones get more
    time_weights = [0.25, 0.35, 0.40]
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time() - 2  # 2s safety margin
        if remaining_time < 5:
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
