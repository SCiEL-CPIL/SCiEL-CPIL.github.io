import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate Weight 2, 3, and 4 vectors
    # Max norm squared is 4 (from Weight 4)
    # Condition: dist_sq >= 4
    candidates = []
    norms_sq = []
    
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            candidates.append(v)
            norms_sq.append(2)
            
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Determine conflicts
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < 4
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors as sets for fast intersection
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
    
    # Also build a boolean conflict matrix for vectorized operations
    conflict_matrix = conflicts  # shape (n, n), bool
    
    def greedy_min_degree():
        """
        Greedy selection picking the vertex with the minimum number of 
        conflicts with the *remaining* available vertices.
        """
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline - 2.0:
                break
            
            # Find vertex with min degree in available subgraph
            # Sample if large to save time
            if len(available) > 300:
                sample = random.sample(list(available), 300)
            else:
                sample = list(available)
                
            best_v = None
            min_deg = float('inf')
            for v in sample:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            selected.append(best_v)
            available.discard(best_v)
            # Remove neighbors of best_v from available
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
                    
        return selected

    def local_search(S):
        """
        Improve an independent set S using vectorized addition, 1-for-k swaps,
        and 2-for-k swaps.
        """
        S_set = set(S)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to add any vertex that fits (vectorized)
            if S_set:
                S_arr = np.array(list(S_set), dtype=np.int64)
                # A vertex v is compatible if no element of S is in neighbors[v]
                # i.e., conflict_matrix[v, S_arr].all() == False
                compat = ~np.any(conflict_matrix[:, S_arr], axis=1)
                # Mask out already-selected vertices
                compat[S_arr] = False
                addable = np.where(compat)[0]
                
                if len(addable) > 0:
                    # Add all compatible vertices at once
                    for v in addable:
                        S_set.add(v)
                    improved = True
            else:
                # If S is empty, pick the vertex with minimum total degree
                degrees = np.array([len(neighbors[i]) for i in range(n_candidates)])
                v = int(np.argmin(degrees))
                S_set.add(v)
                improved = True
            
            # 2. Try 1-for-k swaps (vectorized)
            if not improved and len(S_set) >= 1:
                S_list = list(S_set)
                random.shuffle(S_list)
                
                for v in S_list:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Remove v temporarily
                    test_set = S_set - {v}
                    if not test_set:
                        test_arr = np.array([], dtype=np.int64)
                    else:
                        test_arr = np.array(list(test_set), dtype=np.int64)
                    
                    # Find all vertices compatible with test_set
                    if len(test_arr) > 0:
                        compat = ~np.any(conflict_matrix[:, test_arr], axis=1)
                    else:
                        compat = np.ones(n_candidates, dtype=bool)
                    
                    # Exclude vertices already in test_set and v
                    compat[test_arr] = False
                    compat[v] = False
                    
                    candidates_for_swap = np.where(compat)[0]
                    
                    if len(candidates_for_swap) < 2:
                        continue
                    
                    # Greedily pick an independent subset from candidates_for_swap
                    available_swap = set(candidates_for_swap.tolist())
                    added_swap = []
                    
                    while available_swap:
                        if time.time() > deadline - 5.0:
                            break
                        if len(available_swap) > 50:
                            sample = random.sample(list(available_swap), 50)
                        else:
                            sample = list(available_swap)
                            
                        best_u = None
                        min_deg = float('inf')
                        for u in sample:
                            deg = len(neighbors[u] & available_swap)
                            if deg < min_deg:
                                min_deg = deg
                                best_u = u
                                
                        if best_u is None:
                            break
                            
                        added_swap.append(best_u)
                        available_swap.discard(best_u)
                        for n in neighbors[best_u]:
                            if n in available_swap:
                                available_swap.discard(n)
                        
                        if len(added_swap) >= 3:
                            break

                    if len(added_swap) > 1:
                        S_set = test_set.union(added_swap)
                        improved = True
            
            # 3. Try 2-for-k swaps (remove 2, add 3+)
            if not improved and len(S_set) >= 2:
                S_list = list(S_set)
                random.shuffle(S_list)
                # Try a limited number of 2-removal pairs
                max_pairs = min(len(S_list), 15)
                for i in range(max_pairs):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, min(i + 3, len(S_list))):
                        if time.time() > deadline - 5.0:
                            break
                        
                        v1, v2 = S_list[i], S_list[j]
                        test_set = S_set - {v1, v2}
                        
                        if not test_set:
                            test_arr = np.array([], dtype=np.int64)
                        else:
                            test_arr = np.array(list(test_set), dtype=np.int64)
                        
                        if len(test_arr) > 0:
                            compat = ~np.any(conflict_matrix[:, test_arr], axis=1)
                        else:
                            compat = np.ones(n_candidates, dtype=bool)
                        
                        compat[test_arr] = False
                        compat[v1] = False
                        compat[v2] = False
                        
                        candidates_for_swap = np.where(compat)[0]
                        
                        if len(candidates_for_swap) < 3:
                            continue
                        
                        # Greedily pick independent subset
                        available_swap = set(candidates_for_swap.tolist())
                        added_swap = []
                        
                        while available_swap:
                            if time.time() > deadline - 5.0:
                                break
                            if len(available_swap) > 50:
                                sample = random.sample(list(available_swap), 50)
                            else:
                                sample = list(available_swap)
                                
                            best_u = None
                            min_deg = float('inf')
                            for u in sample:
                                deg = len(neighbors[u] & available_swap)
                                if deg < min_deg:
                                    min_deg = deg
                                    best_u = u
                                    
                            if best_u is None:
                                break
                                
                            added_swap.append(best_u)
                            available_swap.discard(best_u)
                            for n in neighbors[best_u]:
                                if n in available_swap:
                                    available_swap.discard(n)
                            
                            if len(added_swap) >= 4:
                                break

                        if len(added_swap) > 2:
                            S_set = test_set.union(added_swap)
                            improved = True
                            break  # Restart the outer loop

        return list(S_set)

    def final_polish(S):
        """
        High-density verification pass using bitmask for speed.
        Attempts to add any remaining compatible vertices.
        """
        S_set = set(S)
        if not S_set:
            return list(S_set)
        
        S_arr = np.array(list(S_set), dtype=np.int64)
        compat = ~np.any(conflict_matrix[:, S_arr], axis=1)
        compat[S_arr] = False
        addable = np.where(compat)[0]
        
        for v in addable:
            S_set.add(v)
        
        return list(S_set)

    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Randomized Greedy Restarts
    # Allocate 50% of time to greedy restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.5
    
    num_trials = 0
    while time.time() < greedy_deadline:
        num_trials += 1
        random.seed(42 + num_trials)
        sel = greedy_min_degree()
        if len(sel) > len(best_selected):
            best_selected = sel
            
    # Phase 2: Local Search on the best greedy result
    if best_selected and time.time() < deadline - 5.0:
        # Run local search on the best result found so far
        refined = local_search(best_selected)
        if len(refined) > len(best_selected):
            best_selected = refined
        
        # If time permits, try local search on a few other good greedy results
        if time.time() < deadline - 20.0:
            # One more restart + local search
            random.seed(42 + num_trials + 100)
            sel2 = greedy_min_degree()
            if len(sel2) > len(best_selected) * 0.9:
                refined2 = local_search(sel2)
                if len(refined2) > len(best_selected):
                    best_selected = refined2
        
        # Phase 3: Final polish
        if time.time() < deadline - 3.0:
            polished = final_polish(best_selected)
            if len(polished) > len(best_selected):
                best_selected = polished

    # Construct final array
    if not best_selected:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = candidates[best_selected]
    return result
