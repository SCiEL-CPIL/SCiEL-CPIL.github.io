"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

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
    
    # Precompute neighbors
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
        
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
        Improve an independent set S using addition and 1-for-k swaps.
        """
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to add any vertex that fits
            for v in range(n_candidates):
                if time.time() > deadline - 5.0:
                    break
                if v not in S_set:
                    if not (neighbors[v] & S_set):
                        S_set.add(v)
                        improved = True
            
            # 2. Try 1-for-k swaps
            if not improved:
                # Iterate over a copy to allow modification
                for v in list(S_set):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Remove v temporarily
                    test_set = S_set - {v}
                    
                    # Find all vertices that are compatible with test_set
                    # These are candidates for the swap
                    # Note: v's neighbors were blocked, but now they might be free
                    # Also, other vertices not neighbors of v might be free
                    candidates_for_swap = []
                    for u in range(n_candidates):
                        if u not in test_set and u != v:
                            if not (neighbors[u] & test_set):
                                candidates_for_swap.append(u)
                    
                    if not candidates_for_swap:
                        continue
                        
                    # Greedily pick an independent subset from candidates_for_swap
                    # We want at least 2 to improve size (remove 1, add 2 => net +1)
                    # If we can only add 1, it's a wash, but might help future moves? 
                    # Usually we prefer size increase.
                    
                    # Sort candidates by degree in the subgraph induced by candidates_for_swap?
                    # Or just greedy min-degree on the induced subgraph.
                    
                    available_swap = set(candidates_for_swap)
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
                        
                        # Early stop if we have enough to justify the swap
                        if len(added_swap) >= 3: 
                             break

                    if len(added_swap) > 1:
                        S_set = test_set.union(added_swap)
                        improved = True

        return list(S_set)

    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Randomized Greedy Restarts
    # Allocate 60% of time to greedy restarts
    greedy_deadline = start_time + (deadline - start_time) * 0.6
    
    num_trials = 0
    while time.time() < greedy_deadline:
        num_trials += 1
        random.seed(42 + num_trials)
        sel = greedy_min_degree()
        if len(sel) > len(best_selected):
            best_selected = sel
            
    # Phase 2: Local Search on the best greedy result
    if best_selected and time.time() < deadline - 10.0:
        refined = local_search(best_selected)
        if len(refined) > len(best_selected):
            best_selected = refined
            
    # Phase 3: Try a few more greedy restarts if time remains, and local search on them
    if time.time() < deadline - 10.0:
         while time.time() < deadline - 10.0:
             num_trials += 1
             random.seed(1000 + num_trials)
             sel = greedy_min_degree()
             if len(sel) > len(best_selected):
                 best_selected = sel
                 # Quick local search on new best
                 if time.time() < deadline - 10.0:
                     refined = local_search(best_selected)
                     if len(refined) > len(best_selected):
                         best_selected = refined
             else:
                 # If greedy didn't beat best, maybe local search on this greedy result 
                 # could beat the global best? (Unlikely but possible if greedy was lucky)
                 # To save time, only do local search if greedy result is close to best
                 if len(sel) > len(best_selected) - 5:
                     refined = local_search(sel)
                     if len(refined) > len(best_selected):
                         best_selected = refined

    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]

# EVOLVE-BLOCK-END
