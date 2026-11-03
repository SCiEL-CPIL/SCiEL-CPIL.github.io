import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Problem parameters are fixed by the task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Precompute neighbors
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    neighbors = []
    for i in range(N):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < q:
                    new_word = w.copy()
                    new_word[pos] = new_val
                    idx = word_to_idx[tuple(new_word)]
                    neigh.append(idx)
        neighbors.append(neigh)
        
    # Convert neighbors to sets for faster subset checks if needed, 
    # but lists are fine for size 15.
    # We will use lists for iteration, and sets for membership if necessary.
    # Actually, for "is i in neighbors[k]", list is O(15). Set is O(1) but overhead of creation.
    # Let's keep lists.

    best_size = N
    best_S = set()
    
    # Helper to update coverage
    def update_coverage(S, cover_count, add=True):
        # This is for initial setup or bulk updates
        pass

    # Helper to check validity
    def is_valid(cover_count):
        return np.all(cover_count > 0)

    # Greedy Initialization
    def greedy_init():
        S = set()
        cover_count = np.zeros(N, dtype=np.int32)
        # To speed up greedy, we can track uncovered words
        # But N is small, simple loop is okay.
        
        # Randomize the order of candidates to break symmetry?
        # Or use a heuristic: pick word covering most uncovered.
        
        # We'll use a priority queue or just scan.
        # Scanning N=2187 is fast.
        
        while True:
            # Check if all covered
            if np.all(cover_count > 0):
                break
                
            best_j = -1
            best_gain = -1
            
            # Optimize: Only consider words that cover at least one uncovered word?
            # We can precompute for each uncovered word i, the candidates j.
            # But cover_count changes.
            
            # Simple scan
            for j in range(N):
                if j in S:
                    continue
                # Calculate gain: number of neighbors of j with cover_count == 0
                gain = 0
                for i in neighbors[j]:
                    if cover_count[i] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    
            if best_j == -1 or best_gain == 0:
                # Should not happen if not all covered
                break
                
            S.add(best_j)
            for i in neighbors[best_j]:
                cover_count[i] += 1
                
        return S, cover_count

    # Local Search
    def local_search(S, cover_count):
        nonlocal best_size, best_S
        
        stuck_counter = 0
        local_best_size = len(S)
        local_best_S = set(S)
        
        while time.time() < deadline:
            # 1. Removal Phase
            removed_any = False
            while True:
                redundant_found = False
                # Iterate in random order to avoid bias
                for j in random.sample(list(S), len(S)):
                    # Check if j is redundant
                    is_redundant = True
                    for i in neighbors[j]:
                        if cover_count[i] <= 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        S.remove(j)
                        for i in neighbors[j]:
                            cover_count[i] -= 1
                        removed_any = True
                        redundant_found = True
                        break # Restart scan
                
                if not redundant_found:
                    break
            
            if len(S) < local_best_size:
                local_best_size = len(S)
                local_best_S = set(S)
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = set(S)
                stuck_counter = 0
            elif removed_any:
                stuck_counter = 0
            else:
                stuck_counter += 1
            
            if not removed_any:
                # 2. Swap Phase
                swap_done = False
                # Try to find a valid swap
                # We want to move to a different local minimum of the same size
                # to enable further removals.
                
                # To avoid getting stuck in the same swap loop, we can limit swaps per "stuck" cycle
                # or just try random swaps.
                
                for j in random.sample(list(S), min(len(S), 50)): # Try a subset
                    U_j = [i for i in neighbors[j] if cover_count[i] == 1]
                    if not U_j:
                        continue # Should be redundant
                
                    # Find k not in S such that U_j subset neighbors[k]
                    # Heuristic: Pick a random i in U_j, look in neighbors[i]
                    i_target = random.choice(U_j)
                    candidates = [k for k in neighbors[i_target] if k not in S]
                    
                    if not candidates:
                        continue
                        
                    random.shuffle(candidates)
                    
                    for k in candidates:
                        # Check if k covers all of U_j
                        valid = True
                        # Optimization: Check intersection
                        # U_j is small (<=15)
                        neigh_k_set = set(neighbors[k])
                        for i in U_j:
                            if i not in neigh_k_set:
                                valid = False
                                break
                        if valid:
                            # Perform swap
                            S.remove(j)
                            for i in neighbors[j]:
                                cover_count[i] -= 1
                            S.add(k)
                            for i in neighbors[k]:
                                cover_count[i] += 1
                            swap_done = True
                            break
                    if swap_done:
                        break
                
                if swap_done:
                    stuck_counter = 0
                else:
                    # 3. Kick / Perturbation
                    if stuck_counter > 20:
                        # Save state
                        S_old = set(S)
                        cc_old = cover_count.copy()
                        
                        # Remove 2 random, Add 2 random
                        if len(S) > 2:
                            j1, j2 = random.sample(list(S), 2)
                            S.remove(j1)
                            S.remove(j2)
                            for i in neighbors[j1]:
                                cover_count[i] -= 1
                            for i in neighbors[j2]:
                                cover_count[i] -= 1
                            
                            candidates_add = [k for k in range(N) if k not in S]
                            if len(candidates_add) >= 2:
                                k1, k2 = random.sample(candidates_add, 2)
                                S.add(k1)
                                S.add(k2)
                                for i in neighbors[k1]:
                                    cover_count[i] += 1
                                for i in neighbors[k2]:
                                    cover_count[i] += 1
                            
                            # Repair
                            # While invalid, add best word
                            max_repairs = 100
                            repairs = 0
                            while np.any(cover_count == 0) and repairs < max_repairs:
                                uncovered_idx = np.where(cover_count == 0)[0]
                                if len(uncovered_idx) == 0:
                                    break
                                i_unc = random.choice(uncovered_idx)
                                
                                best_k = -1
                                best_gain = -1
                                for k in neighbors[i_unc]:
                                    if k in S:
                                        continue
                                    gain = 0
                                    for i in neighbors[k]:
                                        if cover_count[i] == 0:
                                            gain += 1
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_k = k
                                
                                if best_k == -1:
                                    break
                                
                                S.add(best_k)
                                for i in neighbors[best_k]:
                                    cover_count[i] += 1
                                repairs += 1
                            
                            # Check if we improved or maintained size
                            if len(S) <= len(S_old):
                                stuck_counter = 0
                                # If we improved, the removal phase will catch it next iteration
                                # If we maintained, we are in a new landscape
                            else:
                                # Revert
                                S = S_old
                                cover_count = cc_old
                                stuck_counter += 10

        return local_best_S, local_best_size

    # Run multiple restarts
    # First, a strong greedy start
    S, cc = greedy_init()
    S, size = local_search(S, cc)
    
    # If time remains, try different random greedy starts or kicks on the best solution
    # We can also try to kick the best_S found so far
    
    # Convert best_S to list of lists
    return [all_words[j].tolist() for j in best_S]

def _generic_greedy(q, n, radius):
    # Fallback
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    # Simple greedy
    covered = [False] * N
    S = []
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = []
        for j, v in enumerate(all_words):
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                neigh.append(j)
        neighbors.append(neigh)
        
    while any(not c for c in covered):
        best_idx = -1
        best_gain = -1
        for j in range(N):
            if j in S:
                continue
            gain = sum(1 for i in neighbors[j] if not covered[i])
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        if best_idx == -1:
            break
        S.append(best_idx)
        for i in neighbors[best_idx]:
            covered[i] = True
            
    return [all_words[j] for j in S]
