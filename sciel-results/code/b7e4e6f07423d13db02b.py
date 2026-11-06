import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    N = 1000
    best_set = []
    best_size = 0

    def try_greedy_fast(order):
        """Fast greedy: add element if it doesn't violate B2[2] constraint."""
        S = []
        S_set = set()
        counts = np.zeros(2 * N + 1, dtype=np.int8)
        is_full = np.zeros(2 * N + 1, dtype=bool)
        
        for x in order:
            if x in S_set:
                continue
            # Check if 2x is full
            if is_full[2 * x]:
                continue
            # Check all sums x+y for y in S
            if S:
                S_arr = np.array(S)
                new_sums = x + S_arr
                if np.any(is_full[new_sums]):
                    continue
            # Add x
            S.append(x)
            S_set.add(x)
            counts[2 * x] += 1
            if counts[2 * x] == 2:
                is_full[2 * x] = True
            if S:
                S_arr = np.array(S[:-1])
                new_sums = x + S_arr
                counts[new_sums] += 1
                is_full[new_sums] = (counts[new_sums] >= 2)
        
        return S

    def try_greedy_cost(order):
        """Cost-based greedy: prefer elements that create fewer 'tight' sums (count==1)."""
        S = []
        S_set = set()
        counts = np.zeros(2 * N + 1, dtype=np.int8)
        is_full = np.zeros(2 * N + 1, dtype=bool)
        candidates = list(order)
        
        while candidates:
            best_x = None
            best_cost = float('inf')
            best_candidates = []
            
            for x in candidates:
                if x in S_set:
                    continue
                if is_full[2 * x]:
                    continue
                
                cost = 0
                if counts[2 * x] == 1:
                    cost += 1
                
                valid = True
                if S:
                    S_arr = np.array(S)
                    new_sums = x + S_arr
                    if np.any(is_full[new_sums]):
                        valid = False
                    else:
                        cost += int(np.sum(counts[new_sums] == 1))
                
                if not valid:
                    continue
                
                if cost < best_cost:
                    best_cost = cost
                    best_candidates = [x]
                elif cost == best_cost:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
            
            best_x = random.choice(best_candidates)
            S.append(best_x)
            S_set.add(best_x)
            counts[2 * best_x] += 1
            if counts[2 * best_x] == 2:
                is_full[2 * best_x] = True
            if len(S) > 1:
                S_arr = np.array(S[:-1])
                new_sums = best_x + S_arr
                counts[new_sums] += 1
                is_full[new_sums] = (counts[new_sums] >= 2)
            
            candidates.remove(best_x)
        
        return S

    def local_search(S, max_iters=200):
        """Local search: try to swap elements to improve set size."""
        S = list(S)
        S_set = set(S)
        counts = np.zeros(2 * N + 1, dtype=np.int8)
        is_full = np.zeros(2 * N + 1, dtype=bool)
        
        # Build counts
        for i, a in enumerate(S):
            counts[2 * a] += 1
            if counts[2 * a] == 2:
                is_full[2 * a] = True
            if i > 0:
                S_arr = np.array(S[:i])
                new_sums = a + S_arr
                counts[new_sums] += 1
                is_full[new_sums] = (counts[new_sums] >= 2)
        
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if not S:
                break
            
            # Pick random element to remove
            idx = random.randint(0, len(S) - 1)
            old_x = S[idx]
            
            # Remove old_x
            S_arr = np.array([y for y in S if y != old_x])
            if len(S_arr) > 0:
                new_sums = old_x + S_arr
                counts[new_sums] -= 1
                is_full[new_sums] = (counts[new_sums] >= 2)
            counts[2 * old_x] -= 1
            is_full[2 * old_x] = (counts[2 * old_x] >= 2)
            
            S.pop(idx)
            S_set.discard(old_x)
            
            # Try to find a replacement
            found = False
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            random.shuffle(candidates)
            
            for x in candidates[:200]:
                if is_full[2 * x]:
                    continue
                
                valid = True
                if S:
                    S_arr = np.array(S)
                    new_sums = x + S_arr
                    if np.any(is_full[new_sums]):
                        valid = False
                    else:
                        # Compute cost
                        cost = int(np.sum(counts[new_sums] == 1))
                        if counts[2 * x] == 1:
                            cost += 1
                        # Only accept if cost is low
                        if cost > 5:
                            continue
                
                if valid:
                    # Add x
                    S.append(x)
                    S_set.add(x)
                    counts[2 * x] += 1
                    if counts[2 * x] == 2:
                        is_full[2 * x] = True
                    if len(S) > 1:
                        S_arr = np.array(S[:-1])
                        new_sums = x + S_arr
                        counts[new_sums] += 1
                        is_full[new_sums] = (counts[new_sums] >= 2)
                    found = True
                    improved = True
                    break
            
            if not found:
                # Put old_x back
                S.append(old_x)
                S_set.add(old_x)
                if len(S) > 1:
                    S_arr = np.array(S[:-1])
                    new_sums = old_x + S_arr
                    counts[new_sums] += 1
                    is_full[new_sums] = (counts[new_sums] >= 2)
                counts[2 * old_x] += 1
                if counts[2 * old_x] == 2:
                    is_full[2 * old_x] = True
        
        return S

    # Phase 1: Fast greedy with different orderings
    S = try_greedy_fast(range(1, N + 1))
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    S = try_greedy_fast(range(N, 0, -1))
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    # Phase 2: Cost-based greedy with random restarts
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        random.seed(seed * 12345 + 67890)
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = try_greedy_cost(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Phase 3: More fast greedy restarts
    while time.time() < deadline - 30:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = try_greedy_fast(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Phase 4: Local search on the best solution
    if time.time() < deadline - 5:
        improved_S = local_search(best_set, max_iters=500)
        if len(improved_S) > best_size:
            best_set = improved_S
            best_size = len(improved_S)
    
    # Phase 5: Try local search on a few other good solutions
    if time.time() < deadline - 5:
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = try_greedy_cost(order)
            if len(S) >= best_size - 3:
                improved_S = local_search(S, max_iters=200)
                if len(improved_S) > best_size:
                    best_set = improved_S
                    best_size = len(improved_S)
    
    return best_set
