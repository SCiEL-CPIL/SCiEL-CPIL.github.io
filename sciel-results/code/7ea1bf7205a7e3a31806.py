"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0
    start = time.time()
    
    def get_sums_set(S):
        sums = set()
        n = len(S)
        for i in range(n):
            si = S[i]
            sums.add(si + si)
            for j in range(i + 1, n):
                sums.add(si + S[j])
        return sums
    
    def can_add(x, S, sums):
        if 2 * x in sums:
            return False
        for a in S:
            if x + a in sums:
                return False
        return True
    
    def is_sidon(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(S):
        if not S:
            return False
        sums = get_sums_set(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add(x, S, sums):
                return False
        return True
    
    def greedy_construct(order):
        S = []
        sums = set()
        for x in order:
            ok = True
            if 2 * x in sums:
                ok = False
            else:
                for a in S:
                    if x + a in sums:
                        ok = False
                        break
            if ok:
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S
    
    def prune_for_maximality(S):
        """Remove elements while maintaining maximality."""
        S = list(S)
        changed = True
        while changed:
            changed = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i + 1:]
                if len(new_S) == 0:
                    continue
                if is_maximal(new_S):
                    S = new_S
                    changed = True
                    break
        return S
    
    def vectorized_can_add(x, S_arr, sums_arr):
        """Check if x can be added using NumPy vectorization."""
        # Check 2*x
        if np.any(sums_arr == 2 * x):
            return False
        # Check x + a for all a in S
        if len(S_arr) == 0:
            return True
        new_sums = x + S_arr
        # Check if any of these are already in sums
        for ns in new_sums:
            if np.any(sums_arr == ns):
                return False
        return True
    
    def vectorized_greedy(order, S_arr, sums_arr):
        """Vectorized greedy construction."""
        for x in order:
            # Check if x can be added
            if np.any(sums_arr == 2 * x):
                continue
            if len(S_arr) > 0:
                new_sums = x + S_arr
                conflict = False
                for ns in new_sums:
                    if np.any(sums_arr == ns):
                        conflict = True
                        break
                if conflict:
                    continue
            # Add x
            S_arr = np.append(S_arr, [x])
            new_sums_to_add = [2 * x]
            if len(S_arr) > 1:
                new_sums_to_add.extend(x + S_arr[:-1])
            sums_arr = np.union1d(sums_arr, np.array(new_sums_to_add))
        return S_arr, sums_arr
    
    # Phase 1: Build initial Sidon set using ascending greedy
    S = greedy_construct(range(1, N + 1))
    S = prune_for_maximality(S)
    best_S = list(S)
    best_size = len(S)
    
    # Phase 2: Try different greedy orderings with vectorized approach
    rng = random.Random(12345)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Various ordering strategies
        if trial % 7 == 0:
            # Random
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif trial % 7 == 1:
            # Descending
            order = list(range(N, 0, -1))
        elif trial % 7 == 2:
            # Start from middle
            order = list(range(N // 2, N + 1)) + list(range(1, N // 2))
        elif trial % 7 == 3:
            # Alternating pattern
            order = list(range(1, N + 1))
            for i in range(0, len(order), 2):
                if i + 1 < len(order):
                    order[i], order[i + 1] = order[i + 1], order[i]
        elif trial % 7 == 4:
            # Random with bias toward larger numbers
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif trial % 7 == 5:
            # Start from ends
            order = []
            left, right = 1, N
            while left <= right:
                order.append(left)
                if left != right:
                    order.append(right)
                left += 1
                right -= 1
        else:
            # Random with seed variation
            order = list(range(1, N + 1))
            rng2 = random.Random(trial * 7919)
            rng2.shuffle(order)
        
        # Use vectorized greedy
        S_arr = np.array([], dtype=int)
        sums_arr = np.array([], dtype=int)
        S_arr, sums_arr = vectorized_greedy(order, S_arr, sums_arr)
        S_list = S_arr.tolist()
        
        if S_list:
            S_pruned = prune_for_maximality(S_list)
            if len(S_pruned) < best_size:
                best_size = len(S_pruned)
                best_S = list(S_pruned)
    
    # Phase 3: Local search - try swapping elements
    S = list(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i + 1:]
            if not S_without:
                continue
            
            sums_without = get_sums_set(S_without)
            S_without_set = set(S_without)
            
            # Try to find a replacement that keeps it maximal
            for x in range(1, N + 1):
                if x in S_without_set:
                    continue
                if not can_add(x, S_without, sums_without):
                    continue
                new_S = sorted(S_without + [x])
                if is_maximal(new_S):
                    S = new_S
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
    
    # Phase 4: More aggressive removal with different strategies
    S = list(best_S)
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Try removing multiple elements at once
        if len(S) > 10:
            removed = set()
            new_S = list(S)
            # Try removing elements from the end
            for i in range(len(new_S) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = new_S[:i] + new_S[i + 1:]
                if is_maximal(candidate):
                    new_S = candidate
                    removed.add(S[i])
            
            if len(new_S) < len(best_S):
                best_S = new_S
                best_size = len(new_S)
    
    # Phase 5: Try different initial constructions and combine
    for trial in range(50):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_arr = np.array([], dtype=int)
        sums_arr = np.array([], dtype=int)
        S_arr, sums_arr = vectorized_greedy(order, S_arr, sums_arr)
        S_list = S_arr.tolist()
        
        if S_list:
            S_pruned = prune_for_maximality(S_list)
            if len(S_pruned) < best_size:
                best_size = len(S_pruned)
                best_S = list(S_pruned)
    
    # Final verification and repair
    S = list(best_S)
    if not S or not is_sidon(S):
        # Fall back to ascending greedy
        S = greedy_construct(range(1, N + 1))
        S = prune_for_maximality(S)
    
    if not is_maximal(S):
        # Repair: add elements that can be added
        changed = True
        while changed:
            changed = False
            sums = get_sums_set(S)
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add(x, S, sums):
                    S.append(x)
                    S_set.add(x)
                    sums.add(2 * x)
                    for a in S[:-1]:
                        sums.add(x + a)
                    changed = True
                    break
        S = sorted(S)
    
    return S

# EVOLVE-BLOCK-END
