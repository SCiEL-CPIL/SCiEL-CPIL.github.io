N = 256

import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Leave 5s buffer for safety

    def greedy_randomized():
        # Shuffle candidates
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            # Check if x can be added
            added_sums = [x + a for a in selected] + [2 * x]
            # Optimization: check intersection first
            if sums.intersection(added_sums):
                continue
            if len(set(added_sums)) != len(added_sums):
                continue
            
            selected.append(x)
            sums.update(added_sums)
            
        return selected

    def greedy_weighted(weights):
        # Greedy with weighted random selection
        candidates = list(range(1, N + 1))
        
        selected = []
        sums = set()
        
        while candidates:
            # Weighted random choice
            total_w = sum(weights[c] for c in candidates)
            r = random.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c in candidates:
                cum += weights[c]
                if r <= cum:
                    chosen = c
                    break
            
            candidates.remove(chosen)
            x = chosen
            
            # Check if x can be added
            added_sums = [x + a for a in selected] + [2 * x]
            if sums.intersection(added_sums):
                continue
            if len(set(added_sums)) != len(added_sums):
                continue
            
            selected.append(x)
            sums.update(added_sums)
            
        return selected

    def is_sidon(s):
        if not s:
            return True
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def is_maximal(s):
        if not is_sidon(s):
            return False
        s_set = set(s)
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if adding x preserves Sidon property
            new_sums = [x + a for a in s] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if sums.intersection(new_sums):
                continue
            # x can be added, so not maximal
            return False
        return True

    def try_remove(s, idx):
        """Try removing element at index idx and check if result is still maximal"""
        new_s = s[:idx] + s[idx+1:]
        if not new_s:
            return None
        if not is_maximal(new_s):
            return None
        return new_s

    def local_search_reduce(s, max_iters=200):
        """Try to reduce size by removing elements while maintaining maximality"""
        current = list(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                new_s = try_remove(current, i)
                if new_s is not None:
                    current = new_s
                    improved = True
                    break
        return current

    def swap_improve(s):
        """Try swapping elements to potentially enable further reductions"""
        current = list(s)
        s_set = set(current)
        
        # Try each element against each missing element
        for i in range(len(current)):
            if time.time() > deadline:
                break
            old = current[i]
            for new in range(1, N + 1):
                if new in s_set:
                    continue
                # Check if swap preserves Sidon
                test = current[:i] + [new] + current[i+1:]
                if is_sidon(test):
                    # Check if we can then remove something
                    reduced = local_search_reduce(test, max_iters=50)
                    if len(reduced) < len(current):
                        return reduced
        return current

    def greedy_with_priority(priority_func):
        """Greedy that picks candidates based on a priority function"""
        candidates = list(range(1, N + 1))
        
        selected = []
        sums = set()
        
        for _ in range(N):
            if not candidates:
                break
            if time.time() > deadline:
                break
            
            # Score each candidate
            best_candidate = None
            best_score = float('inf')
            
            for x in candidates:
                added_sums = [x + a for a in selected] + [2 * x]
                if sums.intersection(added_sums):
                    continue
                if len(set(added_sums)) != len(added_sums):
                    continue
                
                score = priority_func(x, selected, added_sums)
                if score < best_score:
                    best_score = score
                    best_candidate = x
            
            if best_candidate is None:
                break
            
            candidates.remove(best_candidate)
            selected.append(best_candidate)
            sums.update([best_candidate + a for a in selected[:-1]] + [2 * best_candidate])
            
        return selected

    def priority_spread(x, selected, added_sums):
        """Prefer elements that are far from existing ones"""
        if not selected:
            return x
        # Prefer elements that create large sums (spread out)
        return -max(added_sums)

    def priority_central(x, selected, added_sums):
        """Prefer central elements"""
        return abs(x - N/2)

    best_set = None
    best_size = float('inf')
    
    # Strategy 1: Randomized greedy restarts
    while time.time() < deadline:
        current_set = greedy_randomized()
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Try local search reduction
        reduced = local_search_reduce(current_set, max_iters=100)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
    
    # Strategy 2: Weighted greedy with different weightings
    if time.time() < deadline:
        # Uniform weights
        weights = {i: 1.0 for i in range(1, N+1)}
        current_set = greedy_weighted(weights)
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
        reduced = local_search_reduce(current_set, max_iters=100)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
    
    # Strategy 3: Priority-based greedy
    if time.time() < deadline:
        current_set = greedy_with_priority(priority_spread)
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
        reduced = local_search_reduce(current_set, max_iters=100)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
    
    if time.time() < deadline:
        current_set = greedy_with_priority(priority_central)
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
        reduced = local_search_reduce(current_set, max_iters=100)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
    
    # Fallback to deterministic if time ran out or no result (unlikely)
    if best_set is None:
        # Deterministic ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected

    # Final local search polish
    if time.time() < deadline - 2:
        final = local_search_reduce(best_set, max_iters=500)
        if len(final) < len(best_set):
            best_set = final
    
    return best_set
# EVOLVE-BLOCK-END
