import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # For n <= 4, recursive construction is optimal
    if n <= 4:
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap_val = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap_val:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 5, use greedy construction + local search
    N = n ** n  # Not factorial, but we'll use permutations
    # Actually N = n!
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute overlap costs between all pairs? Too memory intensive for n=7 (5040^2 = 25M entries)
    # Instead, compute on the fly with caching
    
    overlap_cache = {}
    
    def get_overlap(a, b):
        """Find max k such that a[n-k:] == b[:k]"""
        key = (a, b)
        if key in overlap_cache:
            return overlap_cache[key]
        for k in range(n - 1, 0, -1):
            if a[n-k:] == b[:k]:
                overlap_cache[key] = k
                return k
        overlap_cache[key] = 0
        return 0
    
    def calc_cost(path):
        """Total length of superpermutation from path"""
        total = n
        for i in range(len(path) - 1):
            ov = get_overlap(perms[path[i]], perms[path[i+1]])
            total += n - ov
        return total
    
    # Greedy construction with multiple random starts
    best_path = None
    best_cost = float('inf')
    
    num_starts = min(5, max(1, int(30 * 60 / N)))  # Scale with time budget
    random.seed(42)
    
    for start_idx in range(num_starts):
        unvisited = set(range(N))
        # Start with a random permutation
        current = random.choice(list(unvisited))
        unvisited.remove(current)
        path = [current]
        
        while unvisited:
            if time.time() > deadline - 30:
                break
            min_cost = float('inf')
            next_node = -1
            curr = perms[current]
            for v in unvisited:
                v_perm = perms[v]
                ov = get_overlap(curr, v_perm)
                cost = n - ov
                if cost < min_cost:
                    min_cost = cost
                    next_node = v
            if next_node == -1:
                break
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        
        if len(path) < N:
            continue
        
        cost = calc_cost(path)
        if cost < best_cost:
            best_cost = cost
            best_path = path[:]
        
        if time.time() > deadline - 60:
            break
    
    # Local search: try to improve by swapping adjacent elements
    if best_path is not None:
        improved = True
        while improved and (time.time() - start_time) < deadline - 20:
            improved = False
            for i in range(len(best_path) - 1):
                if time.time() > deadline - 20:
                    break
                # Try swapping best_path[i] and best_path[i+1]
                old_a = best_path[i]
                old_b = best_path[i+1]
                best_path[i] = old_b
                best_path[i+1] = old_a
                new_cost = calc_cost(best_path)
                if new_cost < best_cost:
                    best_cost = new_cost
                    improved = True
                else:
                    # Swap back
                    best_path[i] = old_a
                    best_path[i+1] = old_b
    
    # Convert path to word
    if best_path is not None:
        word = list(perms[best_path[0]])
        for i in range(1, len(best_path)):
            a = perms[best_path[i-1]]
            b = perms[best_path[i]]
            ov = get_overlap(a, b)
            word.extend(b[ov:])
        return "".join(map(str, word))
    
    # Fallback to recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms_list, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_list.append(window)
        new = []
        for p in perms_list:
            block = list(p) + [m] + list(p)
            overlap_val = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap_val:]
        word = new
    return "".join(map(str, word))
