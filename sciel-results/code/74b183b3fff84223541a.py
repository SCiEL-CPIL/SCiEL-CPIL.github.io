import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    base = [1, 2, 5, 11]
    base_set = frozenset(base)
    
    def compute_sums(arr):
        n = len(arr)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in sums:
                    return None
                sums.add(s)
        return sums
    
    def can_add(arr, sums, x):
        new_sums = set()
        for y in arr:
            s = x + y
            if s in sums or s in new_sums:
                return False
            new_sums.add(s)
        s = 2 * x
        if s in sums or s in new_sums:
            return False
        return True
    
    def greedy_expand(arr, sums, rng):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = list(range(1, 285))
            rng.shuffle(candidates)
            for x in candidates:
                if x in arr:
                    continue
                if can_add(arr, sums, x):
                    arr.append(x)
                    arr.sort()
                    for y in arr[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    improved = True
                    break
        return arr, sums
    
    def kick_and_reexpand(arr, sums, rng, kick_size=1):
        if len(arr) <= len(base):
            return arr.copy(), sums.copy()
        
        removable = [x for x in arr if x not in base_set]
        if len(removable) < kick_size:
            kick_size = len(removable)
            if kick_size == 0:
                return arr.copy(), sums.copy()
        
        indices = rng.sample(range(len(removable)), kick_size)
        to_remove = set(removable[i] for i in indices)
        
        new_arr = [x for x in arr if x not in to_remove]
        new_arr.sort()
        
        new_sums = compute_sums(new_arr)
        if new_sums is None:
            return arr.copy(), sums.copy()
        
        new_arr, new_sums = greedy_expand(new_arr, new_sums, rng)
        return new_arr, new_sums
    
    rng = random.Random(12345)
    current = base.copy()
    current.sort()
    sums = compute_sums(current)
    current, sums = greedy_expand(current, sums, rng)
    
    best_arr = current.copy()
    best_size = len(current)
    
    no_improve = 0
    max_no_improve = 80
    
    while time.time() < deadline:
        r = rng.random()
        if r < 0.4:
            kick_size = 1
            new_arr, new_sums = kick_and_reexpand(current.copy(), sums.copy(), rng, kick_size)
        elif r < 0.7:
            kick_size = 2
            new_arr, new_sums = kick_and_reexpand(current.copy(), sums.copy(), rng, kick_size)
        elif r < 0.9:
            kick_size = 3
            new_arr, new_sums = kick_and_reexpand(current.copy(), sums.copy(), rng, kick_size)
        else:
            new_arr = base.copy()
            new_sums = compute_sums(new_arr)
            new_arr, new_sums = greedy_expand(new_arr, new_sums, rng)
        
        if new_sums is None:
            continue
        
        if len(new_arr) > best_size:
            best_arr = new_arr.copy()
            best_size = len(new_arr)
            current = new_arr.copy()
            sums = new_sums.copy()
            no_improve = 0
        elif len(new_arr) >= len(current) and rng.random() < 0.4:
            current = new_arr.copy()
            sums = new_sums.copy()
        no_improve += 1
        
        if no_improve > max_no_improve:
            no_improve = 0
            max_no_improve = min(max_no_improve + 20, 300)
        
        if best_size >= 19:
            break
    
    return best_arr
