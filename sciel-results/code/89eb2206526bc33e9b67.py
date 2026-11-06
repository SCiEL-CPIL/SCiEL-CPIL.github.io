import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s wall clock
    
    n = 4096
    best_set = []
    
    def get_new_sums(x, S):
        """Compute all triple sums involving x and elements of S (unordered, with repetition)."""
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
        for i in range(len(S)):
            y = S[i]
            for j in range(i, len(S)):
                z = S[j]
                new_sums.add(x + y + z)
        return new_sums
    
    def is_valid_addition(x, S, existing_sums):
        """Check if x can be added to S without creating duplicate triple sums."""
        new_sums = get_new_sums(x, S)
        expected = 1 + len(S) + len(S) * (len(S) + 1) // 2
        if len(new_sums) != expected:
            return False
        if new_sums & existing_sums:
            return False
        return True
    
    def compute_all_sums(S):
        """Compute all triple sums for a set S."""
        sums = set()
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            a = S_sorted[i]
            sums.add(3 * a)
            for j in range(i, len(S_sorted)):
                b = S_sorted[j]
                sums.add(2 * a + b)
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    sums.add(a + b + c)
        return sums
    
    def greedy_build(ordering, deadline, S_init=None, sums_init=None):
        """Build a set using greedy algorithm with given ordering."""
        S = S_init[:] if S_init else []
        existing_sums = sums_init.copy() if sums_init else set()
        
        for x in ordering:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S, existing_sums):
                S.append(x)
                existing_sums.update(get_new_sums(x, S[:-1]))
        
        return S, existing_sums
    
    # Run 1: Power-of-3 based construction (guaranteed valid)
    base_power3 = []
    val = 1
    while val <= n:
        base_power3.append(val)
        val *= 3
    
    if len(base_power3) > len(best_set):
        best_set = base_power3.copy()
    
    # Run 2: Smallest-first greedy (baseline)
    S1 = []
    sums1 = set()
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        new_sums = get_new_sums(x, S1)
        expected = 1 + len(S1) + len(S1) * (len(S1) + 1) // 2
        if len(new_sums) == expected and not (new_sums & sums1):
            S1.append(x)
            sums1.update(new_sums)
    
    if len(S1) > len(best_set):
        best_set = S1.copy()
    
    # Run 3: Multiple randomized greedy attempts with different strategies
    strategies = [
        "random",
        "small_bias",
        "large_bias",
        "chunked",
        "spread",
        "power3_mix"
    ]
    
    for seed in range(100):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        strategy = strategies[seed % len(strategies)]
        candidates = list(range(1, n + 1))
        
        if strategy == "random":
            random.shuffle(candidates)
        elif strategy == "small_bias":
            candidates.sort(key=lambda x: x + random.randint(-50, 50))
        elif strategy == "large_bias":
            candidates.sort(key=lambda x: -x + random.randint(-50, 50))
        elif strategy == "chunked":
            chunk_size = 100
            shuffled = []
            for i in range(0, n, chunk_size):
                chunk = list(range(i + 1, min(i + chunk_size + 1, n + 1)))
                random.shuffle(chunk)
                shuffled.extend(chunk)
            candidates = shuffled
        elif strategy == "spread":
            # Prefer values spread across the range
            weights = [1.0 / (1 + abs(x - n/2)) for x in range(1, n + 1)]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            selected = []
            remaining = list(range(1, n + 1))
            for _ in range(len(remaining)):
                if not remaining:
                    break
                r = random.random()
                cumulative = 0
                for idx, val in enumerate(remaining):
                    cumulative += weights[val - 1]
                    if cumulative >= r:
                        selected.append(val)
                        remaining.remove(val)
                        break
            candidates = selected
        else:  # power3_mix
            # Start with power-of-3 values, then add random others
            power3_set = set(base_power3)
            others = [x for x in range(1, n + 1) if x not in power3_set]
            random.shuffle(others)
            candidates = base_power3 + others
        
        S, sums = greedy_build(candidates, deadline)
        
        if len(S) > len(best_set):
            best_set = S.copy()
    
    # Run 4: Local search - try to add more elements to best set
    if time.time() < deadline - 5.0:
        current = best_set.copy()
        current_sums = compute_all_sums(current)
        used = set(current)
        
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in used:
                continue
            if is_valid_addition(x, current, current_sums):
                current.append(x)
                current_sums.update(get_new_sums(x, current[:-1]))
                used.add(x)
        
        if len(current) > len(best_set):
            best_set = current.copy()
    
    # Run 5: Try to improve by removing one element and adding two
    if time.time() < deadline - 15.0:
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            current = best_set.copy()
            current_sums = compute_all_sums(current)
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                
                removed = current[i]
                temp_set = current[:i] + current[i+1:]
                temp_sums = compute_all_sums(temp_set)
                
                # Try to add new elements
                new_elements = []
                for x in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if x in set(temp_set) or x == removed:
                        continue
                    if is_valid_addition(x, temp_set, temp_sums):
                        new_elements.append(x)
                        temp_set.append(x)
                        temp_sums.update(get_new_sums(x, temp_set[:-1]))
                        if len(new_elements) >= 2:
                            break
                
                if len(new_elements) >= 2:
                    current = temp_set
                    improved = True
                    break
            
            if len(current) > len(best_set):
                best_set = current.copy()
                if improved:
                    continue
            else:
                break
    
    return best_set
