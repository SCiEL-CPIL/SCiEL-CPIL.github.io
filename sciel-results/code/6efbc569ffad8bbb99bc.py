import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 4096
    best_set = []
    
    def get_new_sums(x, S):
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

    # Run 1: Smallest First greedy
    S1 = []
    sums1 = set()
    for x in range(1, n + 1):
        new_sums = get_new_sums(x, S1)
        expected = 1 + len(S1) + len(S1)*(len(S1)+1)//2
        if len(new_sums) == expected and not (new_sums & sums1):
            S1.append(x)
            sums1.update(new_sums)
            
    if len(S1) > len(best_set):
        best_set = S1.copy()
    
    # Run 2: Multiple randomized greedy attempts with different orderings
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Create candidates with some randomization in ordering
        candidates = list(range(1, n + 1))
        
        # Try different strategies: random shuffle, or biased toward smaller/larger
        strategy = seed % 4
        if strategy == 0:
            random.shuffle(candidates)
        elif strategy == 1:
            # Bias toward smaller values
            candidates.sort(key=lambda x: x + random.randint(-50, 50))
        elif strategy == 2:
            # Bias toward larger values  
            candidates.sort(key=lambda x: -x + random.randint(-50, 50))
        else:
            # Mix of small and large
            small = list(range(1, n//2 + 1))
            large = list(range(n//2 + 1, n + 1))
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
            random.shuffle(candidates)
        
        S = []
        existing_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            new_sums = get_new_sums(x, S)
            expected = 1 + len(S) + len(S)*(len(S)+1)//2
            if len(new_sums) == expected and not (new_sums & existing_sums):
                S.append(x)
                existing_sums.update(new_sums)
                
        if len(S) > len(best_set):
            best_set = S.copy()
    
    # Run 3: Local search - try to replace elements to fit more
    if time.time() < deadline - 5.0:
        # Start from best_set and try local improvements
        current = best_set.copy()
        if current:
            current_sums = set()
            # Recompute all sums for current set
            S_sorted = sorted(current)
            for i in range(len(S_sorted)):
                a = S_sorted[i]
                current_sums.add(3*a)
                for j in range(i, len(S_sorted)):
                    b = S_sorted[j]
                    current_sums.add(2*a + b)
                    for k in range(j, len(S_sorted)):
                        c = S_sorted[k]
                        current_sums.add(a + b + c)
            
            # Try to add elements that were skipped
            used = set(current)
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in used:
                    continue
                new_sums = get_new_sums(x, current)
                expected = 1 + len(current) + len(current)*(len(current)+1)//2
                if len(new_sums) == expected and not (new_sums & current_sums):
                    current.append(x)
                    current_sums.update(new_sums)
                    used.add(x)
            
            if len(current) > len(best_set):
                best_set = current.copy()
    
    return best_set
