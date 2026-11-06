def construct():
    import time
    import random

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free_add(A, sums, x):
        if 2*x in A:
            return False
        if x in sums:
            return False
        for a in A:
            if (x + a) in A:
                return False
        return True

    def greedy_construct(order):
        A = set()
        sums = set()
        for x in order:
            if is_sum_free_add(A, sums, x):
                old_A = set(A)
                A.add(x)
                sums.add(2*x)
                for a in old_A:
                    sums.add(x + a)
        return A

    def expand_from(A_init, candidates):
        A = set(A_init)
        sums = {a+b for a in A for b in A}
        for x in candidates:
            if is_sum_free_add(A, sums, x):
                old_A = set(A)
                A.add(x)
                sums.add(2*x)
                for a in old_A:
                    sums.add(x + a)
        return A

    def solve_instance(S):
        n = len(S)
        if n == 0: return []
        
        m = min(S)
        M = max(S)
        threshold = (m + M) / 2.0
        
        A_top = {x for x in S if x > threshold}
        cand_top = [x for x in S if x <= threshold]
        
        best_A = set()
        best_size = 0
        
        # Heuristic 1: Expand Top Half
        A1 = expand_from(A_top, sorted(cand_top, reverse=True))
        if len(A1) > best_size:
            best_size = len(A1)
            best_A = A1
            
        # Heuristic 2: Expand Top Half (sorted by abs)
        A2 = expand_from(A_top, sorted(cand_top, key=abs, reverse=True))
        if len(A2) > best_size:
            best_size = len(A2)
            best_A = A2
            
        # Random Restarts from Scratch
        start_time = time.time()
        limit_time = 20 # 20s per instance is safe
        
        # We can also do perturbation from best_A
        # But simple restarts are robust.
        
        for _ in range(2000):
            if time.time() - start_time > limit_time:
                break
            random.shuffle(S)
            A_rand = greedy_construct(S)
            if len(A_rand) > best_size:
                best_size = len(A_rand)
                best_A = A_rand
                
        return sorted(best_A)

    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    for S in instances:
        res = solve_instance(S)
        results.append(res)
        
    return results
