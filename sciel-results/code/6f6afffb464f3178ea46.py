import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 4096
    best_set = []
    best_size = 0
    
    def compute_all_sums(S):
        k = len(S)
        sums = set()
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                for l in range(j, k):
                    sums.add(si + sj + S[l])
        return sums

    def greedy_build(order):
        S = []
        current_sums = set()
        for x in order:
            if time.time() > deadline:
                break
            new_sums = set()
            new_sums.add(3 * x)
            for y in S:
                new_sums.add(2 * x + y)
            k = len(S)
            for i in range(k):
                for j in range(i, k):
                    new_sums.add(x + S[i] + S[j])
            
            expected_new = 1 + k + k * (k + 1) // 2
            if len(new_sums) != expected_new:
                continue
                
            if new_sums & current_sums:
                continue
                
            S.append(x)
            current_sums.update(new_sums)
            
        return S

    # 1. Greedy Initialization
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    for _ in range(20):
        random.seed()
        o = list(range(1, n + 1))
        random.shuffle(o)
        orders.append(o)
        
    for o in orders:
        if time.time() > deadline:
            break
        S = greedy_build(o)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # 2. Local Search
    def local_search(start_S):
        nonlocal best_set, best_size
        S = list(start_S)
        sums = compute_all_sums(S)
        size = len(S)
        available = [x for x in range(1, n + 1) if x not in S]
        
        no_improve_count = 0
        
        while time.time() < deadline:
            improved = False
            
            # Try Add
            random.shuffle(available)
            for x in available:
                if time.time() > deadline:
                    return
                if x in S:
                    continue
                    
                new_sums = set()
                new_sums.add(3 * x)
                for y in S:
                    new_sums.add(2 * x + y)
                k = len(S)
                for i in range(k):
                    for j in range(i, k):
                        new_sums.add(x + S[i] + S[j])
                        
                expected_new = 1 + k + k * (k + 1) // 2
                if len(new_sums) != expected_new:
                    continue
                    
                if new_sums & sums:
                    continue
                    
                # Accept
                S.append(x)
                sums.update(new_sums)
                available.remove(x)
                size += 1
                improved = True
                no_improve_count = 0
                break
                
            if improved:
                if size > best_size:
                    best_size = size
                    best_set = S.copy()
                if size >= 40:
                    return
                    
            # Try Swap
            if not improved:
                if not available:
                    break
                u = random.choice(S)
                v = random.choice(available)
                
                new_S = [x for x in S if x != u]
                new_S.append(v)
                
                new_sums = compute_all_sums(new_S)
                expected = size * (size + 1) * (size + 2) // 6
                if len(new_sums) == expected:
                    S = new_S
                    sums = new_sums
                    available.remove(v)
                    available.append(u)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If stuck for too long, restart?
                    # For now, just continue
                
        return

    if best_size < 40:
        local_search(best_set)
        
    return best_set
