import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s safety margin
    
    best_set = None
    best_score = float('inf')
    
    def score(A):
        if not A: return 0
        sums = set()
        prods = set()
        for x in A:
            for y in A:
                sums.add(x + y)
                prods.add(x * y)
        return max(len(sums), len(prods))
        
    def local_search(A, iterations=5000):
        A = list(A)
        current_score = score(A)
        temp = 10.0
        best_local = list(A)
        best_local_score = current_score
        
        for i in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick random element to swap
            idx = random.randint(0, len(A)-1)
            old_val = A[idx]
            
            # Pick random new value
            # Bias towards small numbers
            # If current set is small, try small numbers.
            max_val = max(A)
            # Explore range [1, max(100, max_val*2)]
            upper = max(100, int(max_val * 1.5))
            if upper > 10000: upper = 10000
            
            new_val = random.randint(1, upper)
            
            if new_val == old_val:
                continue
            if new_val in A:
                continue
                
            # Apply swap
            A[idx] = new_val
            new_score = score(A)
            
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local = list(A)
            else:
                # Accept with probability exp(-delta/temp)
                if random.random() < pow(2.71828, -delta / temp):
                    current_score = new_score
                else:
                    A[idx] = old_val
            
            # Cooling
            temp *= 0.995
            
        return best_local, best_local_score

    # 1. Greedy Construction with restricted pools
    def greedy_construct(pool_size):
        A = [1]
        for k in range(1, 20):
            best_cand = None
            best_cost = float('inf')
            # Sample candidates to speed up
            # If pool is large, sample. If small, iterate.
            candidates = list(range(1, pool_size + 1))
            random.shuffle(candidates)
            # Limit to 200 best candidates to save time? 
            # Or just iterate if pool is small.
            # Let's iterate if pool <= 500, else sample 500.
            if len(candidates) > 500:
                candidates = random.sample(candidates, 500)
            
            for c in candidates:
                if c in A:
                    continue
                A_temp = A + [c]
                cost = score(A_temp)
                if cost < best_cost:
                    best_cost = cost
                    best_cand = c
                if time.time() > deadline:
                    return A
            A.append(best_cand)
        return A

    # Initial Candidates
    initial_sets = []
    
    # 1. Range 1-20
    initial_sets.append(list(range(1, 21)))
    
    # 2. Greedy with small pool
    initial_sets.append(greedy_construct(100))
    
    # 3. Greedy with medium pool
    initial_sets.append(greedy_construct(500))
    
    # 4. Powers of 2 padded
    p2 = [2**i for i in range(14)] # 1..8192
    # Add 6 more small numbers
    extra = [x for x in range(1, 100) if x not in p2][:6]
    initial_sets.append(p2 + extra)
    
    # 5. Random small
    initial_sets.append(random.sample(range(1, 100), 20))
    
    # 6. Random medium
    initial_sets.append(random.sample(range(1, 500), 20))

    # Refine each initial set
    for i, A_init in enumerate(initial_sets):
        if time.time() > deadline:
            break
        
        A_refined, s_refined = local_search(A_init, iterations=2000)
        if s_refined < best_score:
            best_score = s_refined
            best_set = A_refined
            
    # Continue with random restarts if time remains
    while time.time() < deadline:
        A_rand = random.sample(range(1, 1000), 20)
        A_refined, s_refined = local_search(A_rand, iterations=1000)
        if s_refined < best_score:
            best_score = s_refined
            best_set = A_refined

    if best_set is None:
        return list(range(1, 21))
        
    return sorted(best_set)
