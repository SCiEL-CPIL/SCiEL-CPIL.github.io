import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGET_SIZE = TARGET_MAX - TARGET_MIN + 1

    # Precompute target indices
    target_indices = np.arange(TARGET_MIN, TARGET_MAX + 1)

    # Helper to calculate cost
    # Priority: 1. No Gaps, 2. Min Peak, 3. Min Sum of Squares (flatness)
    def get_cost(sums):
        target_sums = sums[TARGET_MIN:TARGET_MAX+1]
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            # Heavy penalty for gaps
            return 1000000 + gaps * 10000 + np.max(target_sums)
        
        peak = np.max(target_sums)
        sq = np.sum(target_sums**2)
        # We want to minimize peak first. 
        # If peak is 1, sq is 129 (all 1s).
        # If peak is 2, sq is at least 129 + (number of 2s).
        return peak * 10000 + sq

    # Helper to update sums for adding/removing x
    # A is a set of ints
    # sums is a numpy array of size 257
    def apply_add(A, sums, x):
        # Add pairs {a, x} for all a in A, a != x
        for a in A:
            s = a + x
            if s <= 256:
                sums[s] += 1
        # Add pair {x, x}
        if 2*x <= 256:
            sums[2*x] += 1
        A.add(x)
        
    def apply_remove(A, sums, x):
        # Remove pairs {a, x} for all a in A, a != x
        for a in A:
            if a == x:
                continue
            s = a + x
            if s <= 256:
                sums[s] -= 1
        # Remove pair {x, x}
        if 2*x <= 256:
            sums[2*x] -= 1
        A.remove(x)

    # Greedy Initialization: Build a set that covers the targets
    def greedy_init():
        A = set()
        sums = np.zeros(257, dtype=np.int32)
        
        # Force add 128 (required for 256) and 0 (helps cover 128)
        apply_add(A, sums, 128)
        apply_add(A, sums, 0)
        
        candidates = [x for x in range(N+1) if x not in A]
        
        # Greedily add elements that minimize cost
        for _ in range(20):
            best_cand = None
            best_cost = float('inf')
            
            for x in candidates:
                apply_add(A, sums, x)
                c = get_cost(sums)
                apply_remove(A, sums, x)
                
                if c < best_cost:
                    best_cost = c
                    best_cand = x
            
            if best_cand is None:
                break
                
            apply_add(A, sums, best_cand)
            candidates.remove(best_cand)
            
            # Stop early if we have no gaps and low peak
            if best_cost < 20000:
                break

        return A, sums

    # Polish Phase: Try to improve the solution via local moves
    def polish(A, sums, max_iters=2000, time_limit=8.0):
        start_t = time.time()
        current_cost = get_cost(sums)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() - start_t < time_limit and max_iters > 0:
            improved = False
            max_iters -= 1
            no_improve_count = 0
            
            A_list = list(A)
            random.shuffle(A_list)
            
            # Try removing elements
            for x in A_list:
                if len(A) <= 2: continue
                if x == 128: continue  # 128 is essential for 256
                
                apply_remove(A, sums, x)
                new_cost = get_cost(sums)
                
                if new_cost <= current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    apply_add(A, sums, x)
            
            # Try swapping elements
            if time.time() - start_t >= time_limit: break
            
            A_list = list(A)
            random.shuffle(A_list)
            
            for x in A_list:
                if x not in A or x == 128:
                    continue
                    
                candidates = [y for y in range(N+1) if y not in A and y != 128]
                if not candidates: continue
                
                random.shuffle(candidates)
                for y in candidates[:15]:
                    apply_remove(A, sums, x)
                    apply_add(A, sums, y)
                    
                    new_cost = get_cost(sums)
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        x = y
                        break
                    else:
                        apply_remove(A, sums, y)
                        apply_add(A, sums, x)
                    
                    if time.time() - start_t >= time_limit: break
                    
                if time.time() - start_t >= time_limit: break
                
            # Try adding elements
            if time.time() - start_t >= time_limit: break
            
            candidates = [y for y in range(N+1) if y not in A and y != 128]
            if candidates:
                random.shuffle(candidates)
                for y in candidates[:10]:
                    apply_add(A, sums, y)
                    new_cost = get_cost(sums)
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                    else:
                        apply_remove(A, sums, y)
                    
                    if time.time() - start_t >= time_limit: break
        
        return A, sums, current_cost

    # Initialize best solution
    best_A = None
    best_cost = float('inf')
    best_sums = None
    
    # 1. Greedy Initialization
    A_greedy, sums_greedy = greedy_init()
    cost_greedy = get_cost(sums_greedy)
    
    if cost_greedy < best_cost:
        best_cost = cost_greedy
        best_A = set(A_greedy)
        best_sums = sums_greedy.copy()
        
    # Polish the greedy result
    A_polished, sums_polished, cost_polished = polish(set(A_greedy), sums_greedy.copy())
    if cost_polished < best_cost:
        best_cost = cost_polished
        best_A = set(A_polished)
        best_sums = sums_polished.copy()

    # 2. Simulated Annealing from multiple starts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 10: break
        
        # Random initialization with 128 forced
        size = random.randint(12, 25)
        A = {128}
        remaining = [x for x in range(N+1) if x != 128]
        if size > 1:
            A.update(random.sample(remaining, size - 1))
            
        sums = np.zeros(257, dtype=np.int32)
        A_list = sorted(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= 256:
                    sums[s] += 1
        
        current_cost = get_cost(sums)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(A)
            best_sums = sums.copy()
            
        # Simulated Annealing Parameters
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        steps = 8000
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Choose move
            move_type = random.random()
            
            if move_type < 0.1:
                # Add
                candidates = [x for x in range(N+1) if x not in A]
                if not candidates: continue
                x = random.choice(candidates)
                
                apply_add(A, sums, x)
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(A, sums, x)
                    
            elif move_type < 0.2:
                # Remove
                if len(A) < 8: continue
                others = [a for a in A if a != 128]
                if not others: continue
                x = random.choice(others)
                
                apply_remove(A, sums, x)
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_add(A, sums, x)
                    
            else:
                # Swap
                others = [a for a in A if a != 128]
                if not others: continue
                x = random.choice(others)
                
                candidates = [y for y in range(N+1) if y not in A and y != 128]
                if not candidates: continue
                y = random.choice(candidates)
                
                apply_remove(A, sums, x)
                apply_add(A, sums, y)
                
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(A, sums, y)
                    apply_add(A, sums, x)
            
            T *= cooling
            if T < T_min:
                break
        
        # Polish after SA
        if time.time() < deadline - 5:
            A_pol, sums_pol, cost_pol = polish(set(A), sums.copy(), max_iters=300, time_limit=2.5)
            if cost_pol < best_cost:
                best_cost = cost_pol
                best_A = set(A_pol)
                best_sums = sums_pol.copy()

    if best_A is None:
        # Fallback
        return [128, 0, 64, 32, 16, 8, 4, 2, 1]
        
    return sorted(best_A)
