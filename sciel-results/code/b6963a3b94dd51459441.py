import time
import random
import math

def compute_metrics(A):
    """
    Compute sizes of A+A and A*A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def objective(A):
    s, p = compute_metrics(A)
    return max(s, p)

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. We keep a generous margin to ensure we return.
    # 50s was used before. Let's use 80s to maximize search time.
    deadline = start_time + 80.0
    
    random.seed(12345)
    
    best_set = None
    best_obj = float('inf')
    
    def try_candidate(A):
        nonlocal best_set, best_obj
        # validate: 20 distinct positive ints <= 10000
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        
        A = list(A)
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]
    
    # --- Phase 1: Structured Initializations ---
    
    # 1. Arithmetic Progression (1..20) - strong for sumset
    try_candidate(list(range(1, 21)))
    
    # 2. Geometric Progressions
    for base in [1.5, 1.8, 2.0, 2.5, 3.0]:
        A = []
        v = 1.0
        while len(A) < 20:
            iv = int(round(v))
            if iv > 10000:
                break
            if iv not in A and iv >= 1:
                A.append(iv)
            v = v * base
        # Fill remaining if GP truncated
        if len(A) < 20:
            # Fill with small integers not in A to keep products small?
            # Or just random. Let's try small integers first.
            for k in range(1, 10001):
                if k not in A:
                    A.append(k)
                if len(A) == 20:
                    break
        try_candidate(A[:20])
        
    # 3. Mixed structures: AP then GP?
    # e.g., 1..10, then 20, 40, 80...
    for ap_len in range(5, 15):
        A = list(range(1, ap_len + 1))
        next_val = A[-1] * 2
        while len(A) < 20:
            if next_val > 10000:
                break
            if next_val not in A:
                A.append(next_val)
            next_val *= 2
        if len(A) < 20:
            for k in range(1, 10001):
                if k not in A:
                    A.append(k)
                if len(A) == 20:
                    break
        try_candidate(A[:20])
        
    # 4. Random sets
    for _ in range(20):
        try_candidate(random.sample(range(1, 10001), 20))
    
    if best_set is None:
        return list(range(1, 21))
        
    # --- Phase 2: Local Search / Simulated Annealing ---
    
    # We will run multiple restarts from the best found so far,
    # or from slightly perturbed versions of the best found.
    
    def local_search(start_set, max_iters=None):
        nonlocal best_set, best_obj
        A = list(start_set)
        cur_obj = objective(A)
        
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
            
        # Simulated Annealing parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.9995
        
        it = 0
        no_improve_count = 0
        
        while time.time() < deadline - 2.0:
            it += 1
            if max_iters and it >= max_iters:
                break
            
            # Generate neighbor
            A_new = A[:]
            move_type = random.random()
            
            if move_type < 0.6:
                # Single element replacement
                idx = random.randrange(20)
                old_val = A_new[idx]
                
                # Propose new value
                # Mix of local (near old_val) and global (random)
                if random.random() < 0.5:
                    # Local move
                    # Scale step size with value?
                    step = max(1, int(old_val * 0.1))
                    new_val = old_val + random.randint(-step, step)
                else:
                    # Global move
                    new_val = random.randint(1, 10000)
                    
                if 1 <= new_val <= 10000 and new_val not in A_new:
                    A_new[idx] = new_val
                else:
                    continue
                    
            elif move_type < 0.8:
                # Swap two elements
                i, j = random.sample(range(20), 2)
                A_new[i], A_new[j] = A_new[j], A_new[i]
                
            else:
                # Perturb two elements
                i, j = random.sample(range(20), 2)
                for idx in [i, j]:
                    old_val = A_new[idx]
                    step = max(1, int(old_val * 0.05))
                    new_val = old_val + random.randint(-step, step)
                    if 1 <= new_val <= 10000 and new_val not in A_new:
                        A_new[idx] = new_val
                    else:
                        # If one fails, just perturb the other? Or skip move.
                        # Let's skip move if either fails to maintain simplicity
                        A_new = A[:]
                        break
            
            if A_new == A:
                continue
                
            new_obj = objective(A_new)
            
            # Acceptance criterion
            if new_obj < cur_obj:
                A = A_new
                cur_obj = new_obj
                no_improve_count = 0
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_set = A[:]
            else:
                # Accept worse moves with probability exp(-delta/T)
                delta = new_obj - cur_obj
                if random.random() < math.exp(-delta / T):
                    A = A_new
                    cur_obj = new_obj
                no_improve_count += 1
                
            # Cool down
            T = max(T_min, T * cooling)
            
            # Restart if stuck?
            if no_improve_count > 500:
                # Perturb current best to escape local minimum
                # Or just restart SA loop with higher T?
                # Let's just perturb the current A significantly
                for _ in range(5):
                    idx = random.randrange(20)
                    new_val = random.randint(1, 10000)
                    if new_val not in A:
                        A[idx] = new_val
                cur_obj = objective(A)
                T = 10.0
                no_improve_count = 0
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_set = A[:]
                    
        return A

    # Run local search from the best candidate found in Phase 1
    # and a few random restarts if time permits.
    
    # First run from global best
    local_search(best_set)
    
    # If time remains, try restarting from slightly perturbed best
    # or from other good candidates.
    
    # We can keep a list of "good" candidates from Phase 1
    # But for simplicity, we just run from the current best.
    # If we have time, we can do a second pass.
    
    # Let's check time
    if time.time() < deadline - 10.0:
        # Do a second run, maybe with different seed or perturbation
        # Create a perturbed version of the best set
        A_pert = best_set[:]
        for _ in range(10):
            idx = random.randrange(20)
            new_val = random.randint(1, 10000)
            if new_val not in A_pert:
                A_pert[idx] = new_val
        local_search(A_pert)
        
    if time.time() < deadline - 10.0:
        # Third run from a random restart
        A_rand = random.sample(range(1, 10001), 20)
        local_search(A_rand)
        
    return best_set

