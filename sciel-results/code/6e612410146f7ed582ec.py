import time
import random

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for the 90s limit
    
    rng = random.Random(42)
    
    def calc_score_fast(A):
        A = sorted(A)
        sumset = set()
        prodset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
        
    def local_search_exhaustive(A, time_limit, rng):
        """
        Exhaustive 1-swap local search.
        Tries replacing each element with every candidate in the pool.
        Accepts any improvement immediately.
        """
        A = list(A)
        current_score = calc_score_fast(A)
        best_score = current_score
        best_A = list(A)
        start_time_ls = time.time()
        
        # Build a comprehensive candidate pool
        pool = set()
        
        # 1. Small integers
        for i in range(1, 101):
            pool.add(i)
            
        # 2. 2,3,5-smooth numbers
        for a in range(1, 15):
            for b in range(1, 10):
                for c in range(1, 6):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if val <= 10000:
                        pool.add(val)
                        
        # 3. 2,3-smooth numbers
        for a in range(1, 15):
            for b in range(1, 12):
                val = (2 ** a) * (3 ** b)
                if val <= 10000:
                    pool.add(val)
                    
        # 4. Powers of 2
        for a in range(1, 14):
            pool.add(2 ** a)
            
        # 5. Powers of 3
        for b in range(1, 9):
            pool.add(3 ** b)
            
        # 6. Random numbers
        for _ in range(100):
            pool.add(rng.randint(1, 10000))
            
        pool = list(pool)
        rng.shuffle(pool)
        
        improved = True
        while improved and (time.time() - start_time_ls) < time_limit:
            improved = False
            for i in range(len(A)):
                if (time.time() - start_time_ls) > time_limit:
                    break
                
                old_val = A[i]
                # Try swapping with candidates in the pool
                # To speed up, we might not try all 1000+ candidates if it's too slow, 
                # but for n=20, calc_score is fast enough.
                # However, 20 * 1000 * 200 ops ~ 4M ops per pass. 
                # In Python, this might take ~0.1-0.2s per pass.
                # So we can do a few passes.
                
                for new_val in pool:
                    if new_val == old_val or new_val in A:
                        continue
                    
                    # Quick check: if new_val is very large, product set likely grows
                    # But let's just compute.
                    
                    # Temporarily replace
                    A[i] = new_val
                    new_score = calc_score_fast(A)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_A = list(A)
                        improved = True
                        # Keep the new value and restart the outer loop? 
                        # Or continue? 
                        # Standard greedy: continue to next i, but since we changed A,
                        # the "old_val" for subsequent i's is still valid.
                        # But the score changed, so we should probably restart the i-loop 
                        # to ensure we don't miss improvements caused by the change.
                        # However, restarting is expensive. 
                        # Let's just break the inner loop and continue the outer loop.
                        # If we find a better one later, we'll accept it.
                        break
                    else:
                        A[i] = old_val
        
        return best_A, best_score

    def local_search_sa(A, time_limit, rng):
        """
        Simulated Annealing for escaping local minima.
        """
        A = list(A)
        current_score = calc_score_fast(A)
        best_score = current_score
        best_A = list(A)
        start_time_ls = time.time()
        
        # Pool for SA
        pool = list(range(1, 101))
        for a in range(1, 14):
            for b in range(1, 10):
                for c in range(1, 6):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if val <= 10000:
                        pool.append(val)
        pool = list(set(pool))
        rng.shuffle(pool)
        
        temp = 5.0
        cooling = 0.99
        min_temp = 0.1
        
        while time.time() - start_time_ls < time_limit and temp > min_temp:
            i = rng.randint(0, len(A) - 1)
            old_val = A[i]
            new_val = rng.choice(pool)
            if new_val == old_val or new_val in A:
                continue
            
            A[i] = new_val
            new_score = calc_score_fast(A)
            
            delta = new_score - current_score
            if delta < 0 or rng.random() < pow(2.718281828, -delta / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(A)
            else:
                A[i] = old_val
            
            temp *= cooling
        
        return best_A, best_score

    # Generate diverse initial candidates
    candidates = []
    
    # 1. AP 1..20
    candidates.append(list(range(1, 21)))
    
    # 2. Small smooth numbers (2,3,5-smooth)
    smooth_pool = []
    for a in range(1, 14):
        for b in range(1, 10):
            for c in range(1, 6):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 10000:
                    smooth_pool.append(val)
    smooth_pool = sorted(set(smooth_pool))
    if len(smooth_pool) >= 20:
        candidates.append(smooth_pool[:20])
        
    # 3. Mixed: Powers of 2 and small multiples
    gp2 = [2**i for i in range(14)]
    extra = [3, 6, 12, 24, 48, 96]
    A3 = gp2 + extra
    if len(set(A3)) == 20:
        candidates.append(A3)
        
    # 4. Random small numbers
    A4 = rng.sample(range(1, 100), 20)
    candidates.append(A4)
    
    # 5. Random smooth numbers
    if len(smooth_pool) >= 20:
        candidates.append(rng.sample(smooth_pool, 20))
        
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(set(cand)) != 20 or any(x > 10000 for x in cand):
            continue
        score = calc_score_fast(cand)
        if score < best_score:
            best_score = score
            best_A = list(cand)
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score_fast(best_A)
        
    # Run local search
    time_left = deadline - time.time()
    
    # First, run exhaustive local search on the best candidate
    if time_left > 2.0:
        # Allocate most of the time to exhaustive search
        # But leave some for SA if we get stuck
        alloc_exh = min(40.0, time_left - 1.0)
        exh_result, exh_score = local_search_exhaustive(best_A, alloc_exh, rng)
        if exh_score < best_score:
            best_score = exh_score
            best_A = exh_result
            
    # Then, run SA from the best found so far
    time_left = deadline - time.time()
    if time_left > 2.0:
        alloc_sa = min(20.0, time_left - 1.0)
        sa_result, sa_score = local_search_sa(best_A, alloc_sa, rng)
        if sa_score < best_score:
            best_score = sa_score
            best_A = sa_result
            
        # If time remains, try SA from another good candidate
        time_left = deadline - time.time()
        if time_left > 2.0:
            # Pick the second best initial candidate
            for cand in candidates:
                if cand == best_A or len(set(cand)) != 20:
                    continue
                score = calc_score_fast(cand)
                if score <= best_score + 10:
                    alloc_sa2 = min(10.0, time_left - 1.0)
                    sa_result2, sa_score2 = local_search_sa(cand, alloc_sa2, rng)
                    if sa_score2 < best_score:
                        best_score = sa_score2
                        best_A = sa_result2
                    break
    
    return best_A
