import time
import random
import itertools

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
    
    def local_search_with_sa(A, time_limit, rng):
        """
        Local search with simulated annealing to escape local minima.
        """
        A = list(A)
        current_score = calc_score_fast(A)
        best_score = current_score
        best_A = list(A)
        
        # Build candidate pool
        pool = list(range(1, 101))
        # Add smooth numbers: 2^a * 3^b * 5^c
        for a in range(1, 14):
            for b in range(1, 10):
                for c in range(1, 6):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if val <= 10000:
                        pool.append(val)
        # Add some random numbers
        for _ in range(200):
            pool.append(rng.randint(1, 10000))
        pool = list(set(pool))
        rng.shuffle(pool)
        
        # Simulated annealing parameters
        temp = 10.0
        cooling = 0.995
        min_temp = 0.1
        
        while time.time() - start_time < time_limit and temp > min_temp:
            # Pick a random position to modify
            i = rng.randint(0, len(A) - 1)
            old_val = A[i]
            
            # Pick a new value
            new_val = rng.choice(pool)
            if new_val == old_val or new_val in A:
                continue
            
            A[i] = new_val
            new_score = calc_score_fast(A)
            
            # Accept if better, or with probability exp(-(delta)/temp) if worse
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
    
    # 2. Powers of 2 + multiples of 3*2^k
    A2 = []
    for k in range(10):
        A2.append(2**k)
    for k in range(10):
        A2.append(3 * 2**k)
    if len(set(A2)) == 20:
        candidates.append(A2)
    
    # 3. Small smooth numbers (2,3-smooth)
    smooth = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth.append(val)
    smooth = sorted(set(smooth))
    if len(smooth) >= 20:
        candidates.append(smooth[:20])
    
    # 4. Geometric progression with ratio 2, extended with small multiples
    gp2 = [2**i for i in range(14)]  # 1, 2, 4, ..., 8192
    # Add 6 more numbers that are small multiples
    extra = [3, 6, 12, 24, 48, 96]
    A4 = gp2[:14] + extra
    if len(set(A4)) == 20:
        candidates.append(A4)
    
    # 5. Mixed: powers of 2 and 3
    mixed = []
    for a in range(7):
        for b in range(3):
            val = (2**a) * (3**b)
            if val <= 10000:
                mixed.append(val)
    mixed = sorted(set(mixed))
    if len(mixed) >= 20:
        candidates.append(mixed[:20])
    
    # 6. Random small numbers
    A6 = rng.sample(range(1, 100), 20)
    candidates.append(A6)
    
    # 7. Random smooth numbers
    smooth_pool = []
    for a in range(1, 13):
        for b in range(1, 8):
            for c in range(1, 4):
                val = (2**a) * (3**b) * (5**c)
                if val <= 10000:
                    smooth_pool.append(val)
    smooth_pool = list(set(smooth_pool))
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
    
    # Run local search with simulated annealing from the best candidate
    time_left = deadline - time.time()
    if time_left > 2.0:
        # Run SA from best candidate
        sa_result, sa_score = local_search_with_sa(best_A, time_left - 1.0, rng)
        if sa_score < best_score:
            best_score = sa_score
            best_A = sa_result
        
        # Try SA from a few other good candidates
        time_left = deadline - time.time()
        if time_left > 2.0:
            for cand in candidates[:3]:
                if cand == best_A or len(set(cand)) != 20:
                    continue
                score = calc_score_fast(cand)
                if score <= best_score + 10:  # Only try if reasonably good
                    sa_result2, sa_score2 = local_search_with_sa(cand, min(time_left - 1.0, 15.0), rng)
                    if sa_score2 < best_score:
                        best_score = sa_score2
                        best_A = sa_result2
                    time_left = deadline - time.time()
                    if time_left < 2.0:
                        break
    
    return best_A
