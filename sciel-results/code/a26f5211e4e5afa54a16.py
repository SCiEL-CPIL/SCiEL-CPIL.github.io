import time
import random
from itertools import combinations

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set()
    prodset = set()
    for i in range(len(A)):
        for j in range(len(A)):
            sumset.add(A[i] + A[j])
            prodset.add(A[i] * A[j])
    return max(len(sumset), len(prodset))

def generate_smooth_numbers(limit, primes, max_count=200000):
    """Generate smooth numbers up to limit using given primes."""
    smooth = [1]
    i = 0
    while i < len(smooth) and len(smooth) < max_count:
        for p in primes:
            val = smooth[i] * p
            if val <= limit:
                smooth.append(val)
            else:
                break
        i += 1
    return sorted(set(smooth))

def local_search_with_pairs(A, time_limit, rng, candidate_pool, use_pair_swaps=True):
    """
    Local search with single and pair swap moves.
    Uses incremental score estimation for speed.
    """
    A = list(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    n = len(A)
    
    # Precompute current sumset and prodset
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    
    improved = True
    no_improve_rounds = 0
    max_no_improve = 5
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_rounds += 1
        if no_improve_rounds > max_no_improve:
            break
            
        # Shuffle candidates for diversity
        shuffled_pool = candidate_pool.copy()
        rng.shuffle(shuffled_pool)
        
        # Try single element swaps
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            for new_val in shuffled_pool:
                if new_val == old_val or new_val in A:
                    continue
                if new_val < 1 or new_val > 10000:
                    continue
                
                # Quick estimate
                new_sum_add = set()
                new_prod_add = set()
                old_sum_rem = set()
                old_prod_rem = set()
                
                for other in A:
                    if other is not A[i]:
                        new_sum_add.add(new_val + other)
                        new_prod_add.add(new_val * other)
                        old_sum_rem.add(old_val + other)
                        old_prod_rem.add(old_val * other)
                
                new_sum_add.add(2 * new_val)
                new_prod_add.add(new_val * new_val)
                old_sum_rem.add(2 * old_val)
                old_prod_rem.add(old_val * old_val)
                
                est_sum = len(sumset) + len(new_sum_add - sumset) - len(old_sum_rem & (sumset - new_sum_add))
                est_prod = len(prodset) + len(new_prod_add - prodset) - len(old_prod_rem & (prodset - new_prod_add))
                est_score = max(est_sum, est_prod)
                
                if est_score < current_score:
                    new_A = A.copy()
                    new_A[i] = new_val
                    score = calc_score(new_A)
                    
                    if score < current_score:
                        current_score = score
                        A = new_A
                        improved = True
                        no_improve_rounds = 0
                        sumset = set(a + b for a in A for b in A)
                        prodset = set(a * b for a in A for b in A)
                        
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                        break
        
        # Try pair swaps
        if use_pair_swaps and (time.time() - start_time) < time_limit - 2.0:
            for i in range(n):
                for j in range(i + 1, n):
                    if (time.time() - start_time) > time_limit:
                        return best_A, best_score
                    
                    old_i, old_j = A[i], A[j]
                    
                    # Try swapping A[i] with a candidate and A[j] with another
                    for new_i in shuffled_pool[:30]:
                        if new_i == old_i or new_i == old_j or new_i in A:
                            continue
                        if new_i < 1 or new_i > 10000:
                            continue
                        
                        for new_j in shuffled_pool[:30]:
                            if new_j == old_i or new_j == old_j or new_j == new_i or new_j in A:
                                continue
                            if new_j < 1 or new_j > 10000:
                                continue
                            
                            new_A = A.copy()
                            new_A[i] = new_i
                            new_A[j] = new_j
                            
                            score = calc_score(new_A)
                            if score < current_score:
                                current_score = score
                                A = new_A
                                improved = True
                                no_improve_rounds = 0
                                sumset = set(a + b for a in A for b in A)
                                prodset = set(a * b for a in A for b in A)
                                
                                if score < best_score:
                                    best_score = score
                                    best_A = list(A)
                                break
                        if improved:
                            break
                    if improved:
                        break
        elif use_pair_swaps and not improved:
            # Simpler pair swap: just swap values at positions i and j
            for i in range(n):
                for j in range(i + 1, n):
                    if (time.time() - start_time) > time_limit:
                        return best_A, best_score
                    
                    new_A = A.copy()
                    new_A[i], new_A[j] = A[j], A[i]
                    score = calc_score(new_A)
                    if score < current_score:
                        current_score = score
                        A = new_A
                        improved = True
                        no_improve_rounds = 0
                        sumset = set(a + b for a in A for b in A)
                        prodset = set(a * b for a in A for b in A)
                        
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                        break
                if improved:
                    break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Generate a rich pool of smooth numbers with more primes
    primes = [2, 3, 5, 7, 11]
    smooth_numbers = generate_smooth_numbers(10000, primes)
    
    # Also add small integers for diversity
    candidate_pool = list(range(1, 61)) + smooth_numbers
    candidate_pool = sorted(set(candidate_pool))
    
    best_A = None
    best_score = float('inf')
    
    # Multiple initial candidates with different structures
    
    # Candidate 1: Smallest smooth numbers
    cand1 = smooth_numbers[:20]
    
    # Candidate 2: Arithmetic progression (known to minimize sumset)
    cand2 = list(range(1, 21))
    
    # Candidate 3: Powers of 2 and 3 mixed
    cand3 = []
    for e2 in range(15):
        for e3 in range(10):
            val = (2 ** e2) * (3 ** e3)
            if val <= 10000 and val not in cand3:
                cand3.append(val)
            if len(cand3) >= 20:
                break
        if len(cand3) >= 20:
            break
    cand3 = sorted(cand3[:20])
    
    # Candidate 4: Include 1 and some smooth numbers
    cand4 = [1] + smooth_numbers[1:19]
    
    # Candidate 5: Random selection from smooth numbers
    cand5 = sorted(rng.sample(smooth_numbers[:100], 20))
    
    # Candidate 6: Geometric-like with smooth numbers
    cand6 = []
    val = 1
    while val <= 10000 and len(cand6) < 20:
        cand6.append(val)
        if val % 2 == 0:
            val = val * 3
        elif val % 3 == 0:
            val = val * 2
        else:
            val = val * 2
    cand6 = sorted(cand6[:20])
    
    # Candidate 7: Optimized - use numbers that are highly composite
    cand7 = sorted([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    
    # Candidate 8: Lattice-based - 2^x * 3^y grid points
    cand8 = []
    for x in range(15):
        for y in range(10):
            val = (2 ** x) * (3 ** y)
            if val <= 10000 and val not in cand8:
                cand8.append(val)
            if len(cand8) >= 20:
                break
        if len(cand8) >= 20:
            break
    cand8 = sorted(cand8[:20])
    
    # Candidate 9: Dense geometric with ratio ~1.5
    cand9 = []
    val = 1
    while val <= 10000 and len(cand9) < 20:
        cand9.append(val)
        val = int(val * 1.5) + 1
    cand9 = sorted(set(cand9))[:20]
    
    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8, cand9]
    
    # Remove invalid candidates
    valid_candidates = []
    for c in candidates:
        if len(set(c)) == 20 and all(1 <= x <= 10000 for x in c):
            valid_candidates.append(list(c))
    
    # Evaluate initial scores
    initial_scores = []
    for c in valid_candidates:
        s = calc_score(c)
        initial_scores.append((s, c))
    
    initial_scores.sort(key=lambda x: x[0])
    
    # Run local search on the best candidates
    # Allocate time proportionally, with more time for top candidates
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Allocate time: top 3 get more, rest get less
        time_allocs = []
        for idx, (score, cand) in enumerate(initial_scores):
            if idx < 3:
                t = min(15.0, remaining_time * 0.25)
            else:
                t = min(8.0, remaining_time * 0.1)
            time_allocs.append(t)
        
        for idx, (score, cand) in enumerate(initial_scores):
            if time.time() > deadline - 5.0:
                break
            
            if score < best_score:
                best_score = score
                best_A = list(cand)
            
            # Only do local search if we have enough time
            remaining = deadline - time.time()
            if remaining < 8.0:
                break
            
            alloc_time = min(time_allocs[idx], remaining * 0.7)
            
            improved_A, ls_score = local_search_with_pairs(cand, alloc_time, rng, candidate_pool, use_pair_swaps=True)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = improved_A
    
    # Final check: try a few random restarts if time permits
    if best_A is not None and time.time() < deadline - 15.0:
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Start from current best and perturb
            start_A = best_A.copy()
            # Perturb 2-3 elements
            num_perturb = rng.randint(2, 3)
            for _ in range(num_perturb):
                idx = rng.randint(0, 19)
                new_val = rng.choice(candidate_pool)
                attempts = 0
                while new_val in start_A and attempts < 50:
                    new_val = rng.choice(candidate_pool)
                    attempts += 1
                if new_val not in start_A:
                    start_A[idx] = new_val
            
            if len(set(start_A)) != 20:
                continue
            
            s = calc_score(start_A)
            if s < best_score:
                best_score = s
                best_A = start_A
            
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            improved_A, ls_score = local_search_with_pairs(start_A, min(10.0, remaining * 0.7), rng, candidate_pool, use_pair_swaps=True)
            if ls_score < best_score:
                best_score = ls_score
                best_A = improved_A
    
    # Final polish: try to improve by testing nearby values
    if best_A is not None and time.time() < deadline - 3.0:
        polish_A = best_A.copy()
        polish_score = best_score
        changed = True
        while changed and (time.time() - start_time) < deadline - 2.0:
            changed = False
            for i in range(20):
                old = polish_A[i]
                # Try nearby values in the pool
                for offset in [-3, -2, -1, 1, 2, 3]:
                    new_val = old + offset
                    if new_val < 1 or new_val > 10000 or new_val in polish_A:
                        continue
                    if new_val == old:
                        continue
                    
                    test_A = polish_A.copy()
                    test_A[i] = new_val
                    s = calc_score(test_A)
                    if s < polish_score:
                        polish_score = s
                        polish_A = test_A
                        changed = True
                        break
            
            if polish_score < best_score:
                best_score = polish_score
                best_A = polish_A
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A
