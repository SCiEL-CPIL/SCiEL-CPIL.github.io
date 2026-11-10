import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_candidate_pool():
    """
    Generate a pool of candidate integers likely to be in optimal sets.
    Focus on small integers (good for sumsets) and smooth numbers (good for product sets).
    """
    candidates = set()
    
    # 1. Small integers (1-200)
    candidates.update(range(1, 201))
    
    # 2. 2, 3, 5, 7-smooth numbers
    primes = [2, 3, 5, 7]
    max_val = 10000
    
    def gen_smooth(primes, max_val, current=1, index=0):
        if current > max_val:
            return
        if index == len(primes):
            candidates.add(current)
            return
        p = primes[index]
        gen_smooth(primes, max_val, current, index + 1)
        val = current
        while val * p <= max_val:
            val *= p
            gen_smooth(primes, max_val, val, index)
            
    gen_smooth(primes, max_val)
    
    # 3. Add some random numbers to ensure diversity
    for _ in range(500):
        candidates.add(random.randint(1, 10000))
        
    return sorted(candidates)

def ap_parameter_search(time_limit, rng):
    """
    Search over arithmetic progression parameters (start, step) to find
    good AP-structured initial candidates.
    Returns the best AP and its score.
    """
    best_A = None
    best_score = float('inf')
    start_time = time.time()
    n = 20
    
    # Coarse grid search over start and step
    # Small steps give small sumsets (size 2n-1 = 39)
    # We want to find APs that also have small product sets
    for s in range(1, 51):
        if (time.time() - start_time) > time_limit:
            break
        for d in range(1, 21):
            last = s + (n - 1) * d
            if last > 10000:
                continue
            A = [s + i * d for i in range(n)]
            score = calc_score(A)
            if score < best_score:
                best_score = score
                best_A = A
                
    return best_A, best_score

def local_search(A, time_limit, candidate_pool, rng):
    """
    Perform local search on the set A.
    Tries to replace elements with candidates from the pool to improve the score.
    Uses a hybrid approach: deterministic sweep for initial improvements, then random sampling.
    """
    A = list(A)
    A.sort()
    n = len(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Pre-compute a smaller, high-priority subset of the candidate pool
    priority_pool = [x for x in candidate_pool if x <= 1000]
    if len(priority_pool) > 500:
        smalls = [x for x in priority_pool if x <= 50]
        smooths = [x for x in priority_pool if x > 50]
        rng.shuffle(smooths)
        priority_pool = smalls + smooths[:500 - len(smalls)]
        
    full_pool = candidate_pool
    
    improved = True
    passes = 0
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_set = set(A)
        
        if passes % 2 == 0:
            # Deterministic sweep: try all candidates for each position
            for i in range(n):
                if (time.time() - start_time) > time_limit:
                    return best_A, best_score
                
                old_val = A[i]
                
                for new_val in priority_pool:
                    if new_val in A_set or new_val == old_val:
                        continue
                    
                    new_A = A.copy()
                    new_A[i] = new_val
                    
                    score = calc_score(new_A)
                    
                    if score < current_score:
                        current_score = score
                        A = new_A
                        A_set.remove(old_val)
                        A_set.add(new_val)
                        improved = True
                        
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                        
                        break 
        else:
            # Random sampling: try random candidates to escape local minima
            num_random_tries = min(100, len(full_pool))
            for _ in range(num_random_tries):
                if (time.time() - start_time) > time_limit:
                    return best_A, best_score
                
                i = rng.randint(0, n - 1)
                old_val = A[i]
                
                candidates_not_in_A = [x for x in full_pool if x not in A_set and x != old_val]
                if not candidates_not_in_A:
                    continue
                    
                new_val = rng.choice(candidates_not_in_A)
                
                new_A = A.copy()
                new_A[i] = new_val
                
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    A_set.remove(old_val)
                    A_set.add(new_val)
                    improved = True
                    
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    
                    break
        
        passes += 1
        
    return best_A, best_score

def fine_polish(A, time_limit, rng):
    """
    Fine-grained polish: try small perturbations around each element.
    """
    A = list(A)
    A.sort()
    n = len(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    A_set = set(A)
    
    while (time.time() - start_time) < time_limit:
        improved = False
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            for delta in range(1, 20):
                for new_val in (old_val - delta, old_val + delta):
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in A_set:
                        continue
                    
                    new_A = A.copy()
                    new_A[i] = new_val
                    
                    score = calc_score(new_A)
                    
                    if score < current_score:
                        current_score = score
                        A = new_A
                        A_set.remove(old_val)
                        A_set.add(new_val)
                        improved = True
                        
                        if score < best_score:
                            best_score = score
                            best_A = list(A)
                        
                        break
                if improved:
                    break
        
        if not improved:
            for _ in range(5):
                if (time.time() - start_time) > time_limit:
                    return best_A, best_score
                
                i = rng.randint(0, n - 1)
                old_val = A[i]
                
                new_val = old_val + rng.randint(-50, 50)
                if new_val < 1 or new_val > 10000 or new_val in A_set:
                    continue
                    
                new_A = A.copy()
                new_A[i] = new_val
                
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    A_set.remove(old_val)
                    A_set.add(new_val)
                    improved = True
                    
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    
                    break
        
        if not improved:
            break
        
    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds (limit is 90s)
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Generate the candidate pool once
    candidate_pool = generate_candidate_pool()
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: AP parameter search - find the best pure AP
    ap_time = min(5.0, (deadline - start_time) * 0.1)
    ap_A, ap_score = ap_parameter_search(ap_time, rng)
    
    if ap_A is not None:
        best_A = ap_A
        best_score = ap_score
    
    # Phase 2: Generate diverse initial candidates
    initial_candidates = []
    
    # 1. The best AP found
    if ap_A is not None:
        initial_candidates.append(ap_A)
    
    # 2. Additional APs with small parameters
    for s in range(1, 11):
        for d in range(1, 11):
            last = s + 19 * d
            if last > 10000:
                continue
            A = [s + i * d for i in range(20)]
            if len(set(A)) == 20:
                initial_candidates.append(A)
                
    # 3. Hybrid Sets: S U cS
    for k in range(5, 11):
        for c in range(k + 1, k + 20):
            S = list(range(1, k + 1))
            T = [c * x for x in S]
            A = S + T
            if len(set(A)) == 20 and max(A) <= 10000:
                initial_candidates.append(sorted(A))
                
    # 4. Geometric Progressions (small ratio)
    for a in range(1, 11):
        for r in range(2, 4):
            A = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                A.append(val)
                val *= r
            if len(A) == 20:
                initial_candidates.append(A)
                
    # 5. Random Smooth Sets
    smooth_nums = [x for x in candidate_pool if x <= 500]
    for _ in range(10):
        try:
            A = rng.sample(smooth_nums, 20)
            A = sorted(A)
            if max(A) <= 10000:
                initial_candidates.append(A)
        except ValueError:
            continue
            
    # Deduplicate initial candidates
    unique_candidates = []
    seen = set()
    for A in initial_candidates:
        key = tuple(sorted(A))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(A)
            
    # Sort initial candidates by their initial score to prioritize good starting points
    scored_candidates = []
    for A in unique_candidates:
        score = calc_score(A)
        scored_candidates.append((score, A))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Local search on the best candidates
    num_candidates = len(scored_candidates)
    
    for idx, (init_score, A) in enumerate(scored_candidates):
        if time.time() > deadline - 5.0:
            break
            
        # Skip very poor candidates if we have a decent best
        if best_score < float('inf') and init_score > best_score * 1.5 and idx > 5:
            continue
            
        remaining_time = deadline - time.time() - 5.0
        if remaining_time <= 0:
            break
            
        # Allocate a fraction of the remaining time, but at least 0.5s, max 5s per candidate
        alloc_time = max(0.5, min(5.0, remaining_time / max(1, num_candidates - idx)))
        
        improved_A, score = local_search(A, alloc_time, candidate_pool, rng)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    # Phase 4: Fine polish on the best result
    if best_A is not None and time.time() < deadline - 2.0:
        polish_time = min(10.0, deadline - time.time() - 2.0)
        if polish_time > 1.0:
            polished_A, polish_score = fine_polish(best_A, polish_time, rng)
            if polish_score < best_score:
                best_score = polish_score
                best_A = polished_A
        
    # Fallback
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score(best_A)
        
    return best_A

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
