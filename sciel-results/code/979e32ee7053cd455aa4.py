import time
import random
import math
import itertools

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
    
    # 1. Small integers (1-500)
    candidates.update(range(1, 501))
    
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
    
    return sorted(candidates)

def local_search(A, time_limit, candidate_pool, rng):
    """
    Perform local search on the set A.
    Tries to replace elements with candidates from the pool to improve the score.
    """
    A = list(A)
    A.sort()
    n = len(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Pre-compute a smaller, high-priority subset of the candidate pool
    priority_pool = [x for x in candidate_pool if x <= 3000]
    if len(priority_pool) > 1000:
        smalls = [x for x in priority_pool if x <= 150]
        smooths = [x for x in priority_pool if x > 150]
        rng.shuffle(smooths)
        priority_pool = smalls + smooths[:1000 - len(smalls)]
        
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_set = set(A)
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            best_new_val = None
            best_new_score = current_score
            
            # Try multiple candidates and pick the best improvement
            for new_val in priority_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                score = calc_score(new_A)
                
                if score < best_new_score:
                    best_new_score = score
                    best_new_val = new_val
                    
            if best_new_val is not None:
                current_score = best_new_score
                A[i] = best_new_val
                A_set.remove(old_val)
                A_set.add(best_new_val)
                improved = True
                
                if score < best_score:
                    best_score = score
                    best_A = list(A)
                    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    
    rng = random.Random(42)
    
    candidate_pool = generate_candidate_pool()
    
    best_A = None
    best_score = float('inf')
    
    initial_candidates = []
    
    # 1. Arithmetic Progressions
    for a in range(1, 15):
        for d in range(1, 15):
            last = a + 19 * d
            if last > 10000:
                continue
            A = [a + i * d for i in range(20)]
            if len(set(A)) == 20:
                initial_candidates.append(A)
                
    # 2. Hybrid Sets: S U cS
    for k in range(4, 12):
        for c in range(k + 1, k + 30):
            S = list(range(1, k + 1))
            T = [c * x for x in S]
            A = S + T
            if len(set(A)) == 20 and max(A) <= 10000:
                initial_candidates.append(sorted(A))
                
    # 3. Geometric Progressions
    for a in range(1, 11):
        for r in range(2, 5):
            A = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                A.append(val)
                val *= r
            if len(A) == 20:
                initial_candidates.append(A)
                
    # 4. Random Smooth Sets
    smooth_nums = [x for x in candidate_pool if x <= 800]
    for _ in range(15):
        try:
            A = rng.sample(smooth_nums, 20)
            A = sorted(A)
            if max(A) <= 10000:
                initial_candidates.append(A)
        except ValueError:
            continue
            
    # 5. Lattice-based Construction
    pts = []
    for a in range(0, 14):
        for b in range(0, 14):
            val = (2 ** a) * (3 ** b)
            if val <= 10000:
                pts.append((a, b, val))
    pts.sort(key=lambda x: x[2])
    if len(pts) >= 20:
        A_lattice1 = [p[2] for p in pts[:20]]
        if len(set(A_lattice1)) == 20:
            initial_candidates.append(sorted(A_lattice1))
            
    # 6. Mixed structures: small AP + smooth tail
    for base_k in range(5, 10):
        for offset in range(1, 5):
            S = list(range(1, base_k + 1))
            smooth_tail = [x for x in candidate_pool if x > base_k * 2 and x <= 2000][:20 - base_k]
            if len(smooth_tail) >= 20 - base_k:
                A = S + smooth_tail[:20 - base_k]
                if len(set(A)) == 20 and max(A) <= 10000:
                    initial_candidates.append(sorted(A))
                    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for A in initial_candidates:
        key = tuple(sorted(A))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(A)
            
    scored_candidates = []
    for A in unique_candidates:
        score = calc_score(A)
        scored_candidates.append((score, A))
    scored_candidates.sort(key=lambda x: x[0])
    
    num_candidates = len(scored_candidates)
    
    for idx, (init_score, A) in enumerate(scored_candidates):
        if time.time() > deadline - 3.0:
            break
            
        if init_score > best_score * 1.5 and idx > 8:
            continue
            
        remaining_time = deadline - time.time()
        alloc_time = max(0.5, min(8.0, remaining_time / max(1, num_candidates - idx)))
        
        improved_A, score = local_search(A, alloc_time, candidate_pool, rng)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    # Final polish: if we have time, do one more pass on the best solution
    if best_A is not None and (deadline - time.time()) > 5.0:
        polish_time = min(15.0, (deadline - time.time()) - 2.0)
        if polish_time > 2.0:
            polished_A, polish_score = local_search(best_A, polish_time, candidate_pool, rng)
            if polish_score < best_score:
                best_score = polish_score
                best_A = polished_A
                
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
