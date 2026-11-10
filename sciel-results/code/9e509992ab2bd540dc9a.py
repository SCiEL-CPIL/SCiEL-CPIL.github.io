import time
import random
import math
import numpy as np

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for a set of distinct positive integers."""
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

def _calc_score_np(A_list):
    """Compute max(|A+A|, |A*A|) using numpy for speed."""
    A = np.array(A_list, dtype=np.int64)
    n = len(A)
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    sumset_size = len(np.unique(sums))
    prodset_size = len(np.unique(prods))
    return max(sumset_size, prodset_size)

def _generate_geometric_sets(max_val=10000, n=20):
    """Generate geometric progression sets with various bases and ratios."""
    sets = []
    
    # Try integer bases with rational ratios
    for base in range(1, 50):
        for num in range(1, 20):
            for den in range(num + 1, min(num + 20, 30)):
                ratio = num / den
                if ratio <= 1.0:
                    continue
                seq = [base]
                for i in range(1, n):
                    next_val = int(round(seq[-1] * ratio))
                    if next_val <= seq[-1]:
                        break
                    if next_val > max_val:
                        break
                    seq.append(next_val)
                if len(seq) >= n:
                    sets.append(seq[:n])
    
    # Try pure powers: base^e_i where e_i are chosen exponents
    for base in range(2, 20):
        # Powers of base
        powers = []
        val = 1
        while val <= max_val:
            powers.append(val)
            val *= base
        if len(powers) >= n:
            # Try different subsets of exponents
            for start in range(len(powers) - n + 1):
                sets.append(powers[start:start + n])
    
    # Try base^e * k where k is small
    for base in range(2, 15):
        for k in range(1, 10):
            seq = []
            for e in range(n):
                val = (base ** e) * k
                if val > max_val or val in seq:
                    break
                seq.append(val)
            if len(seq) >= n:
                sets.append(seq[:n])
    
    return sets

def _generate_smooth_candidates(max_val, primes):
    """Generate smooth numbers with given prime factors <= max_val."""
    vals = set()
    
    def gen(idx, current_val):
        if current_val > max_val:
            return
        vals.add(current_val)
        if idx < len(primes):
            p = primes[idx]
            v = current_val
            while v * p <= max_val:
                v *= p
                gen(idx, v)
            gen(idx + 1, current_val)
    
    gen(0, 1)
    return sorted(list(vals))

def _local_search(A, pool, timeout=30.0, max_pool_size=500):
    """Local search: try swapping elements with candidates from pool."""
    A = sorted(set(A))
    if len(A) != 20:
        return A, float('inf')
    
    best_A = list(A)
    best_score = _calc_score(A)
    start_time = time.time()
    
    A_set = set(A)
    
    # Limit pool size for speed
    if len(pool) > max_pool_size:
        # Prioritize smaller numbers and smooth numbers
        pool = sorted(pool)[:max_pool_size]
    
    it = 0
    while it < 50000:
        if time.time() - start_time > timeout:
            break
        
        improved = False
        for idx in range(len(A)):
            old_val = A[idx]
            for new_val in pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    improved = True
                    break
            if time.time() - start_time > timeout:
                break
        
        it += 1
        if not improved:
            # Try random perturbations to escape local minima
            for _ in range(5):
                if time.time() - start_time > timeout:
                    break
                idx = random.randint(0, len(A) - 1)
                candidates = [x for x in pool if x not in A_set and x != A[idx]]
                if not candidates:
                    break
                new_val = random.choice(candidates)
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    improved = True
            if not improved:
                break
    
    return best_A, best_score

def _fine_polish(A, timeout=10.0):
    """Fine-grained polish: try small perturbations around each element."""
    A = sorted(set(A))
    if len(A) != 20:
        return A, float('inf')
    
    best_A = list(A)
    best_score = _calc_score(A)
    start_time = time.time()
    
    A_set = set(A)
    
    for it in range(50000):
        if time.time() - start_time > timeout:
            break
        
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        found_improvement = False
        for delta in range(1, 100):
            for new_val in (old_val - delta, old_val + delta):
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A_set:
                    continue
                
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    found_improvement = True
                    break
            if found_improvement:
                break
    
    return best_A, best_score

def _geometric_refinement(A, timeout=20.0):
    """Refine a geometric-like set by trying multiplicative perturbations."""
    A = sorted(set(A))
    if len(A) != 20:
        return A, float('inf')
    
    best_A = list(A)
    best_score = _calc_score(A)
    start_time = time.time()
    
    A_set = set(A)
    
    # Try replacing each element with nearby values that maintain geometric structure
    for it in range(20000):
        if time.time() - start_time > timeout:
            break
        
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Try values that are multiples/divisors of nearby geometric terms
        found = False
        for factor in range(1, 20):
            for new_val in (old_val * factor, old_val // factor if old_val % factor == 0 else -1):
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A_set:
                    continue
                
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    found = True
                    break
            if found:
                break
        
        # Also try small additive perturbations
        if not found:
            for delta in range(-50, 51):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A_set:
                    continue
                
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    found = True
                    break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Generate candidate pools
    pool_2smooth = _generate_smooth_candidates(10000, [2, 3])
    pool_3smooth = _generate_smooth_candidates(10000, [2, 3, 5])
    pool_4smooth = _generate_smooth_candidates(10000, [2, 3, 5, 7])
    
    candidate_pool = list(set(pool_2smooth + pool_3smooth + pool_4smooth))
    
    # Generate geometric progression sets
    geo_sets = _generate_geometric_sets(10000, 20)
    
    # Initial sets to try
    initial_sets = []
    
    # Add geometric sets
    for gs in geo_sets[:50]:
        gs = sorted(set(x for x in gs if 1 <= x <= 10000))
        if len(gs) == 20:
            initial_sets.append(gs)
    
    # Add smooth number sets
    if len(pool_2smooth) >= 20:
        initial_sets.append(pool_2smooth[:20])
    if len(pool_3smooth) >= 20:
        initial_sets.append(pool_3smooth[:20])
    if len(pool_4smooth) >= 20:
        initial_sets.append(pool_4smooth[:20])
    
    # Consecutive integers
    initial_sets.append(list(range(1, 21)))
    
    # Random samples from smooth pool
    for _ in range(10):
        if len(candidate_pool) >= 20:
            initial_sets.append(sorted(random.sample(candidate_pool, 20)))
    
    # Evaluate all initial candidates
    best_A = None
    best_score = float('inf')
    
    for init_set in initial_sets:
        if time.time() > deadline - 10.0:
            break
        
        init_set = sorted(set(x for x in init_set if 1 <= x <= 10000))
        if len(init_set) < 20:
            for x in candidate_pool:
                if x not in init_set and x <= 10000:
                    init_set.append(x)
                    if len(init_set) == 20:
                        break
        init_set = sorted(init_set[:20])
        
        if len(init_set) != 20:
            continue
        
        score = _calc_score(init_set)
        if score < best_score:
            best_score = score
            best_A = list(init_set)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = _calc_score(best_A)
    
    # Phase 1: Geometric refinement on best result
    if time.time() < deadline - 25.0:
        remaining = min(20.0, deadline - time.time() - 15.0)
        if remaining > 2.0:
            best_A, best_score = _geometric_refinement(best_A, timeout=remaining)
    
    # Phase 2: Local search with smooth pool
    if time.time() < deadline - 20.0:
        remaining = min(20.0, deadline - time.time() - 10.0)
        if remaining > 2.0:
            best_A, best_score = _local_search(best_A, candidate_pool, timeout=remaining)
    
    # Phase 3: Fine polish
    if time.time() < deadline - 10.0:
        remaining = min(10.0, deadline - time.time() - 5.0)
        if remaining > 1.0:
            best_A, best_score = _fine_polish(best_A, timeout=remaining)
    
    # Phase 4: Multi-restart with geometric seeds
    if time.time() < deadline - 15.0:
        for restart in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Pick a random geometric set
            if geo_sets:
                A_rand = list(random.choice(geo_sets))
            else:
                A_rand = sorted(random.sample(candidate_pool, 20))
            
            A_rand = sorted(set(x for x in A_rand if 1 <= x <= 10000))
            if len(A_rand) < 20:
                for x in candidate_pool:
                    if x not in A_rand and x <= 10000:
                        A_rand.append(x)
                        if len(A_rand) == 20:
                            break
            A_rand = sorted(A_rand[:20])
            
            if len(A_rand) != 20:
                continue
            
            score_rand = _calc_score(A_rand)
            
            if score_rand < best_score:
                best_score = score_rand
                best_A = list(A_rand)
                
                # Quick refinement
                if time.time() < deadline - 10.0:
                    remaining = min(8.0, deadline - time.time() - 5.0)
                    if remaining > 1.0:
                        best_A, best_score = _geometric_refinement(best_A, timeout=remaining)
    
    # Phase 5: Final polish
    if best_A is not None and time.time() < deadline - 5.0:
        remaining = min(5.0, deadline - time.time() - 2.0)
        if remaining > 1.0:
            best_A, best_score = _fine_polish(best_A, timeout=remaining)
    
    # Fallback
    if best_A is None or len(best_A) != 20:
        best_A = list(range(1, 21))
    
    return best_A
