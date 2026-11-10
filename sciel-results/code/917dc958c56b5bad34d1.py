import time
import math
import random

def calc_score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
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

def calc_score_fast(A):
    """Faster score calculation using numpy-style loops (pure Python but optimized)."""
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sumset.add(ai + A[j])
            prodset.add(ai * A[j])
    return max(len(sumset), len(prodset))

def generate_smooth_pool(max_val=10000, primes=(2, 3, 5)):
    """Generate all p-smooth numbers up to max_val."""
    pool = set()
    n_primes = len(primes)
    exponents = [0] * n_primes
    max_exps = []
    for p in primes:
        e = 0
        v = 1
        while v <= max_val:
            v *= p
            e += 1
        max_exps.append(e)
    
    def gen(idx, current):
        if idx == n_primes:
            if 1 <= current <= max_val:
                pool.add(current)
            return
        p = primes[idx]
        v = 1
        while v <= max_val // max(1, current if current > 1 else 1):
            gen(idx + 1, current * v)
            v *= p
            if v > max_val:
                break
        # Also include current without multiplying by p
        # Actually, let me redo this more carefully
        pass
    
    # Simpler recursive approach
    def gen2(idx, current):
        if idx == n_primes:
            if 1 <= current <= max_val:
                pool.add(current)
            return
        p = primes[idx]
        v = 1
        while v <= max_val and current * v <= max_val:
            gen2(idx + 1, current * v)
            v *= p
        gen2(idx + 1, current)
    
    gen2(0, 1)
    return sorted(pool)

def make_polynomial_set(coeffs, n=20):
    """Evaluate polynomial p(t) at t=0,1,...,n-1. coeffs are [c0, c1, c2, c3]."""
    A = []
    for t in range(n):
        val = coeffs[0] + coeffs[1]*t + coeffs[2]*t*t + coeffs[3]*t*t*t
        A.append(int(round(val)))
    return A

def local_search_poly(coeffs, time_limit, rng, pool):
    """Local search on polynomial coefficients."""
    best_A = make_polynomial_set(coeffs)
    best_score = calc_score(best_A)
    start = time.time()
    current_coeffs = list(coeffs)
    current_A = make_polynomial_set(current_coeffs)
    current_score = best_score
    
    # Also keep a fallback: best non-polynomial set found via element replacement
    fallback_A = list(current_A)
    fallback_score = current_score
    
    step_sizes = [1, 2, 3, 5, 8, 13]
    
    while time.time() - start < time_limit:
        improved = False
        # Try perturbing each coefficient
        for ci in range(len(current_coeffs)):
            for step in step_sizes:
                for delta in [-step, step]:
                    new_coeffs = list(current_coeffs)
                    new_coeffs[ci] += delta
                    new_A = make_polynomial_set(new_coeffs)
                    
                    # Validate
                    if len(set(new_A)) != 20:
                        continue
                    if any(x < 1 or x > 10000 for x in new_A):
                        continue
                    
                    score = calc_score(new_A)
                    if score < current_score:
                        current_score = score
                        current_coeffs = new_coeffs
                        current_A = new_A
                        if score < best_score:
                            best_score = score
                            best_A = list(new_A)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Also try element replacement on the best polynomial set found
        if not improved:
            A_set = set(current_A)
            for i in range(20):
                old = current_A[i]
                # Try a random few candidates
                for _ in range(5):
                    new_val = rng.choice(pool)
                    if new_val in A_set or new_val == old:
                        continue
                    new_A = list(current_A)
                    new_A[i] = new_val
                    if len(set(new_A)) != 20:
                        continue
                    if any(x < 1 or x > 10000 for x in new_A):
                        continue
                    score = calc_score(new_A)
                    if score < current_score:
                        current_score = score
                        current_A = new_A
                        if score < best_score:
                            best_score = score
                            best_A = list(new_A)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved and time.time() - start > time_limit * 0.7:
            # No progress for a while, try a random restart with different initial coeffs
            if rng.random() < 0.1:
                new_coeffs = [rng.randint(1, 10), rng.randint(1, 5), rng.randint(0, 2), rng.randint(0, 1)]
                new_A = make_polynomial_set(new_coeffs)
                if len(set(new_A)) == 20 and all(1 <= x <= 10000 for x in new_A):
                    current_coeffs = new_coeffs
                    current_A = new_A
                    current_score = calc_score(new_A)
    
    return best_A, best_score

def local_search_elements(A, time_limit, rng, pool):
    """Local search by replacing elements, with collision-seeking heuristic."""
    A = list(A)
    A.sort()
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start = time.time()
    
    # Build a frequency map of products to identify collisions
    def get_product_freq(arr):
        freq = {}
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                p = arr[i] * arr[j]
                freq[p] = freq.get(p, 0) + 1
        return freq
    
    while time.time() - start < time_limit:
        improved = False
        A_set = set(A)
        
        # Identify the most "wasteful" element: one that creates unique products
        # and try to replace it with something that creates collisions
        for i in range(len(A)):
            if time.time() - start > time_limit:
                break
            old_val = A[i]
            
            # Generate candidate replacements: smooth numbers that share factors with other elements
            candidates = []
            for other in A:
                if other == old_val:
                    continue
                # Multiples and divisors of other
                for mult in [1, 2, 3, 4, 5]:
                    v = other * mult
                    if 1 <= v <= 10000 and v not in A_set and v != old_val:
                        candidates.append(v)
                    if other % mult == 0:
                        v = other // mult
                        if 1 <= v <= 10000 and v not in A_set and v != old_val:
                            candidates.append(v)
            
            # Also try random smooth numbers
            candidates.extend(rng.sample(pool[:100], min(20, len(pool[:100]))))
            
            candidates = list(set(candidates))
            rng.shuffle(candidates)
            
            for new_val in candidates[:30]:
                if new_val in A_set or new_val == old_val:
                    continue
                if not (1 <= new_val <= 10000):
                    continue
                new_A = A.copy()
                new_A[i] = new_val
                score = calc_score(new_A)
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break
            
            if improved:
                # Refresh A_set
                A_set = set(A)
        
        if not improved:
            # Try random perturbations
            i = rng.randint(0, 19)
            old_val = A[i]
            for _ in range(10):
                new_val = rng.choice(pool)
                if new_val in A_set or new_val == old_val:
                    continue
                new_A = A.copy()
                new_A[i] = new_val
                if len(set(new_A)) != 20:
                    continue
                if not all(1 <= x <= 10000 for x in new_A):
                    continue
                score = calc_score(new_A)
                if score < current_score:
                    current_score = score
                    A = new_A
                    A_set = set(A)
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    rng = random.Random(42)
    
    # Generate smooth number pool
    pool = generate_smooth_pool(10000, (2, 3, 5))
    
    # Candidate initial sets
    candidates = []
    
    # 1. Arithmetic progressions (guarantee |A+A| = 39)
    for a0 in range(1, 6):
        for d in range(1, 6):
            A = [a0 + d*i for i in range(20)]
            if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
                candidates.append(A)
    
    # 2. Polynomial-based: p(t) = c0 + c1*t + c2*t^2 + c3*t^3
    # Quadratic APs: a + d*t + e*t^2
    for a0 in range(1, 6):
        for d in range(1, 4):
            for e in range(0, 3):
                coeffs = [a0, d, e, 0]
                A = make_polynomial_set(coeffs)
                if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
                    candidates.append(A)
    
    # 3. Small integers 1..20
    candidates.append(list(range(1, 21)))
    
    # 4. Small smooth numbers
    smooth_small = [x for x in pool if x <= 60]
    if len(smooth_small) >= 20:
        candidates.append(sorted(smooth_small[:20]))
    
    # 5. Geometric-like sequences with integer rounding
    for base in [2, 3]:
        A = []
        for i in range(20):
            val = int(round(base ** (i * 0.3)))
            if val < 1:
                val = 1
            A.append(val)
        # Ensure distinct
        seen = set()
        unique_A = []
        for v in A:
            while v in seen and v < 10000:
                v += 1
            seen.add(v)
            unique_A.append(v)
        if len(set(unique_A)) == 20 and all(1 <= x <= 10000 for x in unique_A):
            candidates.append(unique_A)
    
    # Deduplicate candidates
    seen_configs = set()
    unique_candidates = []
    for A in candidates:
        key = tuple(sorted(A))
        if key not in seen_configs:
            seen_configs.add(key)
            unique_candidates.append(A)
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Quick evaluation of all candidates
    for A in unique_candidates:
        if time.time() > deadline:
            break
        score = calc_score(A)
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    # Phase 2: Local search on top candidates
    # Sort by score, take top 5
    scored = []
    for A in unique_candidates:
        if time.time() > deadline:
            break
        scored.append((calc_score(A), A))
    scored.sort(key=lambda x: x[0])
    
    top_candidates = [A for _, A in scored[:min(5, len(scored))]]
    
    num_top = len(top_candidates)
    for idx, A in enumerate(top_candidates):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        alloc = max(1.0, remaining / (num_top - idx))
        
        # Element replacement local search
        improved_A, score = local_search_elements(A, alloc * 0.6, rng, pool)
        if score < best_score:
            best_score = score
            best_A = improved_A
        
        if time.time() > deadline:
            break
        
        # Polynomial coefficient search if A came from a polynomial
        remaining = deadline - time.time()
        if remaining > 2.0:
            # Try to fit a polynomial to the best A found so far
            # For simplicity, try a few random polynomial initializations
            for _ in range(3):
                if time.time() > deadline:
                    break
                coeffs = [rng.randint(1, 10), rng.randint(1, 5), rng.randint(0, 3), rng.randint(0, 1)]
                poly_A = make_polynomial_set(coeffs)
                if len(set(poly_A)) == 20 and all(1 <= x <= 10000 for x in poly_A):
                    poly_score = calc_score(poly_A)
                    if poly_score < best_score:
                        # Refine this polynomial
                        poly_A2, poly_score2 = local_search_poly(coeffs, min(3.0, deadline - time.time()), rng, pool)
                        if poly_score2 < best_score:
                            best_score = poly_score2
                            best_A = poly_A2
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
