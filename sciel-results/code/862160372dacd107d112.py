"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def _score(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True

def _poly_eval(coeffs, t):
    """Evaluate polynomial with coefficients [c0, c1, ..., cd] at t."""
    result = 0
    for c in coeffs:
        result = result * t + c
    return result

def _generate_poly_sets(max_val=10000, n=20):
    """Generate sets from low-degree polynomials evaluated at t=0..n-1."""
    candidates = []
    rng = random.Random(12345)
    
    # Try different polynomial degrees and coefficient ranges
    for degree in [1, 2, 3]:
        # For degree 1: arithmetic progressions a*t + b
        # For degree 2: quadratic sequences
        # For degree 3: cubic sequences
        
        if degree == 1:
            # Arithmetic progression: a*t + b, t=0..19
            # Need all values positive and <= max_val
            # a*t + b > 0 for t in [0,19], so b > 0
            # a*19 + b <= max_val
            for a in range(1, 300):
                for b in range(1, 500):
                    if a * 19 + b > max_val:
                        break
                    A = [a * t + b for t in range(n)]
                    if len(set(A)) == n and all(0 < x <= max_val for x in A):
                        candidates.append(sorted(A))
        elif degree == 2:
            # Quadratic: c2*t^2 + c1*t + c0
            # Try small coefficients
            for c2 in range(1, 50):
                for c1 in range(-20, 21):
                    for c0 in range(1, 200):
                        A = []
                        valid = True
                        for t in range(n):
                            val = c2 * t * t + c1 * t + c0
                            if val <= 0 or val > max_val:
                                valid = False
                                break
                            A.append(val)
                        if valid and len(set(A)) == n:
                            candidates.append(sorted(A))
        else:  # degree == 3
            # Cubic: small coefficients
            for c3 in range(1, 10):
                for c2 in range(-5, 6):
                    for c1 in range(-5, 6):
                        for c0 in range(1, 50):
                            A = []
                            valid = True
                            for t in range(n):
                                val = c3 * t**3 + c2 * t**2 + c1 * t + c0
                                if val <= 0 or val > max_val:
                                    valid = False
                                    break
                                A.append(val)
                            if valid and len(set(A)) == n:
                                candidates.append(sorted(A))
    
    return candidates

def _local_search(A, max_val=10000, n=20, deadline=None):
    """Local search to improve a starting set."""
    if deadline is None:
        deadline = time.time() + 60
    
    best_A = list(A)
    best_score = _score(best_A)
    A_set = set(best_A)
    rng = random.Random(42)
    
    # Build candidate pool
    pool = set()
    pool.update(best_A)
    
    # Add smooth numbers and related values
    for x in best_A:
        for d in range(-10, 11):
            if x + d > 0 and x + d <= max_val:
                pool.add(x + d)
        if x % 2 == 0:
            pool.add(x // 2)
        if x * 2 <= max_val:
            pool.add(x * 2)
        if x % 3 == 0:
            pool.add(x // 3)
        if x * 3 <= max_val:
            pool.add(x * 3)
        if x % 5 == 0:
            pool.add(x // 5)
        if x * 5 <= max_val:
            pool.add(x * 5)
    
    # Add some smooth numbers
    smooth = set()
    def gen_smooth(idx, current, primes=[2, 3, 5, 7]):
        if current > max_val:
            return
        if idx == len(primes):
            if current > 0:
                smooth.add(current)
            return
        p = primes[idx]
        val = current
        while val <= max_val:
            gen_smooth(idx + 1, val)
            if val > max_val // p:
                break
            val *= p
    gen_smooth(0, 1)
    pool.update(smooth)
    
    pool = sorted(pool)
    
    # Limit pool size
    if len(pool) > 3000:
        pool = pool[:1500] + pool[-1500:]
        pool = sorted(set(pool))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            old_val = best_A[i]
            for new_val in pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                best_A[i] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                
                sc = _score(best_A)
                
                if sc < best_score:
                    best_score = sc
                    improved = True
                    break
                else:
                    best_A[i] = old_val
                    A_set.discard(new_val)
                    A_set.add(old_val)
            
            if improved:
                break
        
        if not improved:
            # Random perturbations
            for _ in range(200):
                if time.time() > deadline:
                    break
                i = rng.randint(0, n - 1)
                old_val = best_A[i]
                new_val = rng.choice(pool)
                if new_val == old_val or new_val in A_set:
                    continue
                
                best_A[i] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                
                sc = _score(best_A)
                
                if sc < best_score:
                    best_score = sc
                    improved = True
                    break
                else:
                    best_A[i] = old_val
                    A_set.discard(new_val)
                    A_set.add(old_val)
            
            if not improved:
                break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 20
    maximum = 10000
    
    # Generate initial candidates from polynomial sequences
    poly_candidates = _generate_poly_sets(maximum, n)
    
    # Also include some simple structures
    simple_candidates = []
    
    # Arithmetic progressions with various starts and steps
    for a in [1, 2, 3, 5, 7, 10, 15, 20]:
        for b in range(1, 100):
            if a * 19 + b <= maximum:
                A = [a * t + b for t in range(n)]
                if len(set(A)) == n:
                    simple_candidates.append(sorted(A))
    
    # Geometric-like sequences
    for base in [2, 3]:
        A = [base**i for i in range(n)]
        if all(x <= maximum for x in A):
            simple_candidates.append(sorted(A))
    
    # Mix of small and large
    for _ in range(10):
        rng = random.Random(_)
        A = sorted(rng.sample(range(1, maximum + 1), n))
        simple_candidates.append(A)
    
    all_candidates = poly_candidates + simple_candidates
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    
    for A in all_candidates:
        if time.time() > deadline:
            break
        if _valid(A, n, maximum):
            sc = _score(A)
            if sc < best_score:
                best_score = sc
                best_A = list(A)
    
    if best_A is None:
        best_A = list(range(1, n + 1))
        best_score = _score(best_A)
    
    # Local search from best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_A, best_score = _local_search(best_A, maximum, n, deadline)
    
    # Final validation
    if not _valid(best_A, n, maximum):
        best_A = list(range(1, n + 1))
    
    return best_A
# EVOLVE-BLOCK-END
