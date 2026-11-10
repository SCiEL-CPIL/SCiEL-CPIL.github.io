import time
import random
import sys

# EVOLVE-BLOCK-START

def _score_fast(A):
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

def _score_with_bound(A, current_best):
    """Return max(|A+A|, |A*A|) but stop early if either set exceeds current_best."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            ss.add(s)
            if len(ss) >= current_best:
                return current_best
            p = ai * A[j]
            ps.add(p)
            if len(ps) >= current_best:
                return current_best
    return max(len(ss), len(ps))

def _generate_smooth_numbers(limit):
    """Generate 5-smooth numbers up to limit."""
    smooth = set()
    for i in range(1, limit + 1):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth.add(i)
    return sorted(smooth)

def _try_geometric_base(b, n=20, maximum=10000):
    """Try a geometric progression with base b, then refine with multiplicative perturbations."""
    # Generate geometric sequence: b^0, b^1, b^2, ...
    # But we need exactly n distinct positive integers <= maximum
    # For base b, the sequence is b^0=1, b^1=b, b^2=b^2, ...
    # We need b^(n-1) <= maximum
    
    if b <= 1:
        return None
    
    # Check if b^(n-1) <= maximum
    val = 1
    for _ in range(n-1):
        val *= b
        if val > maximum:
            return None
    
    # Generate the geometric sequence
    A = [1]
    for i in range(1, n):
        A.append(A[-1] * b)
    
    if max(A) > maximum or len(set(A)) != n:
        return None
    
    score = _score_fast(A)
    
    # Now try to refine by replacing some elements with nearby smooth numbers
    # to break the exact geometric structure and reduce sum/product set sizes
    smooth_nums = _generate_smooth_numbers(maximum)
    smooth_set = set(smooth_nums)
    
    # For each position, try swapping with nearby smooth numbers
    improved = True
    while improved:
        improved = False
        for i in range(n):
            old_val = A[i]
            A_set = set(A)
            A_set.discard(old_val)
            
            # Try nearby smooth numbers (within a factor of 3)
            candidates = [s for s in smooth_nums if old_val // 4 <= s <= old_val * 4]
            
            for new_val in candidates:
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, score)
                if sc < score:
                    score = sc
                    improved = True
                    A.sort()
                    break
                A[i] = old_val
            
            if improved:
                break
    
    return A, score

def _local_search(A, best_score, pool, deadline):
    """Local search to improve A."""
    n = len(A)
    A = list(A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            old_val = A[i]
            A_set = set(A)
            A_set.discard(old_val)
            
            for new_val in pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    improved = True
                    A.sort()
                    break
                A[i] = old_val
            
            if improved:
                break
        
        if not improved:
            # Try random perturbations
            found_improvement = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = random.randint(0, n-1)
                old_val = A[i]
                new_val = random.choice(pool)
                if new_val == old_val or new_val in A:
                    continue
                
                A[i] = new_val
                sc = _score_with_bound(A, best_score)
                if sc < best_score:
                    best_score = sc
                    found_improvement = True
                    A.sort()
                    break
                A[i] = old_val
            
            if not found_improvement:
                break
    
    return A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 20
    maximum = 10000
    
    # Strategy: Try geometric progressions with various bases, then refine
    best_A = None
    best_score = float('inf')
    
    # Try different bases for geometric progressions
    # Base 2: 1,2,4,8,...,2^19 = 524288 (too big)
    # Base 2 with fewer terms won't work, we need exactly 20 terms
    # So base must be small enough that b^19 <= 10000
    # 10000^(1/19) ≈ 1.77, so base can only be 1 (invalid) or we need to be creative
    
    # Actually, for base b, we need b^19 <= 10000
    # b <= 10000^(1/19) ≈ 1.77
    # So only base b=1 works, but that gives all 1s (not distinct)
    
    # So pure geometric progressions with integer base won't work for n=20
    # We need a different approach: use a geometric-like structure with perturbations
    
    # Alternative: Use a multiplicative lattice structure
    # A = {2^a * 3^b * 5^c : (a,b,c) in S} for some set S of 20 exponents
    
    # Let's try to find 20 smooth numbers that form a "nice" multiplicative structure
    smooth_nums = _generate_smooth_numbers(maximum)
    
    # Start with the first 20 smooth numbers
    best_A = smooth_nums[:n]
    best_score = _score_fast(best_A)
    
    # Create a pool for local search: smooth numbers near the current set
    # Include smooth numbers from 1 to max(best_A)*2
    max_val = max(best_A)
    pool = [s for s in smooth_nums if s <= max_val * 3]
    pool.sort()
    
    # Local search
    best_A, best_score = _local_search(best_A, best_score, pool, deadline)
    
    # Try different initial sets: smooth numbers with different density
    # Try powers of 2 mixed with other smooth numbers
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Generate a random set of 20 smooth numbers
        random.shuffle(smooth_nums)
        candidate = smooth_nums[:n]
        candidate.sort()
        
        sc = _score_with_bound(candidate, best_score)
        if sc < best_score:
            best_score = sc
            best_A = candidate
        
        # Local search from this candidate
        pool = [s for s in smooth_nums if s <= max(candidate) * 3]
        pool.sort()
        best_A, best_score = _local_search(best_A, best_score, pool, deadline)
    
    # Final validation
    if len(best_A) != n or len(set(best_A)) != n or any(x <= 0 or x > maximum for x in best_A):
        best_A = list(range(1, n+1))
    
    return best_A

# EVOLVE-BLOCK-END
