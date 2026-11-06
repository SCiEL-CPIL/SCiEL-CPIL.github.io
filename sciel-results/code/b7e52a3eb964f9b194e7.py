"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
import random
from math import isqrt
import numpy as np

def _is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    r = isqrt(n)
    if r * r == n:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit."""
    powerful = set([1])
    max_b = int(limit ** (1/3)) + 2
    
    # Sieve for square-free
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    
    limit_sieve = int(max_b ** (1/2)) + 2
    for p in range(2, limit_sieve):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def _count_reps(N, powerful_list, powerful_set):
    """Count the number of pairs (a, b) with a <= b, a + b = N, a, b in powerful_set."""
    count = 0
    pairs = []
    for a in powerful_list:
        if 2 * a > N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer before 150s limit
    
    # Generate powerful numbers up to 10^9 for a comprehensive lookup set
    limit_pw = 10**9
    
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    n_pw = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > 40:
        # If generation took too long, reduce
        limit_pw = 10**8
        powerful = _generate_powerful(limit_pw)
        powerful_set = set(powerful)
        n_pw = len(powerful)
    
    # Strategy: Find N with many representations
    # Key insight: N that is a sum of many small powerful numbers will have many reps
    # We'll use a two-phase approach:
    # Phase 1: Sample candidate N values and estimate their representation counts
    # Phase 2: Verify top candidates with full enumeration
    
    # Phase 1: Generate candidate N values
    # Use small powerful numbers to create candidates likely to have many representations
    small_pw = powerful[:min(500, n_pw)]  # First 500 powerful numbers (smallest)
    
    # Generate candidate N values as sums of pairs of small powerful numbers
    candidates = set()
    for i in range(len(small_pw)):
        for j in range(i, len(small_pw)):
            s = small_pw[i] + small_pw[j]
            if s <= 10**18 and s >= 2:
                candidates.add(s)
    
    # Also add some random candidates from larger range
    random.seed(42)
    for _ in range(2000):
        idx1 = random.randint(0, min(2000, n_pw) - 1)
        idx2 = random.randint(0, min(2000, n_pw) - 1)
        s = powerful[idx1] + powerful[idx2]
        if s <= 10**18 and s >= 2:
            candidates.add(s)
    
    # Phase 2: Score candidates by estimating representation count
    # For each candidate, count how many powerful numbers a <= N/2 have N-a also powerful
    # Use a faster estimation: sample a subset of powerful numbers
    
    scored_candidates = []
    candidate_list = list(candidates)
    
    # Sort candidates and prioritize those in the "sweet spot" range
    # where many powerful numbers can pair up
    candidate_list.sort()
    
    # Focus on candidates between 1000 and 10^7 first (where density is highest)
    sweet_spot = [c for c in candidate_list if 1000 <= c <= 10**7]
    other_candidates = [c for c in candidate_list if not (1000 <= c <= 10**7)]
    
    # Process sweet spot first
    for N in sweet_spot:
        if time.time() > deadline - 30:
            break
        
        # Quick estimate: count representations using full scan
        count, _ = _count_reps(N, powerful, powerful_set)
        if count > 0:
            scored_candidates.append((count, N))
    
    # Process other candidates if time permits
    for N in other_candidates[:1000]:
        if time.time() > deadline - 20:
            break
        
        count, _ = _count_reps(N, powerful, powerful_set)
        if count > 0:
            scored_candidates.append((count, N))
    
    # Sort by count and verify top candidates
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_n = None
    best_count = 0
    best_pairs = []
    
    for count, N in scored_candidates[:50]:
        if time.time() > deadline - 10:
            break
        
        # Full verification
        actual_count, pairs = _count_reps(N, powerful, powerful_set)
        if actual_count > best_count:
            best_count = actual_count
            best_n = N
            best_pairs = pairs
    
    # If we found a good candidate, return it
    if best_n is not None and best_pairs:
        return {'N': best_n, 'pairs': best_pairs}
    
    # Fallback: try random search
    random.seed(123)
    for _ in range(10000):
        if time.time() > deadline - 5:
            break
        idx1 = random.randint(0, n_pw - 1)
        idx2 = random.randint(0, n_pw - 1)
        n_candidate = powerful[idx1] + powerful[idx2]
        
        if n_candidate > 10**18 or n_candidate < 2:
            continue
        
        count, pairs = _count_reps(n_candidate, powerful, powerful_set)
        
        if count > best_count:
            best_count = count
            best_n = n_candidate
            best_pairs = pairs
    
    if best_n is not None and best_pairs:
        return {'N': best_n, 'pairs': best_pairs}
    
    # Last resort: return a simple valid answer
    return {'N': 2, 'pairs': [[1, 1]]}
