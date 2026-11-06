import time
import random
from math import gcd
from itertools import combinations

def _is_valid_pairwise_coprime(sums):
    """Check if all distinct sums are pairwise coprime."""
    distinct = sorted(set(sums))
    for i in range(len(distinct)):
        for j in range(i+1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Evaluate and return (score, sums_list)."""
    sums = [a + b for a in A for b in B]
    if _is_valid_pairwise_coprime(sums):
        return len(set(sums)), sums
    return 0, sums

def _count_prime_factors(n):
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    def try_candidate(A, B):
        nonlocal best_score, best_A, best_B
        score, _ = _evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    # Strategy 1: Systematic search with fixed A, optimized B selection
    # Key insight: for fixed A = [a1, a2, a3], a candidate b is "good" if
    # {a1+b, a2+b, a3+b} are pairwise coprime. Then we need to pick 3 such b's
    # where ALL 9 sums are pairwise coprime.
    
    def search_with_fixed_A(A, deadline):
        """Given A, find B that maximizes pairwise coprime sumset."""
        a1, a2, a3 = A[0], A[1], A[2]
        
        # For a given b, check if the triple {a1+b, a2+b, a3+b} is pairwise coprime
        # Also precompute the prime factors of each sum for cross-checking
        
        good_b = []
        for b in range(1, 129):
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            
            if s1 < 2 or s3 > 256:
                continue
            
            if gcd(s1, s2) != 1 or gcd(s1, s3) != 1 or gcd(s2, s3) != 1:
                continue
            
            # Collect prime factors for cross-term checking
            pf = set()
            for s in (s1, s2, s3):
                n = s
                d = 2
                while d * d <= n:
                    if n % d == 0:
                        pf.add(d)
                        while n % d == 0:
                            n //= d
                    d += 1
                if n > 1:
                    pf.add(n)
            
            # Score: fewer total prime factors is better (less likely to conflict)
            total_pf = sum(1 for s in (s1, s2, s3) for _ in range(_count_prime_factors(s)))
            good_b.append((total_pf, b, pf, (s1, s2, s3)))
        
        if len(good_b) < 3:
            return 0, None
        
        # Sort by prime factor count (ascending)
        good_b.sort()
        
        # Try combinations of top candidates
        # Limit to top 80 for efficiency
        top_candidates = good_b[:80]
        
        best_local_score = 0
        best_local_B = None
        
        for i in range(len(top_candidates)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(top_candidates)):
                for k in range(j+1, len(top_candidates)):
                    b1 = top_candidates[i][1]
                    b2 = top_candidates[j][1]
                    b3 = top_candidates[k][1]
                    
                    B = [b1, b2, b3]
                    sums = set(a + b for a in A for b in B)
                    
                    if len(sums) <= best_local_score:
                        continue
                    
                    if _is_valid_pairwise_coprime(list(sums)):
                        best_local_score = len(sums)
                        best_local_B = B
        
        return best_local_score, best_local_B
    
    # Generate diverse A candidates
    A_candidates = []
    
    # Small consecutive/near-consecutive sets
    for start in range(1, 30):
        for gap2 in range(1, 8):
            for gap3 in range(gap2+1, 12):
                a1 = start
                a2 = start + gap2
                a3 = start + gap2 + gap3
                if a3 <= 50:
                    A_candidates.append([a1, a2, a3])
    
    # Spread out sets
    for a1 in range(1, 15):
        for a2 in range(a1+2, 35):
            for a3 in range(a2+2, 60):
                A_candidates.append([a1, a2, a3])
    
    # Remove duplicates
    A_candidates = list(set(tuple(A) for A in A_candidates))
    A_candidates = [list(A) for A in A_candidates]
    
    # Sort by sum of elements (smaller tends to give more room for B)
    A_candidates.sort(key=lambda x: sum(x))
    
    # Search over A candidates
    for A in A_candidates:
        if time.time() > deadline:
            break
        
        score, B = search_with_fixed_A(A, deadline)
        if score > best_score and B is not None:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    # Strategy 2: Random search with more diverse A values
    random.seed(12345)
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Random A with varied spread
        a1 = random.randint(1, 40)
        a2 = random.randint(a1+1, min(a1+30, 80))
        a3 = random.randint(a2+1, min(a2+30, 100))
        A = sorted([a1, a2, a3])
        
        score, B = search_with_fixed_A(A, deadline)
        if score > best_score and B is not None:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    # Strategy 3: Local search / hill climbing on the best found
    if best_score >= 5:
        def local_search(A, B, max_iters=50000):
            nonlocal best_score, best_A, best_B
            current_A = A[:]
            current_B = B[:]
            current_score, _ = _evaluate(current_A, current_B)
            
            for iteration in range(max_iters):
                if time.time() > deadline:
                    break
                
                improved = False
                # Try changing one element in A or B
                for set_idx in range(2):
                    for pos in range(3):
                        old_val = current_A[pos] if set_idx == 0 else current_B[pos]
                        
                        # Try nearby values
                        candidates = set()
                        for delta in range(-8, 9):
                            new_val = old_val + delta
                            if 1 <= new_val <= 128 and new_val != old_val:
                                candidates.add(new_val)
                        # Add random candidates
                        for _ in range(15):
                            candidates.add(random.randint(1, 128))
                        
                        for new_val in sorted(candidates):
                            if set_idx == 0:
                                if new_val in current_A:
                                    continue
                                new_A = current_A[:]
                                new_A[pos] = new_val
                                new_B = current_B
                            else:
                                if new_val in current_B:
                                    continue
                                new_A = current_A
                                new_B = current_B[:]
                                new_B[pos] = new_val
                            
                            score, _ = _evaluate(new_A, new_B)
                            if score > current_score:
                                current_A = new_A
                                current_B = new_B
                                current_score = score
                                if score > best_score:
                                    best_score = score
                                    best_A = current_A[:]
                                    best_B = current_B[:]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            return current_score
        
        local_search(best_A, best_B)
    
    # Strategy 4: Targeted search for specific high-cardinality patterns
    # Try to find 9 distinct pairwise coprime sums
    # This requires all 9 sums to be distinct and pairwise coprime
    # A good pattern: A = [1, p, q] where p, q are chosen so that
    # differences p-1, q-1, q-p have few prime factors
    
    for a1 in range(1, 20):
        if time.time() > deadline:
            break
        for a2 in range(a1+1, 40):
            if time.time() > deadline:
                break
            for a3 in range(a2+1, 60):
                if time.time() > deadline:
                    break
                
                A = [a1, a2, a3]
                
                # Quick check: differences should have few prime factors
                d12 = a2 - a1
                d13 = a3 - a1
                d23 = a3 - a2
                
                if _count_prime_factors(d12) > 2 or _count_prime_factors(d13) > 2 or _count_prime_factors(d23) > 2:
                    continue
                
                score, B = search_with_fixed_A(A, deadline)
                if score > best_score and B is not None:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
    
    # Final verification
    final_score, _ = _evaluate(best_A, best_B)
    if final_score < 5:
        # Fallback to a known valid solution
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]
