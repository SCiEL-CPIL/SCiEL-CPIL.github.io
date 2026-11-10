"""Search-based program: generate varied candidate (A,B) pairs and keep the best valid one."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin under 90s wall-clock

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    # Initial best from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    if best_score < 0:
        best_score = 0

    # Build a pool of candidate numbers
    def is_prime(n):
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

    # Numbers that are "good" for making coprime sums:
    # - 1, primes, and numbers with few small prime factors
    # - Avoid numbers sharing common small prime factors with many others
    good_numbers = []
    for n in range(1, 129):
        # Count distinct prime factors
        pf = set()
        m = n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if p * p > m:
                break
            if m % p == 0:
                pf.add(p)
                while m % p == 0:
                    m //= p
        if m > 1:
            pf.add(m)
        # Prefer numbers with 1 or 2 distinct prime factors
        if len(pf) <= 2:
            good_numbers.append(n)
    
    pool = good_numbers
    random.shuffle(pool)

    # Try a broader set of structured candidates
    structured = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 3, 5], [2, 4, 6]),
        ([1, 7, 13], [4, 10, 16]),
        ([2, 3, 5], [7, 11, 13]),
        ([1, 2, 4], [3, 5, 7]),
        ([3, 5, 7], [2, 4, 6]),
        ([1, 4, 9], [2, 3, 5]),
        ([2, 5, 11], [3, 7, 13]),
        ([1, 3, 7], [2, 5, 11]),
        ([4, 9, 16], [2, 3, 5]),
        ([1, 2, 5], [3, 7, 11]),
        ([1, 3, 8], [2, 5, 10]),
        ([2, 4, 9], [1, 5, 7]),
        ([3, 5, 10], [1, 4, 8]),
        ([1, 6, 11], [2, 5, 9]),
        ([2, 7, 12], [1, 4, 8]),
        ([1, 4, 10], [3, 6, 11]),
        ([2, 5, 14], [1, 7, 12]),
        ([1, 8, 15], [2, 5, 11]),
        ([3, 6, 13], [1, 4, 10]),
        ([1, 9, 14], [2, 5, 11]),
        ([4, 7, 12], [1, 5, 9]),
        ([2, 8, 15], [1, 5, 11]),
        ([1, 10, 17], [2, 5, 11]),
        ([3, 7, 14], [1, 4, 10]),
        ([1, 11, 19], [2, 5, 13]),
        ([2, 9, 16], [1, 5, 11]),
        ([4, 8, 15], [1, 5, 11]),
        ([1, 12, 20], [2, 5, 13]),
        ([3, 8, 17], [1, 4, 10]),
        # Additional structured candidates
        ([1, 5, 11], [2, 7, 13]),
        ([2, 6, 11], [1, 5, 9]),
        ([1, 4, 8], [3, 7, 11]),
        ([2, 5, 9], [1, 4, 8]),
        ([1, 6, 10], [3, 7, 11]),
        ([3, 5, 9], [1, 4, 8]),
        ([1, 7, 11], [2, 5, 9]),
        ([2, 6, 10], [1, 5, 9]),
        ([1, 8, 12], [3, 5, 9]),
        ([2, 7, 11], [1, 4, 8]),
        ([1, 9, 13], [2, 5, 10]),
        ([3, 6, 10], [1, 4, 8]),
        ([1, 10, 14], [2, 5, 9]),
        ([2, 8, 12], [1, 5, 9]),
        ([1, 11, 15], [2, 4, 8]),
        ([3, 7, 11], [1, 4, 9]),
        ([1, 12, 16], [2, 5, 10]),
        ([2, 9, 13], [1, 4, 8]),
        ([1, 13, 17], [2, 5, 11]),
        ([3, 8, 12], [1, 4, 9]),
        # More diverse candidates
        ([1, 2, 5], [4, 7, 11]),
        ([1, 3, 4], [5, 8, 12]),
        ([2, 3, 7], [4, 5, 9]),
        ([1, 5, 6], [2, 7, 11]),
        ([3, 4, 7], [1, 5, 9]),
        ([1, 6, 7], [2, 4, 10]),
        ([2, 5, 6], [1, 7, 11]),
        ([1, 4, 5], [3, 8, 12]),
        ([2, 3, 8], [1, 5, 9]),
        ([1, 7, 8], [2, 5, 11]),
        ([3, 5, 6], [1, 4, 8]),
        ([1, 9, 10], [2, 5, 11]),
        ([2, 7, 8], [1, 5, 9]),
        ([1, 6, 9], [3, 5, 11]),
        ([4, 5, 7], [1, 3, 8]),
        ([2, 6, 7], [1, 4, 9]),
        ([1, 8, 9], [2, 5, 11]),
        ([3, 7, 8], [1, 4, 10]),
        ([1, 10, 11], [2, 5, 9]),
        ([2, 9, 10], [1, 5, 8]),
    ]
    for A, B in structured:
        if time.time() > deadline:
            break
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]

    # Smart Search: Greedy construction with backtracking
    # We want 9 distinct pairwise coprime sums.
    # Strategy: Pick A, then pick B elements one by one, ensuring new sums are coprime to existing ones.
    
    def try_build(A_fixed):
        # A_fixed is a list of 3 ints
        # Try to find B
        used_sums = set()
        B = []
        # Sort pool by "goodness" (primes first, then few factors)
        # Pre-computed in pool? No, just use primes and numbers with large factors
        candidates_B = [n for n in range(1, 129) if n not in A_fixed]
        # Heuristic: prefer primes
        candidates_B.sort(key=lambda x: (0 if is_prime(x) else 1, x))
        
        for b in candidates_B:
            if len(B) == 3:
                break
            new_sums = [a + b for a in A_fixed]
            # Check distinctness within new_sums
            if len(set(new_sums)) != 3:
                continue
            # Check distinctness against used_sums
            if any(s in used_sums for s in new_sums):
                continue
            # Check pairwise coprime with used_sums
            valid = True
            for s in new_sums:
                for us in used_sums:
                    if gcd(s, us) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                B.append(b)
                used_sums.update(new_sums)
        
        if len(B) == 3:
            return B
        return None

    # Try a set of A candidates
    # A should also be "good"
    A_candidates = [n for n in range(1, 129) if is_prime(n) or len([p for p in [2,3,5,7,11,13] if n%p==0]) <= 1]
    A_candidates.sort(key=lambda x: (0 if is_prime(x) else 1, x))
    
    # Sample a few A sets
    import itertools
    A_samples = list(itertools.combinations(A_candidates[:50], 3))
    random.shuffle(A_samples)
    
    for A in A_samples:
        if time.time() > deadline - 2:
            break
        B = try_build(A)
        if B:
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
                if best_score >= 9:
                    break

    # Smart local refinement: prefer candidates that minimize gcd conflicts
    if best_score < 9 and time.time() < deadline - 5:
        # Pass 1: single-element swaps with constraint-aware scoring
        for _ in range(20000):
            if time.time() > deadline - 3:
                break
            A = best_A[:]
            B = best_B[:]
            if random.random() < 0.5:
                idx = random.randrange(3)
                A[idx] = random.choice(pool)
            else:
                idx = random.randrange(3)
                B[idx] = random.choice(pool)
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                continue
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
                if best_score >= 9:
                    break

        # Pass 2: two-element swaps in one set
        if best_score < 9 and time.time() < deadline - 3:
            for _ in range(15000):
                if time.time() > deadline - 2:
                    break
                A = best_A[:]
                B = best_B[:]
                if random.random() < 0.5:
                    idx1, idx2 = random.sample(range(3), 2)
                    A[idx1] = random.choice(pool)
                    A[idx2] = random.choice(pool)
                else:
                    idx1, idx2 = random.sample(range(3), 2)
                    B[idx1] = random.choice(pool)
                    B[idx2] = random.choice(pool)
                if len(set(A)) != 3 or len(set(B)) != 3:
                    continue
                if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                    continue
                s = score_of(A, B)
                if s > best_score:
                    best_score = s
                    best_A, best_B = A[:], B[:]
                    if best_score >= 9:
                        break

        # Pass 3: simultaneous swap in A and B
        if best_score < 9 and time.time() < deadline - 2:
            for _ in range(10000):
                if time.time() > deadline - 1:
                    break
                A = best_A[:]
                B = best_B[:]
                idxA = random.randrange(3)
                idxB = random.randrange(3)
                A[idxA] = random.choice(pool)
                B[idxB] = random.choice(pool)
                if len(set(A)) != 3 or len(set(B)) != 3:
                    continue
                if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                    continue
                s = score_of(A, B)
                if s > best_score:
                    best_score = s
                    best_A, best_B = A[:], B[:]
                    if best_score >= 9:
                        break

    # Final polish: try to fix specific gcd conflicts
    if best_score < 9 and time.time() < deadline - 1:
        # Identify which sums have gcd conflicts and try to fix them
        sums = distinct_sums(best_A, best_B)
        sum_list = list(sums)
        conflicts = []
        for i in range(len(sum_list)):
            for j in range(i + 1, len(sum_list)):
                if gcd(sum_list[i], sum_list[j]) != 1:
                    conflicts.append((sum_list[i], sum_list[j]))
        
        # Try to resolve each conflict by changing one element
        for s1, s2 in conflicts[:5]:
            if time.time() > deadline - 1:
                break
            # Find which (a,b) pairs produce s1 and s2
            pairs1 = [(a, b) for a in best_A for b in best_B if a + b == s1]
            pairs2 = [(a, b) for a in best_A for b in best_B if a + b == s2]
            
            for a1, b1 in pairs1:
                for a2, b2 in pairs2:
                    # Try changing a1 to something else
                    for new_a in pool:
                        if new_a == a1 or new_a in best_A:
                            continue
                        A_new = best_A[:]
                        A_new[best_A.index(a1)] = new_a
                        if len(set(A_new)) != 3:
                            continue
                        s = score_of(A_new, best_B)
                        if s > best_score:
                            best_score = s
                            best_A = A_new
                            break
                    # Try changing b1 to something else
                    for new_b in pool:
                        if new_b == b1 or new_b in best_B:
                            continue
                        B_new = best_B[:]
                        B_new[best_B.index(b1)] = new_b
                        if len(set(B_new)) != 3:
                            continue
                        s = score_of(best_A, B_new)
                        if s > best_score:
                            best_score = s
                            best_B = B_new
                            break
                    if best_score >= 9:
                        break
                if best_score >= 9:
                    break
            if best_score >= 9:
                break

    return [best_A, best_B]
# EVOLVE-BLOCK-END
