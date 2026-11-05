"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start_time = time.time()
    # 90s limit, use 85s as deadline to be safe
    deadline = start_time + 85.0

    def pairwise_coprime(vals):
        """Check if all distinct values are pairwise coprime."""
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        """Compute the number of distinct sums that are pairwise coprime."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if not pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    # Initial best solution (the seed, known to be valid with score 5)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_pair(best_A, best_B) # Should be 5
    
    # Try some known good starting points to get a better initial best
    # These are hand-picked or from previous searches
    test_solutions = [
        ([1, 3, 5], [2, 4, 6]),
        ([1, 3, 5], [2, 8, 14]),
        ([1, 3, 7], [2, 4, 10]),
        ([1, 5, 9], [2, 6, 10]),
        ([3, 5, 7], [2, 4, 10]),
        ([1, 7, 11], [2, 6, 10]),
        ([1, 5, 11], [2, 4, 8]),
        ([3, 7, 11], [2, 4, 10]),
    ]
    for A_t, B_t in test_solutions:
        s_t = score_pair(A_t, B_t)
        if s_t > best_score:
            best_score = s_t
            best_A = A_t[:]
            best_B = B_t[:]
            if best_score == 9:
                return [best_A, best_B]

    # Precompute primes up to 256 (max sum 128+128)
    max_sum = 256
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_sum)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_sum + 1, i):
                is_prime[j] = False

    # Precompute gcd for all pairs up to 256
    # This will be used for fast pairwise coprimality checks
    # But we'll use math.gcd directly for simplicity

    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))

    def search_case(A_source, B_source):
        nonlocal best_A, best_B, best_score
        
        # Phase 1: Small values (1-60) - expanded for better coverage
        limit1 = 60
        A_cands = [x for x in A_source if x <= limit1]
        B_cands = [x for x in B_source if x <= limit1]
        
        # Precompute compatibility: for each b, which a's make a+b prime?
        # Actually, let's precompute for each (a,b) pair whether a+b is prime
        # And also check pairwise coprimality of the 9 sums
        
        for i1 in range(len(A_cands)):
            if time.time() > deadline: return
            a1 = A_cands[i1]
            for i2 in range(i1 + 1, len(A_cands)):
                a2 = A_cands[i2]
                for i3 in range(i2 + 1, len(A_cands)):
                    a3 = A_cands[i3]
                    A = [a1, a2, a3]
                    
                    # Find compatible b's: a1+b, a2+b, a3+b should be "good"
                    # First try: all three are prime
                    # Second try: all three are pairwise coprime with each other
                    # and also coprime with other sums
                    
                    # Quick filter: for each b, check if a1+b, a2+b, a3+b are pairwise coprime
                    compat_b_prime = []
                    compat_b_coprime = []
                    for b in B_cands:
                        s1, s2, s3 = a1 + b, a2 + b, a3 + b
                        # Check pairwise coprime among these three
                        if math.gcd(s1, s2) == 1 and math.gcd(s1, s3) == 1 and math.gcd(s2, s3) == 1:
                            if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                                compat_b_prime.append(b)
                            else:
                                compat_b_coprime.append(b)
                    
                    # Try prime-compatible b's first (higher chance of all 9 being coprime)
                    for b_list in [compat_b_prime, compat_b_coprime]:
                        if len(b_list) >= 3:
                            # Try different combinations of 3 b's
                            # First try the first 3
                            B = b_list[:3]
                            s = score_pair(A, B)
                            if s > best_score:
                                best_score = s
                                best_A = A[:]
                                best_B = B[:]
                                if best_score == 9:
                                    return
                            
                            # Try more diverse combinations if time permits
                            if len(b_list) > 3 and time.time() < deadline - 10:
                                # Try consecutive windows and also spread-out selections
                                tried = set()
                                # Consecutive windows
                                for k in range(min(10, len(b_list) - 2)):
                                    B2 = b_list[k:k+3]
                                    key = tuple(sorted(B2))
                                    if key not in tried:
                                        tried.add(key)
                                        s2 = score_pair(A, B2)
                                        if s2 > best_score:
                                            best_score = s2
                                            best_A = A[:]
                                            best_B = B2[:]
                                            if best_score == 9:
                                                return
                                # Spread out selections (pick from beginning, middle, end)
                                n = len(b_list)
                                if n >= 5:
                                    B_spread = [b_list[0], b_list[n//2], b_list[n-1]]
                                    key = tuple(sorted(B_spread))
                                    if key not in tried:
                                        tried.add(key)
                                        s2 = score_pair(A, B_spread)
                                        if s2 > best_score:
                                            best_score = s2
                                            best_A = A[:]
                                            best_B = B_spread[:]
                                            if best_score == 9:
                                                return
                                if time.time() > deadline - 5:
                                    return

        # Phase 2: If we haven't found score 9, expand to larger values
        if best_score < 9 and time.time() < deadline - 15:
            limit2 = 100
            A_cands2 = [x for x in A_source if x <= limit2 and x > limit1]
            # Also include some smaller ones for mixing
            A_cands2 = A_cands + A_cands2
            B_cands2 = [x for x in B_source if x <= limit2]
            
            for i1 in range(len(A_cands2)):
                if time.time() > deadline: return
                a1 = A_cands2[i1]
                for i2 in range(i1 + 1, len(A_cands2)):
                    if time.time() > deadline: return
                    a2 = A_cands2[i2]
                    for i3 in range(i2 + 1, len(A_cands2)):
                        if time.time() > deadline: return
                        a3 = A_cands2[i3]
                        A = [a1, a2, a3]
                        
                        # Quick check: are a1, a2, a3 such that their differences are small?
                        # This helps ensure sums don't collide
                        diffs = [a2 - a1, a3 - a2, a3 - a1]
                        if any(d == 0 for d in diffs):
                            continue
                        
                        # Find b's where a1+b, a2+b, a3+b are pairwise coprime
                        compat_b = []
                        for b in B_cands2:
                            s1, s2, s3 = a1 + b, a2 + b, a3 + b
                            if math.gcd(s1, s2) == 1 and math.gcd(s1, s3) == 1 and math.gcd(s2, s3) == 1:
                                compat_b.append(b)
                        
                        if len(compat_b) >= 3:
                            B = compat_b[:3]
                            s = score_pair(A, B)
                            if s > best_score:
                                best_score = s
                                best_A = A[:]
                                best_B = B[:]
                                if best_score == 9:
                                    return
                            
                            if len(compat_b) > 3 and time.time() < deadline - 5:
                                tried = set()
                                for k in range(min(8, len(compat_b) - 2)):
                                    B2 = compat_b[k:k+3]
                                    key = tuple(sorted(B2))
                                    if key not in tried:
                                        tried.add(key)
                                        s2 = score_pair(A, B2)
                                        if s2 > best_score:
                                            best_score = s2
                                            best_A = A[:]
                                            best_B = B2[:]
                                            if best_score == 9:
                                                return
                                n = len(compat_b)
                                if n >= 5:
                                    B_spread = [compat_b[0], compat_b[n//2], compat_b[n-1]]
                                    key = tuple(sorted(B_spread))
                                    if key not in tried:
                                        tried.add(key)
                                        s2 = score_pair(A, B_spread)
                                        if s2 > best_score:
                                            best_score = s2
                                            best_A = A[:]
                                            best_B = B_spread[:]
                                            if best_score == 9:
                                                return

    # Case 1: A odd, B even
    search_case(odd_vals, even_vals)
    
    # Case 2: A even, B odd
    if time.time() < deadline - 5:
        search_case(even_vals, odd_vals)

    return [best_A, best_B]

# EVOLVE-BLOCK-END
