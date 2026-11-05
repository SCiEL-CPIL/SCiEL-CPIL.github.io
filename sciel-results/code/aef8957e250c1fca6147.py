import itertools
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Precompute primes up to 256
    max_sum = 256
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_sum + 1, i):
                is_prime[j] = False
    
    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]
    
    # Precompute: for each odd o and even e, is o+e prime?
    prime_sum = [[False]*len(evens) for _ in range(len(odds))]
    sum_vals = [[0]*len(evens) for _ in range(len(odds))]
    for i, o in enumerate(odds):
        for j, e in enumerate(evens):
            s = o + e
            sum_vals[i][j] = s
            prime_sum[i][j] = is_prime[s]
    
    def check_full(oi, oj, oj2, ei, ej, ej2):
        """Check if 9 sums are pairwise coprime. Returns count of distinct sums."""
        sums = [
            sum_vals[oi][ei], sum_vals[oi][ej], sum_vals[oi][ej2],
            sum_vals[oj][ei], sum_vals[oj][ej], sum_vals[oj][ej2],
            sum_vals[oj2][ei], sum_vals[oj2][ej], sum_vals[oj2][ej2]
        ]
        for i in range(9):
            for j in range(i+1, 9):
                if math.gcd(sums[i], sums[j]) != 1:
                    return 0
        return len(set(sums))
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Phase 1: Targeted search for 9 distinct prime sums
    # For each odd triple, find even triples where all 9 sums are prime
    random.seed(123)
    
    # First, for each odd, count how many evens give prime sums
    odd_prime_counts = []
    for i in range(len(odds)):
        cnt = sum(1 for j in range(len(evens)) if prime_sum[i][j])
        odd_prime_counts.append(cnt)
    
    # Sort odds by prime count (descending) for better search
    odd_order = sorted(range(len(odds)), key=lambda x: -odd_prime_counts[x])
    
    # Try top odd triples
    top_odds = odd_order[:40]  # Top 40 odds
    
    found_9 = False
    for oi1 in top_odds:
        if time.time() > deadline - 5:
            break
        for oi2 in top_odds:
            if oi2 <= oi1:
                continue
            if time.time() > deadline - 5:
                break
            for oi3 in top_odds:
                if oi3 <= oi2:
                    continue
                if time.time() > deadline - 5:
                    break
                
                # For this odd triple, find evens where all 3 sums with this odd are prime
                # Actually, we need all 9 sums to be prime (or at least pairwise coprime)
                # Let's find evens that work with all 3 odds
                
                # For each even, check if o1+e, o2+e, o3+e are all prime
                # Then pick 3 such evens and verify all cross-sums are also prime/coprime
                
                compatible_evens = []
                for ej in range(len(evens)):
                    if prime_sum[oi1][ej] and prime_sum[oi2][ej] and prime_sum[oi3][ej]:
                        compatible_evens.append(ej)
                
                if len(compatible_evens) < 3:
                    continue
                
                # Try combinations of 3 compatible evens
                for ej1, ej2, ej3 in itertools.combinations(compatible_evens, 3):
                    score = check_full(oi1, oi2, oi3, ej1, ej2, ej3)
                    if score > best_score:
                        best_score = score
                        best_A = sorted([odds[oi1], odds[oi2], odds[oi3]])
                        best_B = sorted([evens[ej1], evens[ej2], evens[ej3]])
                        if score == 9:
                            found_9 = True
                            break
                if found_9:
                    break
            if found_9:
                break
        if found_9:
            break
    
    # If we found 9, we're done
    if found_9:
        A, B = best_A, best_B
        # Validate
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        valid = True
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    valid = False
                    break
        if valid and len(A) == 3 and len(B) == 3 and all(1 <= x <= 128 for x in A) and all(1 <= x <= 128 for x in B):
            return [A, B]
    
    # Phase 2: Broader random search if 9 not found
    if not found_9 and time.time() < deadline - 5:
        for _ in range(200000):
            if time.time() > deadline - 5:
                break
            oi = random.sample(range(len(odds)), 3)
            oi.sort()
            ej = random.sample(range(len(evens)), 3)
            ej.sort()
            score = check_full(oi[0], oi[1], oi[2], ej[0], ej[1], ej[2])
            if score > best_score:
                best_score = score
                best_A = sorted([odds[oi[0]], odds[oi[1]], odds[oi[2]]])
                best_B = sorted([evens[ej[0]], evens[ej[1]], evens[ej[2]]])
    
    # Final validation
    A, B = best_A, best_B
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    valid = True
    for i in range(len(distinct)):
        for j in range(i+1, len(distinct)):
            if math.gcd(distinct[i], distinct[j]) != 1:
                valid = False
                break
    if not valid or len(set(A)) != 3 or len(set(B)) != 3:
        return [[1, 7, 13], [4, 10, 16]]
    if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
        return [[1, 7, 13], [4, 10, 16]]
    
    return [A, B]
