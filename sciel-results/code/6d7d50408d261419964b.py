import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(256)
    prime_set = set(primes)
    odd_primes = [p for p in primes if p > 2]
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    possible_diffs = [2*d for d in range(1, 64)]
    
    for d2 in possible_diffs:
        if time.time() > deadline:
            break
        for d3 in possible_diffs:
            if d3 <= d2:
                continue
            if time.time() > deadline:
                break
                
            valid_p = []
            for p in odd_primes:
                if p + d3 > 256:
                    break
                if (p + d2) in prime_set and (p + d3) in prime_set:
                    valid_p.append(p)
            
            if len(valid_p) < 3:
                continue
            
            # Try a few combinations of 3 primes from valid_p
            # If valid_p is large, just take the first few
            # We can limit the number of combinations to avoid explosion
            # But valid_p is likely small.
            
            found_solution = False
            # Iterate over first 10 combinations if available, else all
            # Actually, just iterating over all triples might be too slow if valid_p is large?
            # Max valid_p is 50. 50 choose 3 is ~ 19600.
            # 4000 * 19600 is too much.
            # But valid_p is usually very small for specific d2, d3.
            # Let's just try the first 5 triples.
            
            triples_to_try = list(itertools.combinations(valid_p, 3))
            # Limit to first 20 triples to be safe
            for p1, p2, p3 in triples_to_try[:20]:
                lower = max(1, 1 - d2, 1 - d3, p1 - 128, p2 - 128, p3 - 128)
                upper = min(128, 128 - d2, 128 - d3, p1 - 1, p2 - 1, p3 - 1)
                
                if lower > upper:
                    continue
                
                start_even = lower if lower % 2 == 0 else lower + 1
                for b1 in range(start_even, upper + 1, 2):
                    b1_val = b1
                    b2_val = b1_val + d2
                    b3_val = b1_val + d3
                    a1_val = p1 - b1_val
                    a2_val = p2 - b1_val
                    a3_val = p3 - b1_val
                    
                    A = [a1_val, a2_val, a3_val]
                    B = [b1_val, b2_val, b3_val]
                    
                    if len(set(A)) != 3 or len(set(B)) != 3:
                        continue
                    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
                        continue
                    
                    sums = [a + b for a in A for b in B]
                    distinct_sums = list(set(sums))
                    
                    if len(distinct_sums) < 9:
                        continue
                    
                    valid = True
                    for i in range(len(distinct_sums)):
                        for j in range(i + 1, len(distinct_sums)):
                            if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        score = len(distinct_sums)
                        if score > best_score:
                            best_score = score
                            best_A = A
                            best_B = B
                            if score == 9:
                                return [best_A, best_B]
                        break # Found a valid b1 for this triplet, try next triplet? No, we want max score.
                             # If score is 9, we return. If score < 9, we continue searching.
                             # But if we found a valid one, we can break the b1 loop.
                             # We should continue to other triplets to see if we get 9?
                             # If we found 9, we returned.
                             # If we found <9, we keep looking for 9.
            
            # If we found a solution with score > best_score, we updated best.
            # We continue searching for 9.

    if best_score > 5:
        return [best_A, best_B]
    
    return [best_A, best_B]
