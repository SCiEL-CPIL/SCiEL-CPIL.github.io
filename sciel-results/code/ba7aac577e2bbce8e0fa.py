# EVOLVE-BLOCK-START
import math

def construct():
    N = 256
    
    # Sieve of Eratosthenes to find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    # The problem is to maximize the sum of a set of distinct integers from 1..N
    # such that every pair is coprime.
    # 
    # Key insight: The set of prime factors used by the selected numbers must be disjoint.
    # For any prime p, if we select numbers that are multiples of p, they all must be
    # of the form p^k * m where m is coprime to p. But since we want to maximize the sum,
    # and the numbers must be distinct, we essentially partition the primes into groups.
    # For each group of primes, we pick a single number that is the product of powers of
    # those primes, such that the product is <= N. To maximize the sum, for a given set of
    # primes, we should pick the largest possible number <= N that uses only those primes.
    # 
    # Actually, a stronger observation:
    # If a number x is in the set, and x = a * b with gcd(a,b)=1 and a,b > 1, then we could
    # potentially replace x with a and b (if they are <= N and distinct from others) to get a
    # larger sum (a+b > x for a,b >= 2? No, 2*2=4, 2+2=4. 2*3=6, 2+3=5. So a+b < a*b for a,b>2).
    # Wait, the constraint is that the numbers must be pairwise coprime.
    # If we replace x with a and b, we need a and b to be coprime to all other numbers in the set.
    # Since x was coprime to others, and a,b divide x, a and b are also coprime to others.
    # But a and b must be coprime to each other (which they are by construction).
    # So if we can split x into a and b, we gain a+b - x.
    # For x=6, a=2, b=3. 2+3=5 < 6. So splitting reduces sum.
    # For x=12, a=4, b=3. 4+3=7 < 12.
    # For x=4, a=2, b=2. Not distinct, and not coprime.
    # 
    # So it seems we want to use composite numbers as much as possible, provided they are
    # "large enough" that splitting them doesn't increase the sum.
    # Specifically, if x = a*b, we want a*b >= a+b, which is true for a,b >= 3.
    # If a=2, b must be such that 2b >= 2+b => b >= 2.
    # So 2*2=4, 2+2=4 (tie, but distinctness fails).
    # 2*3=6, 2+3=5 (splitting is worse).
    # 2*4=8, 2+4=6 (splitting is worse).
    # 
    # So the optimal strategy is:
    # 1. Identify the primes.
    # 2. Group primes such that the product of their maximal powers (within N) forms a number
    #    that is "efficient" to keep together.
    # 
    # Actually, the problem is equivalent to:
    # Partition the set of primes into disjoint subsets S1, S2, ..., Sk.
    # For each subset Si, let Pi be the product of the primes in Si.
    # We can choose a number xi such that xi is a multiple of Pi and xi <= N.
    # To maximize the sum, for each Si, we should choose the largest multiple of Pi that is <= N.
    # Let xi = floor(N/Pi) * Pi.
    # We want to maximize sum(xi).
    # 
    # Note: If Si = {p}, xi = largest power of p <= N.
    # If Si = {p, q}, xi = largest multiple of p*q <= N.
    # 
    # Since N=256 is small, we can try to find the optimal partition of the primes.
    # Primes <= 256: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251.
    # 
    # For large primes p > sqrt(N) = 16, p*q > N for any other prime q.
    # So large primes must be in singleton groups.
    # For a singleton group {p}, the best number is p itself (since 2p > N).
    # Wait, if p > 16, then 2p > 32. If p=17, 2*17=34 <= 256.
    # So for p=17, we can pick 17, 34, 51, ..., 238.
    # The largest is floor(256/17)*17 = 15*17 = 255.
    # So for p=17, we can pick 255.
    # Is 255 coprime to other numbers?
    # 255 = 3 * 5 * 17.
    # So if we pick 255, we "use" primes 3, 5, 17.
    # This means 3 and 5 cannot be used in other numbers.
    # 
    # So the problem is:
    # Assign each prime to a "bucket". Each bucket corresponds to a number xi.
    # xi must be a multiple of the product of primes in the bucket.
    # xi <= N.
    # To maximize sum, for a bucket with prime product P, we pick xi = floor(N/P)*P.
    # 
    # This is a complex optimization problem. However, N=256 is small.
    # The number of primes is 53.
    # Most primes are large and will be singletons.
    # Only small primes (2, 3, 5, 7, 11, 13) can be grouped.
    # 
    # Let's check the small primes.
    # Primes <= 16: 2, 3, 5, 7, 11, 13.
    # Primes > 16: 17, 19, ..., 251.
    # 
    # For a prime p > 16, can it be grouped with a small prime?
    # If we group p with q (small), the product is p*q.
    # The number would be floor(N/(p*q)) * p*q.
    # Since p > 16, p*q > 16*q.
    # If q=2, p*q > 32.
    # If p=17, q=2, P=34. xi = floor(256/34)*34 = 7*34 = 238.
    # If we kept them separate:
    # p=17 -> 255 (uses 3,5,17? No, 255=3*5*17. Wait, if I pick 255 for 17, I am using 3 and 5 too.
    # This is the tricky part. The "bucket" logic assumes we only use the primes in the bucket.
    # But if I pick 255 for the bucket {17}, I am effectively using {3, 5, 17}.
    # So the bucket is actually {3, 5, 17}.
    # 
    # So the decision is: which primes go together?
    # Let's analyze the small primes {2, 3, 5, 7, 11, 13}.
    # And the medium primes {17, 19, 23, 29, 31, 37}.
    # 
    # Let's try to construct the optimal set manually or with a small search.
    # 
    # Candidate 1: All primes separate.
    # Sum = 1 + sum(primes).
    # 
    # Candidate 2: Group small primes to form large numbers.
    # 
    # Let's look at the current best score 1.1516.
    # The seed program uses prime powers.
    # 
    # Let's try to find a better set.
    # 
    # Consider the number 255 = 3*5*17.
    # If we use 255, we use primes 3, 5, 17.
    # We lose the ability to use 3, 5, 17 separately.
    # Gains: 255.
    # Losses: largest power of 3 (243), largest power of 5 (125), largest power of 17 (255? No, 17^1=17, 17^2=289>256. So 17).
    # Wait, if I use 255, I get 255.
    # If I use 3, 5, 17 separately:
    # 3 -> 243
    # 5 -> 125
    # 17 -> 255 (largest multiple of 17 <= 256 is 15*17=255).
    # Sum = 243 + 125 + 255 = 623.
    # If I use 255 (bucket {3,5,17}), I get 255.
    # 255 < 623. So separating is better.
    # 
    # What if I use 255 for bucket {17} but keep 3 and 5 separate?
    # No, 255 uses 3 and 5. So they are not available.
    # 
    # So it seems prime powers are very strong.
    # 
    # Let's check if any composite number is better than the sum of its prime power components.
    # Let x = p^a * q^b.
    # Value if separate: p^a_max + q^b_max.
    # Value if together: x.
    # We want x > p^a_max + q^b_max.
    # 
    # Example: 2 and 3.
    # 2^7 = 128. 3^5 = 243. Sum = 371.
    # Max multiple of 6 <= 256: 42*6 = 252.
    # 252 < 371. Separate is better.
    # 
    # Example: 2 and 5.
    # 2^7 = 128. 5^3 = 125. Sum = 253.
    # Max multiple of 10 <= 256: 25*10 = 250.
    # 250 < 253. Separate is better.
    # 
    # Example: 2 and 7.
    # 2^7 = 128. 7^2 = 49. Sum = 177.
    # Max multiple of 14 <= 256: 18*14 = 252.
    # 252 > 177. Together is better!
    # 
    # So {2, 7} -> 252 is better than {128, 49}.
    # Gain: 252 - 177 = 75.
    # 
    # Let's check other pairs.
    # {2, 11}: 128 + 11 = 139. Max mult of 22: 11*22=242. 242 > 139. Gain 103.
    # {2, 13}: 128 + 13 = 141. Max mult of 26: 9*26=234. 234 > 141. Gain 93.
    # {2, 17}: 128 + 255? No, 17^1=17. 128+17=145. Max mult of 34: 7*34=238. 238 > 145. Gain 93.
    # 
    # It seems grouping 2 with other primes is often beneficial because 2^7=128 is not that large compared to the multiples.
    # 
    # What about {3, 5}?
    # 243 + 125 = 368. Max mult of 15: 17*15=255. 255 < 368. Separate better.
    # 
    # {3, 7}?
    # 243 + 49 = 292. Max mult of 21: 12*21=252. 252 < 292. Separate better.
    # 
    # {5, 7}?
    # 125 + 49 = 174. Max mult of 35: 7*35=245. 245 > 174. Gain 71.
    # 
    # So promising groups: {2, 7}, {2, 11}, {2, 13}, {2, 17}, {5, 7}.
    # 
    # Can we group more?
    # {2, 7, 11}?
    # P = 154. Max mult: 1*154=154.
    # Separate: 128 + 49 + 11 = 188. 154 < 188. Worse.
    # 
    # {2, 5, 7}?
    # P = 70. Max mult: 3*70=210.
    # Separate: 128 + 125 + 49 = 302. 210 < 302. Worse.
    # 
    # So it seems pairs are the best.
    # 
    # Let's try to construct a set using these pairs.
    # We have primes: 2, 3, 5, 7, 11, 13, 17, 19, ...
    # 
    # Option A: Group {2, 11} -> 242.
    # Remaining small primes: 3, 5, 7, 13, 17, ...
    # Group {5, 7} -> 245.
    # Remaining: 3, 13, 17, ...
    # 3 -> 243.
    # 13 -> 13 (or 26? No, 2 is used. 39? 3 is used. 13 is best).
    # 17 -> 255 (3,5 used? No, 3 is used by 243, 5 by 245. 255=3*5*17. Conflict!).
    # If I use 243 (3) and 245 (5,7), I cannot use 255 (3,5,17).
    # So 17 must be used as 17 or a multiple of 17 that doesn't use 3 or 5.
    # 17, 34 (2 used), 51 (3 used), 68 (2 used), 85 (5 used), 102 (2,3 used), 119 (7 used), 136 (2 used), 153 (3 used), 170 (2,5 used), 187 (11 used), 204 (2,3 used), 221 (13 used), 238 (2,7 used), 255 (3,5 used).
    # So 17 is the only option if 2,3,5,7,11,13 are used by others.
    # 
    # Let's sum Option A:
    # 242 ({2,11})
    # 245 ({5,7})
    # 243 ({3})
    # 13 ({13})
    # 17 ({17})
    # Primes > 17: 19, 23, ..., 251.
    # Sum = 242 + 245 + 243 + 13 + 17 + sum(primes > 17).
    # 
    # Option B: Group {2, 7} -> 252.
    # Remaining: 3, 5, 11, 13, 17, ...
    # Group {5, 11}? P=55. Max mult: 4*55=220.
    # Separate: 125 + 11 = 136. 220 > 136. Gain 84.
    # So {5, 11} -> 220.
    # Remaining: 3, 13, 17, ...
    # 3 -> 243.
    # 13 -> 13.
    # 17 -> 17 (conflicts with 3,5? 255=3*5*17. 3 used, 5 used. So 17).
    # Sum = 252 + 220 + 243 + 13 + 17 + sum(primes > 17).
    # 
    # Compare A and B:
    # A: 242 + 245 + 243 + 13 + 17 = 760 + S
    # B: 252 + 220 + 243 + 13 + 17 = 745 + S
    # A is better.
    # 
    # Option C: Group {2, 13} -> 234.
    # Remaining: 3, 5, 7, 11, 17, ...
    # Group {5, 7} -> 245.
    # Remaining: 3, 11, 17, ...
    # 3 -> 243.
    # 11 -> 11.
    # 17 -> 17.
    # Sum = 234 + 245 + 243 + 11 + 17 = 750 + S.
    # Worse than A.
    # 
    # Option D: Group {2, 17} -> 238.
    # Remaining: 3, 5, 7, 11, 13, ...
    # Group {5, 7} -> 245.
    # Remaining: 3, 11, 13, ...
    # 3 -> 243.
    # 11 -> 11.
    # 13 -> 13.
    # Sum = 238 + 245 + 243 + 11 + 13 = 750 + S.
    # Worse than A.
    # 
    # Option E: No grouping of 2.
    # 2 -> 128.
    # Group {5, 7} -> 245.
    # 3 -> 243.
    # 11 -> 11.
    # 13 -> 13.
    # 17 -> 255? (3,5 used? 3 used, 5 used. No). 17.
    # Sum = 128 + 245 + 243 + 11 + 13 + 17 = 657 + S.
    # Much worse.
    # 
    # So Option A seems best so far.
    # 242, 245, 243, 13, 17.
    # 
    # Can we improve?
    # What if we don't use 17 as 17?
    # In Option A, 17 is 17.
    # What if we group 17 with something else?
    # 17 is left over.
    # 
    # What if we change the groups?
    # {2, 11} -> 242.
    # {3, 5} -> 255? No, 255=3*5*17.
    # If we use {3, 5} -> 255, we use 17 too.
    # So {3, 5, 17} -> 255.
    # Remaining: 2, 7, 11, 13.
    # {2, 7} -> 252.
    # {11, 13} -> 143? 11*13=143. Max mult: 1*143=143.
    # Separate: 11+13=24. 143 > 24.
    # So {11, 13} -> 143.
    # Sum = 255 + 252 + 143 = 650.
    # Compare to A: 760. Worse.
    # 
    # What about {2, 5} -> 250?
    # {3, 7} -> 252? No, 3*7=21, 12*21=252.
    # {11, 13} -> 143.
    # 17 -> 17.
    # Sum = 250 + 252 + 143 + 17 = 662. Worse.
    # 
    # It seems Option A is strong.
    # 
    # Let's check if we can use 19, 23, etc. in groups.
    # 19: 255 (3,5,17), 247 (13,19).
    # 13 is used in A. So 247 not possible.
    # 19 -> 19.
    # 
    # So the set is:
    # 1
    # 242 (2*11*11? No, 2*121=242. 11^2=121. 2*121=242. Yes.)
    # 245 (5*49=5*7^2. Yes.)
    # 243 (3^5. Yes.)
    # 13
    # 17
    # Primes > 17: 19, 23, ..., 251.
    # 
    # Sum = 1 + 242 + 245 + 243 + 13 + 17 + sum(primes > 17).
    # 
    # Let's verify pairwise coprimality.
    # 242 = 2 * 11^2
    # 245 = 5 * 7^2
    # 243 = 3^5
    # 13 = 13
    # 17 = 17
    # Primes > 17 are distinct primes.
    # All prime factors are distinct: {2, 11}, {5, 7}, {3}, {13}, {17}, {19, 23, ...}.
    # Yes, pairwise coprime.
    # 
    # This should be a valid and high-scoring solution.
    
    # Calculate sum of primes > 17
    sum_large_primes = sum(p for p in primes if p > 17)
    
    result = [1, 242, 245, 243, 13, 17] + [p for p in primes if p > 17]
    
    # Sanity check
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                return [1] + primes
    
    return result
# EVOLVE-BLOCK-END
