"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def _is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_kernel(n):
    """Compute the squarefree kernel of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # 210 = 2*3*5*7. We need distinct ints in (210, 10210]
    # whose product * 210 is a perfect square.
    # sf(210 * prod(a_i)) = 1, meaning sf(prod(a_i)) = 210 (since sf is multiplicative mod squares)
    # More precisely: for each prime p, exponent in prod(a_i) + exponent in 210 must be even.
    # For p in {2,3,5,7}: exponent in prod(a_i) must be odd.
    # For p not in {2,3,5,7}: exponent in prod(a_i) must be even.

    best_max = 10211
    best_set = None

    # Strategy 1: Try pairs (a, b) where a*b*210 is a perfect square
    # a*b must have sf = 210
    # a = d1 * k1^2, b = d2 * k2^2 where sf(a)=d1, sf(b)=d2, and d1*d2 = 210 * (square)
    # Since d1,d2 are squarefree, d1*d2 must be 210 * s^2 for some integer s.
    # But d1*d2 is squarefree only if d1 and d2 share no primes. If they share primes,
    # those primes appear with even exponent in d1*d2.
    # So sf(d1*d2) = 210.
    # d1*d2 = 210 * m^2 for some m, where m's primes are those shared between d1 and d2.
    # Actually: let d1*d2 = 210 * t where t is a perfect square (since extra shared primes contribute even powers).
    # Wait: if prime p appears in both d1 and d2, it appears with exponent 2 in d1*d2, so it's "cancelled".
    # So sf(d1*d2) = sf(210) = 210.
    # d1*d2 = 210 * u^2 for some integer u.
    # d1 = 210 * u^2 / d2, and d1 must be squarefree.
    # For each divisor d2 of 210*u^2... this is complex. Let's just enumerate.

    # Simpler: for each candidate a in range(211, 10211), compute sf(a).
    # For a pair (a,b): sf(a)*sf(b) must have squarefree part 210.
    # i.e., sf(sf(a) * sf(b)) = 210.
    # sf(a)*sf(b) = 210 * k^2 for some k.

    # Let's precompute sf for all numbers in range
    # But 10210-210 = ~10000 numbers, that's fine.

    # Actually, let's be smarter. For a pair (a,b):
    # a*b*210 = square means a*b = 210 * (square/210)... no.
    # a*b*210 = c^2, so a*b = c^2/210 = c^2/(2*3*5*7)
    # So 210 must divide c^2, meaning 2*3*5*7 divides c, so c = 210*d for some d? No.
    # c^2 = 210*a*b. Since 210 is squarefree, 210 | c, so c = 210*k.
    # Then (210*k)^2 = 210*a*b => 210*k^2 = a*b.
    # So we need a*b = 210*k^2 for some integer k.
    # a and b must be distinct, both in (210, 10210].
    # a = 210*k^2/b, so b must divide 210*k^2.
    # For each k, 210*k^2 must be factorable into two distinct factors a,b both in (210,10210].
    # a*b = 210*k^2, a != b, 210 < a <= 10210, 210 < b <= 10210.

    # Let's iterate k from 1 upward.
    # For k=1: a*b = 210. Factors of 210: 1,2,3,5,6,7,10,14,15,21,30,35,42,70,105,210.
    #   Pairs: (1,210),(2,105),(3,70),(5,42),(6,35),(7,30),(10,21). None have both > 210.
    # For k=2: a*b = 210*4 = 840. Factors... need both > 210.
    #   840 = 210*4. Pairs: (1,840),(2,420),(3,280),(4,210),(5,168),(6,140),(7,120),(8,105),(10,84),(12,70),(14,60),(15,56),(20,42),(21,40),(24,35),(28,30).
    #   None have both > 210.
    # For k=3: a*b = 210*9 = 1890. Pairs with both > 210:
    #   1890/210 = 9. So if a=210, b=9 (no). Let's find pairs.
    #   1890 = 2*3^3*5*7. Divisors... let's check: 210*9=1890.
    #   a*b=1890, a>210, b>210. a*b=1890, min(a,b) > 210 => a*b > 210^2 = 44100. But 1890 < 44100. No solution.
    # For k=4: a*b = 210*16 = 3360. 3360 < 44100. No.
    # For k=5: a*b = 210*25 = 5250. 5250 < 44100. No.
    # ...
    # We need 210*k^2 > 210^2, so k^2 > 210, k >= 15 (15^2=225).
    # For k=15: a*b = 210*225 = 47250. Need a,b in (210, 10210], a*b=47250, a!=b.
    #   sqrt(47250) ≈ 217.4. So one factor < 217, other > 217. But we need both > 210.
    #   47250/210 ≈ 225. So b = 47250/a. If a=211, b=47250/211 ≈ 223.9 (not integer).
    #   Let's find divisors of 47250 in range (210, 218) [since the smaller factor must be < sqrt(47250)≈217.4].
    #   47250 = 2*3^2*5^3*7. Divisors near 210-217:
    #   210 = 2*3*5*7. 47250/210 = 225. So (210, 225) but 210 is not > 210.
    #   216 = 2^3*3^3. 47250/216 = 218.75. No.
    #   215 = 5*43. 47250/215 = 219.77. No.
    #   Let me just check: 47250 = 225*210. 225 = 15^2.
    #   Other factorizations: 47250 = 250*189 (189<210, no). 300*157.5 (no).
    #   47250/211 = 223.93... 47250/212 = 222.88... 47250/213 = 221.83...
    #   47250/214 = 220.79... 47250/215 = 219.77... 47250/216 = 218.75... 47250/217 = 217.74...
    #   None integer. So k=15 gives no valid pair.

    # This pair approach is slow. Let me think differently.

    # Key insight: the minimum possible max is at least 211 (since we need > 210).
    # Can we achieve max = 211? We'd need a set where all elements are in {211} but they must be distinct, so only [211].
    # 211*210 = 44310. Is 44310 a perfect square? sqrt(44310) ≈ 210.5. 210^2=44100, 211^2=44521. No.

    # max=212: [212] or [211,212] etc.
    # [212]: 212*210 = 44520. sqrt ≈ 211.0. 211^2=44521. No.
    # [211,212]: 211*212*210 = 211*44520 = 9394920. sqrt ≈ 3065.1. 3065^2=9394225, 3066^2=9400356. No.

    # Let me try a more systematic approach: for small M (max value), check if there exists a valid subset of {211,...,M}.

    # For efficiency, use the squarefree kernel approach.
    # sf(210 * prod(a_i)) = 1.
    # Let's represent each number by its "parity vector" over all primes.
    # But there are too many primes. Instead, note that only primes that appear in any a_i matter.
    # For a set S, the condition is: for each prime p, sum_{a in S} (exponent of p in a) + (exponent of p in 210) is even.
    # For p in {2,3,5,7}: sum must be odd.
    # For other p: sum must be even.

    # Let's try small sets. Start with size 1, then 2, etc.
    # For size 1: a*210 = square => a = 210*k^2. a > 210 => k >= 2 (a=840). max=840. score=840-210=630.
    # For size 2: a*b*210 = square. As above, a*b = 210*k^2.
    #   We need to find the minimum max(a,b) over all valid pairs.
    #   For each k, factor 210*k^2 and find factor pairs (a,b) with a!=b, both in (210, 10210].
    #   To minimize max(a,b), we want a and b as close as possible to sqrt(210*k^2) = 15*sqrt(2)*k ≈ 21.21*k.
    #   But both must be > 210, so 21.21*k > 210 => k > 9.9, so k >= 10.
    #   For k=10: a*b = 210*100 = 21000. sqrt(21000) ≈ 144.9. Both a,b > 210 => a*b > 44100 > 21000. Impossible.
    #   Wait, I made an error. If a > 210 and b > 210, then a*b > 44100. So 210*k^2 > 44100 => k^2 > 210 => k >= 15.
    #   For k=15: a*b = 210*225 = 47250. sqrt ≈ 217.4. Need a,b in (210, 218) roughly.
    #   As checked above, no integer factor pair.
    #   For k=16: a*b = 210*256 = 53760. sqrt ≈ 231.9. Need a,b in (210, 232).
    #   53760/211 = 254.8... 53760/212 = 253.6... 53760/213 = 252.4...
    #   53760 = 2^7 * 3 * 5 * 7. Let me find divisors in (210, 232):
    #   210: 53760/210 = 256. (210 not > 210)
    #   216: 53760/216 = 248.89. No.
    #   220: 53760/220 = 244.36. No.
    #   224: 53760/224 = 240. Yes! 224*240 = 53760. Both > 210, distinct.
    #   So (224, 240) works! max = 240. score = 240-210 = 30.

    # Let me verify: 224*240*210 = 224*50400 = 11289600. sqrt(11289600) = 3360. 3360^2 = 11289600. Yes!
    # So [224, 240] gives score 30.

    # Can we do better? Check k=15 more carefully, and other k values.
    # For k=15: 47250. Divisors: 47250 = 2*3^2*5^3*7.
    #   Divisors in (210, 218): 210 (no, not >210), 215=5*43 (47250/215=219.77, no), 216=2^3*3^3 (47250/216=218.75, no).
    #   No valid pair for k=15.

    # For k=14: a*b = 210*196 = 41160. sqrt ≈ 202.9. Both > 210 => a*b > 44100 > 41160. Impossible.

    # So k=16 gives the best pair with max=240. But can we get max < 240 with more elements?

    # For 3 elements: a*b*c*210 = square.
    # This is more complex. Let's try to find if max=231 or less is achievable.
    # We need a subset of {211,...,231} such that sf(prod)*210 is a square.
    # Equivalently, sf(prod) = 210 (in the sense that the product has odd exponent for 2,3,5,7 and even for others).

    # Let me compute sf for numbers 211-231:
    # 211: prime. sf=211.
    # 212: 4*53. sf=53.
    # 213: 3*71. sf=213.
    # 214: 2*107. sf=214.
    # 215: 5*43. sf=215.
    # 216: 8*27=2^3*3^3. sf=2*3=6.
    # 217: 7*31. sf=217.
    # 218: 2*109. sf=218.
    # 219: 3*73. sf=219.
    # 220: 4*5*11=2^2*5*11. sf=55.
    # 221: 13*17. sf=221.
    # 222: 2*3*37. sf=222.
    # 223: prime. sf=223.
    # 224: 32*7=2^5*7. sf=2*7=14.
    # 225: 15^2=3^2*5^2. sf=1.
    # 226: 2*113. sf=226.
    # 227: prime. sf=227.
    # 228: 4*57=2^2*3*19. sf=57.
    # 229: prime. sf=229.
    # 230: 2*5*23. sf=230.
    # 231: 3*7*11. sf=231.

    # We need to select a subset S of these such that:
    # - For p=2: odd number of selected have p=2 in their sf with odd exponent... actually, the total exponent of 2 in prod(S) must be odd.
    # - Similarly for 3,5,7.
    # - For other primes: total exponent in prod(S) must be even.

    # The sf values with only primes from {2,3,5,7}:
    # 216: sf=6=2*3
    # 220: sf=55=5*11 (has 11)
    # 224: sf=14=2*7
    # 225: sf=1
    # 228: sf=57=3*19 (has 19)

    # For primes outside {2,3,5,7}, they must appear an even number of times total.
    # So if we use 220 (sf=55=5*11), we need another number with 11 in its sf to cancel.
    # 231 has sf=231=3*7*11. So 220 and 231 together: sf product = 55*231 = 5*11*3*7*11 = 3*5*7*11^2. sf = 3*5*7=105.
    # Plus we need total sf to be 210. So we need additional sf=2.
    # 224 has sf=14=2*7. 105*14 = 105*2*7 = 3*5*7^2*2. sf = 2*3*5=30. Not 210.
    # Let me think in terms of parity vectors.

    # This is getting complex. Let me just do a brute force search for small max values.

    # Precompute sf for all numbers in range
    def compute_sf(n):
        result = 1
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count = 0
                while temp % d == 0:
                    temp //= d
                    count += 1
                if count % 2 == 1:
                    result *= d
            d += 1
        if temp > 1:
            result *= temp
        return result

    # For each M from 211 upward, check if there's a valid subset of {211,...,M}
    # Use the condition: prod(S) * 210 is a perfect square.
    # For efficiency, limit subset size to small numbers (1-5).

    # Actually, let's just try to find the answer by checking M from 211 upward.
    # For each M, we can use meet-in-the-middle or just try all subsets up to size 4-5.
    # But 2^20 = 1M subsets for M=231, which is too many.

    # Better: use the squarefree kernel. The condition is sf(prod(S)) = 210.
    # sf is multiplicative: sf(a*b) = sf(sf(a)*sf(b)).
    # So we need sf(prod_{a in S} sf(a)) = 210.
    # Let g(a) = sf(a). We need sf(prod_{a in S} g(a)) = 210.

    # For small subsets, we can just compute directly.
    # Let's try subset sizes 1, 2, 3, 4.

    # Size 1: g(a) = 210. a = 210*k^2 > 210 => k>=2, a=840. max=840.
    # Size 2: sf(g(a)*g(b)) = 210.
    #   g(a)*g(b) = 210 * (square). Since g(a),g(b) are squarefree, g(a)*g(b) has each prime with exponent 0,1,or 2.
    #   sf(g(a)*g(b)) = product of primes with odd exponent in g(a)*g(b).
    #   For sf to be 210=2*3*5*7, primes 2,3,5,7 must have odd exponent, others even.
    #   So g(a)*g(b) = 210 * (square of product of primes in g(a)∩g(b)).
    #   Let's enumerate pairs.

    # Size 3: sf(g(a)*g(b)*g(c)) = 210.

    # Let me just brute force for M up to maybe 300, checking all subsets of size <= 4.
    # For M=231, there are 21 numbers. C(21,1)+C(21,2)+C(21,3)+C(21,4) = 21+210+1330+5985 = 7546. That's fine.

    # But we need to go higher potentially. Let's be smart.

    # Actually, the pair (224,240) gives max=240, score=30. Let's see if we can beat that.
    # Try to find if max <= 239 is achievable.

    # For M from 211 to 239, check if any valid subset exists.
    # Precompute sf for 211 to 239.

    sf_cache = {}
    for n in range(211, 240):
        sf_cache[n] = compute_sf(n)

    # Check all subsets of {211,...,M} for each M
    found = False
    for M in range(211, 240):
        if time.time() > deadline:
            break
        candidates = list(range(211, M + 1))
        # Try subsets of size 1 to 4
        for size in range(1, 5):
            for subset in combinations(candidates, size):
                prod = 210
                for x in subset:
                    prod *= x
                if _is_perfect_square(prod):
                    max_val = max(subset)
                    if max_val < best_max:
                        best_max = max_val
                        best_set = list(subset)
                    found = True
                    # Don't break, keep searching for potentially better
            if time.time() > deadline:
                break
        if found and best_max <= M:
            # We found a solution with max <= M, and M is increasing, so this is optimal
            # Actually no, we might find a smaller max later... no, M is increasing.
            # Once we find a solution at M, that's the best possible since we're going in order.
            break

    # If we didn't find anything better than the pair, use the pair
    if best_set is None or best_max > 240:
        # Use the known pair (224, 240)
        best_set = [224, 240]
        best_max = 240

    # Verify
    product = 210
    for x in best_set:
        product *= x
    if not _is_perfect_square(product):
        best_set = [840]
        best_max = 840

    # Ensure distinct and in range
    best_set = list(set(best_set))
    best_set = [x for x in best_set if 210 < x <= 10210]
    if not best_set:
        best_set = [840]
        best_max = 840

    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
