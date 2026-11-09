import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s wall-clock limit

    n = 210
    max_val = 10210

    # 210 = 2 * 3 * 5 * 7 (square-free)
    # We need: product(selected) * 210 is a perfect square
    # Equivalently: XOR of parity-signatures of selected numbers XOR {2,3,5,7} = empty set
    # For a k-element solution: XOR of all selected signatures = {2,3,5,7}

    # Key insight: 210 has exactly 4 prime factors, so the target signature has 4 elements.
    # For a 2-element solution: sig(a) XOR sig(b) = {2,3,5,7}
    # This means sig(a) and sig(b) differ in exactly the positions {2,3,5,7}
    # i.e., sig(b) = sig(a) XOR {2,3,5,7}

    # The current program scores 21, which means it found a solution with max-210 = 21/seed.
    # Wait, the score is seed/value. If score is 21, then value = seed/21.
    # The "seed program" scores 1.0, so seed = 1.0 * value_seed.
    # Actually, the problem says "the search score is seed/value, so 1.0 equals the seed program".
    # So if seed program has value V_seed, score = seed/V_seed = 1.0, meaning seed = V_seed.
    # Current program has score 21, so 21 = seed/value_current = V_seed/value_current.
    # This means value_current = V_seed/21.
    # We need to find a better (smaller) value.

    # Let me think about what values are achievable.
    # For 1 element: x*210 must be perfect square. x = 210*k^2.
    # x > 210 means k >= 2 (k=1 gives x=210, not > 210).
    # k=2: x = 210*4 = 840, value = 630
    # k=3: x = 210*9 = 1890, value = 1680
    # So best 1-element: 630

    # For 2 elements: need a,b such that a*b*210 is perfect square.
    # This means a*b = 210 * m^2 for some integer m, OR more precisely,
    # the product a*b*210 has all even exponents in its prime factorization.
    # Let's denote sf(x) as the square-free part of x (product of primes with odd exponent).
    # Then sf(a)*sf(b)*sf(210) must be a perfect square, i.e., sf(a)*sf(b)*210 must be a perfect square.
    # Since 210 is square-free, we need sf(a)*sf(b) to be a perfect square times 210... 
    # Actually, sf(a)*sf(b)*210 being a perfect square means sf(a)*sf(b) must equal 210 * (perfect square).
    # But sf(a) and sf(b) are square-free, so their product is square-free iff they're disjoint,
    # or has some squared factors if they share primes.
    # More precisely: sf(a)*sf(b)*210 is a perfect square iff sf(a) XOR sf(b) XOR 210 = 1 (empty square-free part).
    # In terms of sets: sig(a) Δ sig(b) Δ {2,3,5,7} = ∅, i.e., sig(a) Δ sig(b) = {2,3,5,7}.

    # So sig(b) = sig(a) Δ {2,3,5,7}.

    # Strategy: For each possible signature s, find the smallest number > 210 with that signature.
    # Then for each s, check if s Δ {2,3,5,7} also exists, and find the pair minimizing max.

    # But the key to getting a much better score is to find a very small max element.
    # The minimum possible max element is at least 211.
    # Can we get max = 211? That would require both a and b <= 211, with a,b > 210,
    # so a=211, b=211 or a=211, b=211... but they must be distinct.
    # So min max for 2 elements is at least 212 (if 211 and 212 work).
    # value = 212 - 210 = 2. That would be amazing!

    # Let me check: does there exist a pair (a,b) with max(a,b) very small?
    # We need sig(a) Δ sig(b) = {2,3,5,7}.

    # Let me compute signatures of small numbers > 210:
    # 211: prime, sig = {211}
    # 212 = 4*53 = 2^2 * 53, sig = {53}
    # 213 = 3*71, sig = {3, 71}
    # 214 = 2*107, sig = {2, 107}
    # 215 = 5*43, sig = {5, 43}
    # 216 = 8*27 = 2^3 * 3^3, sig = {2, 3}
    # 217 = 7*31, sig = {7, 31}
    # 218 = 2*109, sig = {2, 109}
    # 219 = 3*73, sig = {3, 73}
    # 220 = 4*5*11 = 2^2*5*11, sig = {5, 11}
    # 221 = 13*17, sig = {13, 17}
    # 222 = 2*3*37, sig = {2, 3, 37}
    # 223: prime, sig = {223}
    # 224 = 32*7 = 2^5*7, sig = {2, 7}
    # 225 = 15^2 = 3^2*5^2, sig = {} (empty, perfect square)
    # 226 = 2*113, sig = {2, 113}
    # 227: prime, sig = {227}
    # 228 = 4*57 = 2^2*3*19, sig = {3, 19}
    # 229: prime, sig = {229}
    # 230 = 2*5*23, sig = {2, 5, 23}
    # 231 = 3*7*11, sig = {3, 7, 11}
    # 232 = 8*29 = 2^3*29, sig = {2, 29}
    # 233: prime, sig = {233}
    # 234 = 2*3^2*13, sig = {2, 13}
    # 235 = 5*47, sig = {5, 47}
    # 236 = 4*59 = 2^2*59, sig = {59}
    # 237 = 3*79, sig = {3, 79}
    # 238 = 2*7*17, sig = {2, 7, 17}
    # 239: prime, sig = {239}
    # 240 = 16*15 = 2^4*3*5, sig = {3, 5}
    # 241: prime, sig = {241}
    # 242 = 2*121 = 2*11^2, sig = {2}
    # 243 = 3^5, sig = {3}
    # 244 = 4*61 = 2^2*61, sig = {61}
    # 245 = 5*49 = 5*7^2, sig = {5}
    # 246 = 2*3*41, sig = {2, 3, 41}
    # 247 = 13*19, sig = {13, 19}
    # 248 = 8*31 = 2^3*31, sig = {2, 31}
    # 249 = 3*83, sig = {3, 83}
    # 250 = 2*125 = 2*5^3, sig = {2, 5}
    # 251: prime, sig = {251}
    # 252 = 4*63 = 2^2*3^2*7, sig = {7}
    # 253 = 11*23, sig = {11, 23}
    # 254 = 2*127, sig = {2, 127}
    # 255 = 3*5*17, sig = {3, 5, 17}
    # 256 = 2^8, sig = {} (perfect square)
    # 257: prime, sig = {257}
    # 258 = 2*3*43, sig = {2, 3, 43}
    # 259 = 7*37, sig = {7, 37}
    # 260 = 4*65 = 2^2*5*13, sig = {5, 13}
    # 261 = 9*29 = 3^2*29, sig = {29}
    # 262 = 2*131, sig = {2, 131}
    # 263: prime, sig = {263}
    # 264 = 8*33 = 2^3*3*11, sig = {2, 3, 11}
    # 265 = 5*53, sig = {5, 53}
    # 266 = 2*7*19, sig = {2, 7, 19}
    # 267 = 3*89, sig = {3, 89}
    # 268 = 4*67 = 2^2*67, sig = {67}
    # 269: prime, sig = {269}
    # 270 = 2*3^3*5, sig = {2, 3, 5}
    # 271: prime, sig = {271}
    # 272 = 16*17 = 2^4*17, sig = {17}
    # 273 = 3*7*13, sig = {3, 7, 13}
    # 274 = 2*137, sig = {2, 137}
    # 275 = 25*11 = 5^2*11, sig = {11}
    # 276 = 4*69 = 2^2*3*23, sig = {3, 23}
    # 277: prime, sig = {277}
    # 278 = 2*139, sig = {2, 139}
    # 279 = 9*31 = 3^2*31, sig = {31}
    # 280 = 8*35 = 2^3*5*7, sig = {2, 5, 7}
    # 281: prime, sig = {281}
    # 282 = 2*3*47, sig = {2, 3, 47}
    # 283: prime, sig = {283}
    # 284 = 4*71 = 2^2*71, sig = {71}
    # 285 = 3*5*19, sig = {3, 5, 19}
    # 286 = 2*11*13, sig = {2, 11, 13}
    # 287 = 7*41, sig = {7, 41}
    # 288 = 32*9 = 2^5*3^2, sig = {2}
    # 289 = 17^2, sig = {} (perfect square)
    # 290 = 2*5*29, sig = {2, 5, 29}
    # 291 = 3*97, sig = {3, 97}
    # 292 = 4*73 = 2^2*73, sig = {73}
    # 293: prime, sig = {293}
    # 294 = 2*3*7^2, sig = {2, 3}
    # 295 = 5*59, sig = {5, 59}
    # 296 = 8*37 = 2^3*37, sig = {2, 37}
    # 297 = 27*11 = 3^3*11, sig = {3, 11}
    # 298 = 2*149, sig = {2, 149}
    # 299 = 13*23, sig = {13, 23}
    # 300 = 4*75 = 2^2*3*5^2, sig = {3}

    # Target: sig(a) Δ sig(b) = {2,3,5,7}
    # Let me check if any pair with small max works.
    # sig(216) = {2,3}. Need sig(b) = {2,3} Δ {2,3,5,7} = {5,7}
    # Is there a number > 210 with sig = {5,7}? 
    # sig(245) = {5}, sig(252) = {7}
    # Need a number whose sig is exactly {5,7}. That means the number has odd exponent of 5 and 7, 
    # and even exponent of all other primes.
    # 5*7 = 35, but 35 < 211. 5*7*k^2 for k>=2: 35*4=140 (<211), 35*9=315 (>210). 
    # 315 = 5*63 = 5*9*7 = 3^2*5*7, sig = {5,7}. Yes! 315 works.
    # So pair (216, 315): max = 315, value = 105.

    # Can we do better? Let me check sig(240) = {3,5}. Need sig(b) = {3,5} Δ {2,3,5,7} = {2,7}
    # sig(224) = {2,7}! Pair (224, 240): max = 240, value = 30.

    # Even better! Let me check if we can get even smaller.
    # sig(242) = {2}. Need sig(b) = {2} Δ {2,3,5,7} = {3,5,7}
    # Need a number with sig {3,5,7}: 3*5*7 = 105, 105*4 = 420, 105*9 = 945, etc.
    # 420 = 2^2*3*5*7, sig = {3,5,7}. So pair (242, 420): max = 420, value = 210. Worse.

    # sig(243) = {3}. Need sig(b) = {3} Δ {2,3,5,7} = {2,5,7}
    # 2*5*7 = 70, 70*4 = 280. 280 = 2^3*5*7, sig = {2,5,7}. Pair (243, 280): max = 280, value = 70.

    # sig(245) = {5}. Need sig(b) = {5} Δ {2,3,5,7} = {2,3,7}
    # 2*3*7 = 42, 42*4 = 168 (<211), 42*9 = 378. 378 = 2*3^3*7, sig = {2,3,7}. Pair (245, 378): max = 378, value = 168.

    # sig(252) = {7}. Need sig(b) = {7} Δ {2,3,5,7} = {2,3,5}
    # 2*3*5 = 30, 30*4 = 120 (<211), 30*9 = 270. 270 = 2*3^3*5, sig = {2,3,5}. Pair (252, 270): max = 270, value = 60.

    # sig(216) = {2,3}, we found (216, 315) with value 105.
    # sig(240) = {3,5}, we found (224, 240) with value 30.

    # Can we get value < 30? That means max < 240.
    # Let me systematically check all numbers from 211 to 239 and see if their complement exists <= 239.

    # I'll just compute this in the code with a proper search.
    # The key structural change: instead of building full signature maps and checking all pairs,
    # do a targeted search from the smallest possible max upward.

    # SPF sieve
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_sig(x):
        sig = set()
        temp = x
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                sig.add(p)
        return frozenset(sig)

    target = frozenset({2, 3, 5, 7})

    # Precompute signatures for all numbers in range
    sigs = {}
    for x in range(211, max_val + 1):
        sigs[x] = get_sig(x)

    # Check 1-element solutions
    best_value = float('inf')
    best_answer = None
    for k in range(1, 100):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            val = x - 210
            if val < best_value:
                best_value = val
                best_answer = [x]
        else:
            break

    # Check 2-element solutions by iterating max from 211 upward
    # For each upper bound U, check if there exists a pair (a,b) with 211 <= a < b <= U
    # such that sig(a) Δ sig(b) = target.
    # To be efficient: for each U from 212 to max_val, maintain a set of seen signatures.
    # When we add number U, check if sig(U) Δ target is in the seen set.

    seen_sigs = {}  # sig -> list of numbers with that sig (for tracking)
    found = False
    for U in range(212, max_val + 1):
        if time.time() > deadline - 5:
            break
        s = sigs[U]
        complement = s ^ target
        if complement in seen_sigs:
            # Found a pair
            a = seen_sigs[complement][0]
            b = U
            val = U - 210
            if val < best_value:
                best_value = val
                best_answer = [a, b]
            found = True
            break
        if s not in seen_sigs:
            seen_sigs[s] = []
        seen_sigs[s].append(U)

    # Also check: what if a = U and b < U? Already handled since we check complement.
    # But we also need to handle the case where both a and b have the same signature and
    # s ^ target == s, i.e., target = empty. But target = {2,3,5,7} is not empty, so s ^ target != s always.

    # Verify
    if best_answer is not None:
        prod = 210
        for x in best_answer:
            prod *= x
        sqrt_val = math.isqrt(prod)
        if sqrt_val * sqrt_val != prod:
            best_answer = [840]
            best_value = 630

    if best_answer is None:
        best_answer = [840]
        best_value = 630

    return [int(x) for x in best_answer]
