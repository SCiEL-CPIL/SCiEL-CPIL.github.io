"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from math import gcd

# EVOLVE-BLOCK-START

def _tau_perp_small(n):
    """Compute tau_perp(n) by generating all divisors, sorting, counting adjacent coprime pairs."""
    if n < 2:
        return 0
    divisors = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
        i += 1
    ds = sorted(divisors)
    count = 0
    for j in range(len(ds) - 1):
        if gcd(ds[j], ds[j + 1]) == 1:
            count += 1
    return count


def _tau_perp_fast(n, prime_powers):
    """
    Compute tau_perp(n) efficiently using the divisor lattice structure.
    
    prime_powers: list of (p, p^a) tuples for the prime factorization of n.
    We generate divisors in a structured way and count adjacent coprime pairs.
    
    Key insight: two divisors d and d' are coprime iff they share no prime factor.
    In the sorted divisor list, adjacent pairs that are coprime can be counted by
    understanding the structure.
    
    For a number with prime factorization p1^a1 * p2^a2 * ... * pk^ak,
    the divisors form a k-dimensional grid. Two adjacent divisors in sorted order
    are coprime if and only if... this is complex.
    
    Alternative approach: For small k (number of distinct primes), we can use
    the fact that the number of divisors is prod(ai+1). If this is manageable
    (< ~100000), we can generate and sort.
    """
    total_divs = 1
    for _, pa in prime_powers:
        a = 0
        p = pa
        while p > 1:
            p //= prime_powers[0][0] if len(prime_powers) == 1 else None
            break
        # Recompute exponent
        break
    
    # Just use the simple method if total divisors is small
    divs = [1]
    for p, pa in prime_powers:
        # pa = p^a, so exponent a = log_p(pa)
        # Build powers: 1, p, p^2, ..., p^a
        powers = [1]
        val = 1
        while val < pa:
            val *= p
            powers.append(val)
        new_divs = []
        for d in divs:
            for pw in powers:
                new_divs.append(d * pw)
        divs = new_divs
    
    divs.sort()
    count = 0
    for j in range(len(divs) - 1):
        if gcd(divs[j], divs[j + 1]) == 1:
            count += 1
    return count


def _tau_perp_analytic(prime_powers):
    """
    Analytic computation of tau_perp for n = prod p_i^{a_i}.
    
    The divisors sorted: we need to count adjacent pairs (d_i, d_{i+1}) with gcd=1.
    
    Key observation: In the sorted list of divisors, two consecutive divisors d and d'
    have gcd(d, d') = 1 if and only if they don't share any prime factor.
    
    For squarefree n (all a_i = 1), the divisors are all products of subsets of primes.
    The number of divisors is 2^k.
    
    Let's think about when consecutive divisors in sorted order are coprime.
    
    Actually, let me use a different approach. For the specific case where we want to
    maximize tau_perp, let's think about what structures work well.
    
    For n = p1 * p2 * ... * pk (squarefree, k primes):
    Divisors are all 2^k subset products.
    
    Consider k=2: n = p*q, p<q. Divisors: 1, p, q, pq.
    Pairs: (1,p): gcd=1, (p,q): gcd=1, (q,pq): gcd=q != 1.
    tau_perp = 2.
    
    k=3: n = p*q*r. Divisors: 1, p, q, r, pq, pr, qr, pqr (assuming p<q<r and pq<pr<qr)
    Actually need to sort properly. If p=2,q=3,r=5: divisors are 1,2,3,5,6,10,15,30.
    Pairs: (1,2)gcd1, (2,3)gcd1, (3,5)gcd1, (5,6)gcd1, (6,10)gcd2, (10,15)gcd5, (15,30)gcd15.
    tau_perp = 4.
    
    k=4: p=2,q=3,r=5,s=7. n=210. Divisors: 1,2,3,5,6,7,10,14,15,21,30,35,42,70,105,210.
    Let me check adjacent gcds:
    (1,2):1, (2,3):1, (3,5):1, (5,6):1, (6,7):1, (7,10):1, (10,14):2, (14,15):1, (15,21):3, (21,30):3, (30,35):5, (35,42):7, (42,70):14, (70,105):35, (105,210):105.
    Count of 1s: positions 1-6, 8. That's 7 pairs with gcd 1.
    tau_perp = 7.
    
    So for squarefree with k primes: tau_perp values are 2, 4, 7, ...
    
    Let me verify k=4 more carefully.
    Divisors of 210: 1,2,3,5,6,7,10,14,15,21,30,35,42,70,105,210
    Adjacent pairs and gcds:
    1-2: gcd(1,2)=1 ✓
    2-3: gcd(2,3)=1 ✓
    3-5: gcd(3,5)=1 ✓
    5-6: gcd(5,6)=1 ✓
    6-7: gcd(6,7)=1 ✓
    7-10: gcd(7,10)=1 ✓
    10-14: gcd(10,14)=2 ✗
    14-15: gcd(14,15)=1 ✓
    15-21: gcd(15,21)=3 ✗
    21-30: gcd(21,30)=3 ✗
    30-35: gcd(30,35)=5 ✗
    35-42: gcd(35,42)=7 ✗
    42-70: gcd(42,70)=14 ✗
    70-105: gcd(70,105)=35 ✗
    105-210: gcd(105,210)=105 ✗
    
    So 7 pairs with gcd 1. tau_perp(210)=7.
    
    Now, what about using exponents? If we use p^a instead of p, we get more divisors.
    But do we get more coprime adjacent pairs?
    
    Let's try n = 2^2 * 3 = 12. Divisors: 1,2,3,4,6,12.
    (1,2):1, (2,3):1, (3,4):1, (4,6):2, (6,12):6. tau_perp=3.
    Compare to squarefree 2*3=6: divisors 1,2,3,6. (1,2):1,(2,3):1,(3,6):3. tau_perp=2.
    So 12 gives 3 > 2.
    
    n = 2^2 * 3 * 5 = 60. Divisors: 1,2,3,4,5,6,10,12,15,20,30,60.
    (1,2):1,(2,3):1,(3,4):1,(4,5):1,(5,6):1,(6,10):2,(10,12):2,(12,15):3,(15,20):5,(20,30):10,(30,60):30.
    tau_perp = 5.
    
    Compare to squarefree 2*3*5=30: divisors 1,2,3,5,6,10,15,30.
    (1,2):1,(2,3):1,(3,5):1,(5,6):1,(6,10):2,(10,15):5,(15,30):15. tau_perp=4.
    
    So 60 gives 5 > 4.
    
    n = 2^2 * 3^2 = 36. Divisors: 1,2,3,4,6,9,12,18,36.
    (1,2):1,(2,3):1,(3,4):1,(4,6):2,(6,9):3,(9,12):3,(12,18):6,(18,36):18. tau_perp=3.
    
    n = 2^3 * 3 = 24. Divisors: 1,2,3,4,6,8,12,24.
    (1,2):1,(2,3):1,(3,4):1,(4,6):2,(6,8):2,(8,12):4,(12,24):12. tau_perp=3.
    
    n = 2 * 3 * 5 * 7 = 210 gives 7.
    n = 2^2 * 3 * 5 * 7 = 420. Divisors: 1,2,3,4,5,6,7,10,12,14,15,20,21,28,30,35,42,60,70,84,105,140,210,420.
    That's 24 divisors. Let me count gcd=1 pairs... this is getting complex. Let me just compute it in code.
    
    The strategy should be:
    1. Search over structures: which primes to use, what exponents.
    2. For each candidate, compute tau_perp exactly.
    3. Find the maximum.
    
    Given the 150s limit, we need to be smart. The key insight from the knowledge is that
    extending to k=15 primes with greedy swapping improved from 15.84 to 21.16.
    
    Let me think about what the optimal structure looks like.
    
    For squarefree n with k primes, the number of divisors is 2^k, and tau_perp seems to grow
    roughly like O(k * 2^(k-2)) or similar. But n = product of first k primes grows as e^(k log k),
    so for n <= 10^18, k is limited.
    
    Product of first 15 primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    Let me estimate: this is the 15th primorial. 
    2*3=6, *5=30, *7=210, *11=2310, *13=30030, *17=510510, *19=9699690, *23=223092870, *29=6469693230, *31=200560490130, *37=7420738134810, *41=304250263527210, *43=13082761331670030, *47=614889782588491410.
    That's about 6.15 * 10^17, which is < 10^18. So k=15 squarefree is feasible.
    
    The 16th prime is 53: 614889782588491410 * 53 = 3.259 * 10^19 > 10^18. So max k for squarefree is 15.
    
    But we can also use exponents. The question is whether using exponents (making n larger in terms of divisor count) helps increase tau_perp.
    
    Let me think about this differently. The score is tau_perp(n), and we want to maximize it.
    
    Let me just do a search: try various combinations of primes and exponents, compute tau_perp exactly, and keep the best.
    """
    # Generate divisors and count
    divs = [1]
    for p, pa in prime_powers:
        powers = [1]
        val = 1
        while val < pa:
            val *= p
            powers.append(val)
        new_divs = []
        for d in divs:
            for pw in powers:
                new_divs.append(d * pw)
        divs = new_divs
    
    divs.sort()
    count = 0
    for j in range(len(divs) - 1):
        if gcd(divs[j], divs[j + 1]) == 1:
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


def _sieve_primes(limit):
    """Generate all primes up to limit using sieve."""
    if limit < 2:
        return []
    is_prime_arr = bytearray(b'\x01') * (limit + 1)
    is_prime_arr[0] = 0
    is_prime_arr[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_arr[j] = 0
    return [i for i in range(2, limit + 1) if is_prime_arr[i]]


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # First, compute the score for the seed answer
    seed_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    seed_score = _tau_perp_small(seed_n)
    
    best_n = seed_n
    best_score = seed_score
    
    # Strategy: search over combinations of primes and exponents
    # Key insight: we want to maximize tau_perp, which benefits from:
    # 1. Many distinct primes (more divisors, more potential coprime pairs)
    # 2. Small primes (to keep n <= 10^18)
    # 3. Possibly some exponents to increase divisor count
    
    # Let's first try squarefree numbers with various prime sets
    # Then try adding exponents to the smallest primes
    
    # Generate primes up to a reasonable limit
    # The 15th prime is 47, 16th is 53, etc.
    # For squarefree, we need product <= 10^18
    # Let's generate primes up to 200 to have options
    primes = _sieve_primes(200)
    
    # Approach 1: Squarefree with k primes (k from 2 to 15)
    # For each k, try different sets of k primes
    # The best set for a given k is likely the k smallest primes, but let's verify
    
    # Actually, for squarefree, the tau_perp depends on the specific primes used
    # because the sorting of divisors depends on the actual values.
    
    # Let's try: for k from 2 to 15, use the k smallest primes
    for k in range(2, 16):
        if time.time() > deadline:
            break
        if k > len(primes):
            break
        # Use first k primes
        selected = primes[:k]
        n = 1
        for p in selected:
            n *= p
        if n > 10**18:
            break
        score = _tau_perp_small(n)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Approach 2: Try replacing larger primes with smaller ones or adjusting
    # Actually, for squarefree, using the smallest primes is likely optimal
    # because it allows more primes to fit under 10^18
    
    # Approach 3: Add exponents to small primes
    # Try n = 2^a * 3^b * 5^c * ... with various exponents
    # The product must be <= 10^18
    
    # Let's try a grid search over exponents for the first few primes
    # Start with the structure that gave the best score so far
    
    # Try: 2^a * 3^b * 5^c * 7^d * ... 
    # For each combination, check if product <= 10^18 and compute score
    
    # This is a large search space, so let's be strategic
    # Try exponents 1-4 for the first 6-8 primes
    
    # Actually, let's think about what's likely optimal.
    # From the knowledge, k=15 with greedy swapping gave 21.16.
    # The seed (8 primes) gives some baseline.
    
    # Let me compute the score for the 15-prime squarefree number
    n15 = 1
    for p in primes[:15]:
        n15 *= p
    if n15 <= 10**18:
        score15 = _tau_perp_small(n15)
        if score15 > best_score:
            best_score = score15
            best_n = n15
    
    # Now try adding exponents to the smallest primes while keeping n <= 10^18
    # Start from the 15-prime squarefree and see if we can increase exponents
    
    # Actually, if we increase an exponent, n gets larger, so we might need to drop some primes
    # Let's try: for each of the first m primes, try exponent 2 or 3, and see if we can
    # still fit enough primes
    
    # Strategy: for a target number of distinct primes k, distribute the "budget" (log n)
    # among exponents to maximize tau_perp
    
    # Let's try a more systematic search:
    # For k from 2 to 15:
    #   For the k smallest primes, try assigning exponents [1,1,...,1] first
    #   Then try increasing exponents of the smallest primes
    
    # But this is still a large search. Let's focus on promising regions.
    
    # Key insight: tau_perp benefits from many divisors that are "spread out" in the
    # prime factor space. Squarefree numbers with many primes are good because each
    # divisor corresponds to a unique subset of primes.
    
    # Let me try: for k=10 to 15, try using the k smallest primes with the first few
    # having exponent 2
    
    for k in range(8, 16):
        if time.time() > deadline:
            break
        if k > len(primes):
            break
        base_primes = primes[:k]
        base_n = 1
        for p in base_primes:
            base_n *= p
        if base_n > 10**18:
            continue
        
        # Try increasing exponent of first prime
        for a1 in range(1, 5):
            if time.time() > deadline:
                break
            n = base_n * (2 ** (a1 - 1))  # 2^a1 instead of 2^1
            if n > 10**18:
                break
            # Build prime powers
            pp = []
            for i, p in enumerate(base_primes):
                if i == 0:
                    pp.append((p, p ** a1))
                else:
                    pp.append((p, p))
            score = _tau_perp_analytic(pp)
            if score > best_score:
                best_score = score
                best_n = n
        
        # Try increasing exponent of second prime
        for a2 in range(1, 5):
            if time.time() > deadline:
                break
            n = base_n * (3 ** (a2 - 1))
            if n > 10**18:
                break
            pp = []
            for i, p in enumerate(base_primes):
                if i == 1:
                    pp.append((p, p ** a2))
                else:
                    pp.append((p, p))
            score = _tau_perp_analytic(pp)
            if score > best_score:
                best_score = score
                best_n = n
        
        # Try increasing exponents of first two primes
        for a1 in range(1, 4):
            for a2 in range(1, 4):
                if time.time() > deadline:
                    break
                n = base_n * (2 ** (a1 - 1)) * (3 ** (a2 - 1))
                if n > 10**18:
                    break
                pp = []
                for i, p in enumerate(base_primes):
                    if i == 0:
                        pp.append((p, p ** a1))
                    elif i == 1:
                        pp.append((p, p ** a2))
                    else:
                        pp.append((p, p))
                score = _tau_perp_analytic(pp)
                if score > best_score:
                    best_score = score
                    best_n = n
    
    # Try a broader search: random prime sets and exponent assignments
    import random
    random.seed(42)
    
    # Generate random candidates
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        # Random number of distinct primes
        k = random.randint(3, 14)
        
        # Choose k random primes from the first 30 primes
        # Bias towards smaller primes
        available = primes[:30]
        selected = random.sample(available, min(k, len(available)))
        selected.sort()
        
        # Assign random exponents (mostly 1, sometimes 2 or 3)
        exponents = []
        n = 1
        valid = True
        for i, p in enumerate(selected):
            if i < 3:
                a = random.choice([1, 1, 1, 2, 2, 3])
            else:
                a = random.choice([1, 1, 1, 1, 2])
            exponents.append(a)
            n *= p ** a
            if n > 10**18:
                valid = False
                break
        
        if not valid or n < 2:
            continue
        
        pp = [(p, p ** a) for p, a in zip(selected, exponents)]
        score = _tau_perp_analytic(pp)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Final: try some hand-crafted structures
    # 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h
    # with product <= 10^18
    
    # Try a few specific patterns
    patterns = [
        # (exponents for 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47)
        [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [3,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [2,2,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [2,1,2,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,2,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,2,1,1,1,1,1,1,1,1,1,1,1,1],
    ]
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        n = 1
        pp = []
        valid = True
        for i, a in enumerate(pattern):
            if i >= len(primes):
                break
            p = primes[i]
            n *= p ** a
            if n > 10**18:
                valid = False
                break
            pp.append((p, p ** a))
        
        if not valid or n < 2:
            continue
        
        score = _tau_perp_analytic(pp)
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
