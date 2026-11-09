import math
import time
import random
import heapq

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute small primes
    def _sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]

    primes = _sieve(300)
    odd_primes = primes[1:]  # 3, 5, 7, ...

    def _score_gray_code(odd_primes_set, a):
        """
        Score calculation using Gray-code traversal of divisor lattice.
        n = 2^a * prod(p_i) where p_i are odd primes (squarefree odd part).
        
        Divisors are 2^j * d where j in [0,a], d is product of subset of odd primes.
        We traverse in a specific order to count coprime adjacencies without sorting.
        
        Key insight: The sorted order of divisors can be understood by their binary
        representation if we use a Gray-code-like ordering that respects numerical order.
        
        Actually, for squarefree odd part with primes p_1 < p_2 < ... < p_k,
        the divisors in increasing order have a complex structure. But we can use
        the fact that adjacent divisors in sorted order differ by small multiplicative
        factors.
        
        Alternative approach: Use the structure that divisors of 2^a * m (m squarefree)
        form a grid. We can count coprime pairs by analyzing transitions between
        "layers" (different powers of 2) and within layers.
        
        Let's use a different structural insight:
        For n = 2^a * m where m is squarefree with k primes, there are (a+1)*2^k divisors.
        Two consecutive divisors d_i, d_{i+1} in sorted order are coprime iff:
        - They share no common prime factor.
        
        We can count this by considering all pairs of divisors that could be adjacent.
        But that's still expensive.
        
        Better: Use the fact that in the sorted divisor list, the sequence has a
        regular structure. For small k (<=15), we can generate divisors more efficiently
        and use a faster comparison.
        """
        if not odd_primes_set:
            # n = 2^a, divisors: 1, 2, 4, ..., 2^a
            # Coprime pairs: only (1,2) since gcd(2^i, 2^j) = 2^min(i,j) > 1 for i,j >= 1
            return 1 if a >= 1 else 0
        
        k = len(odd_primes_set)
        ps = sorted(odd_primes_set)
        
        # Generate all odd divisors (subsets of primes)
        # Use bit representation
        num_odd_divs = 1 << k
        
        # For each odd divisor d, we have divisors 2^j * d for j = 0..a
        # We need to merge these a+1 lists in sorted order.
        
        # Precompute odd divisors with their prime factor masks
        odd_divs = []  # (value, mask) where mask indicates which primes are included
        for mask in range(num_odd_divs):
            val = 1
            for i in range(k):
                if mask & (1 << i):
                    val *= ps[i]
            odd_divs.append((val, mask))
        
        # Sort by value
        odd_divs.sort(key=lambda x: x[0])
        
        # Now merge a+1 lists: list_j = [2^j * d for d in odd_divs]
        # Use heap to merge
        heap = []
        for j in range(a + 1):
            if odd_divs:
                heapq.heappush(heap, (2**j * odd_divs[0][0], j, 0, odd_divs[0][1]))
        
        count = 0
        prev_j = None
        prev_mask = None
        
        while heap:
            val, j, idx, mask = heapq.heappop(heap)
            
            if prev_j is not None:
                # Check if coprime: gcd(2^prev_j * prev_d, 2^j * d) == 1
                # This requires: min(prev_j, j) == 0 AND gcd(prev_d, d) == 1
                # gcd(prev_d, d) == 1 iff (prev_mask & mask) == 0
                if min(prev_j, j) == 0 and (prev_mask & mask) == 0:
                    count += 1
            
            # Push next from same list j
            if idx + 1 < num_odd_divs:
                next_val = 2**j * odd_divs[idx + 1][0]
                heapq.heappush(heap, (next_val, j, idx + 1, odd_divs[idx + 1][1]))
            
            prev_j = j
            prev_mask = mask
        
        return count

    best_n = 2
    best_score = 0

    # Phase 1: Exhaustive search over squarefree odd parts with consecutive primes
    prefix_prod = [1]
    for p in odd_primes:
        prefix_prod.append(prefix_prod[-1] * p)
        
    max_k = 0
    for i, prod in enumerate(prefix_prod):
        if prod <= 10**18:
            max_k = i
        else:
            break

    for k in range(0, min(max_k + 1, 16)):
        if time.time() > deadline - 15:
            break
            
        m = prefix_prod[k]
        if m > 10**18:
            break
            
        max_a = int(math.log2(10**18 / m)) if m > 0 else 60
        
        if k == 0:
            if max_a >= 1:
                n_cand = 2
                s = 1
                if s > best_score:
                    best_score = s
                    best_n = n_cand
            continue
        
        # Try different values of a
        for a in range(1, min(max_a + 1, 61)):
            if time.time() > deadline - 15:
                break
                
            n_cand = (2 ** a) * m
            if n_cand > 10**18:
                break
                
            prime_set = set(odd_primes[:k])
            s = _score_gray_code(prime_set, a)
            
            if s > best_score:
                best_score = s
                best_n = n_cand

    # Phase 2: Local search with prime swaps
    def _factorize(n):
        temp = n
        factors = {}
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors[p] = exp
            p += 1 if p == 2 else 2
        if temp > 1:
            factors[temp] = 1
        return factors

    def _rebuild(factors):
        n = 1
        for p, e in factors.items():
            n *= (p ** e)
        return n

    current_factors = _factorize(best_n)
    current_score = best_score
    current_n = best_n
    
    # Candidate primes for swapping
    candidate_primes = [p for p in primes if p > 2 and p < 300]
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        if time.time() > deadline - 5:
            break
            
        move_type = random.random()
        new_factors = dict(current_factors)
        
        if move_type < 0.6:
            # Swap a prime
            odd_ps = [p for p in new_factors.keys() if p != 2]
            if not odd_ps:
                continue
            p_out = random.choice(odd_ps)
            e_out = new_factors[p_out]
            
            if e_out == 1:
                del new_factors[p_out]
            else:
                new_factors[p_out] = e_out - 1
                
            p_in = random.choice(candidate_primes)
            if p_in == p_out:
                continue
            new_factors[p_in] = new_factors.get(p_in, 0) + 1
            
        elif move_type < 0.8:
            # Change exponent of 2
            a = new_factors.get(2, 0)
            if a == 0:
                new_factors[2] = 1
            elif random.random() < 0.5:
                new_factors[2] = a - 1
                if new_factors[2] == 0:
                    del new_factors[2]
            else:
                new_factors[2] = a + 1
        else:
            # Add or remove a random prime
            if random.random() < 0.5 and len([p for p in new_factors if p != 2]) < 15:
                p_in = random.choice(candidate_primes)
                new_factors[p_in] = new_factors.get(p_in, 0) + 1
            elif len([p for p in new_factors if p != 2]) > 1:
                p_out = random.choice([p for p in new_factors if p != 2])
                if new_factors[p_out] == 1:
                    del new_factors[p_out]
                else:
                    new_factors[p_out] -= 1

        if not new_factors:
            continue
            
        n_new = _rebuild(new_factors)
        
        if n_new < 2 or n_new > 10**18:
            continue
            
        if n_new == current_n:
            continue
            
        # Compute score
        a = new_factors.get(2, 0)
        prime_set = set(p for p in new_factors.keys() if p != 2)
        
        # Only use squarefree assumption for speed; if exponents > 1, we need different handling
        # For now, assume squarefree odd part (exponents = 1)
        has_non_squarefree = any(new_factors[p] > 1 for p in prime_set)
        
        if has_non_squarefree:
            # Fall back to slower method for non-squarefree
            # Generate all divisors
            divs = [1]
            for p, e in sorted(new_factors.items()):
                new_divs = []
                p_powers = [1]
                for _ in range(e):
                    p_powers.append(p_powers[-1] * p)
                for d in divs:
                    for pp in p_powers:
                        new_divs.append(d * pp)
                divs = new_divs
            
            divs.sort()
            s_new = 0
            for i in range(len(divs) - 1):
                if math.gcd(divs[i], divs[i+1]) == 1:
                    s_new += 1
        else:
            s_new = _score_gray_code(prime_set, a)
        
        if s_new > current_score:
            current_score = s_new
            current_n = n_new
            current_factors = new_factors
            if current_score > best_score:
                best_score = current_score
                best_n = current_n

    return best_n

# EVOLVE-BLOCK-END
