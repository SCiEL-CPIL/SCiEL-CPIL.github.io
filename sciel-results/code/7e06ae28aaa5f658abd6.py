import math
import time
import random
import heapq

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute small primes
    def _sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]

    primes = _sieve(1000)
    odd_primes = primes[1:]  # 3, 5, 7, ...

    def _fast_score(factors):
        """
        Optimized score calculation for n = 2^a * prod(p_i^e_i).
        """
        if not factors:
            return 0
            
        a = factors.get(2, 0)
        odd_factors = {p: e for p, e in factors.items() if p != 2}
        
        # Generate divisors of odd part
        divs_odd = [1]
        for p, e in odd_factors.items():
            new_divs = []
            p_powers = [1]
            for _ in range(e):
                p_powers.append(p_powers[-1] * p)
            for d in divs_odd:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divs_odd = new_divs
            
        divs_odd.sort()
        
        # Now we have a+1 lists: L_j = [2^j * d for d in divs_odd]
        # We need to merge them and count coprime adjacencies.
        # Using a min-heap to merge.
        
        # Each element in heap: (value, j, index_in_divs_odd)
        heap = []
        for j in range(a + 1):
            # First element of list j is 2^j * 1
            heapq.heappush(heap, (2**j, j, 0))
            
        count = 0
        prev_j = None
        prev_d = None
        
        while heap:
            val, j, idx = heapq.heappop(heap)
            d = divs_odd[idx]
            
            if prev_j is not None:
                # Check if coprime
                if min(prev_j, j) == 0:
                    # One of them is odd (j=0)
                    # We need gcd(prev_d, d) == 1
                    if math.gcd(prev_d, d) == 1:
                        count += 1
            
            # Push next from same list j
            if idx + 1 < len(divs_odd):
                next_d = divs_odd[idx + 1]
                next_val = (2**j) * next_d
                heapq.heappush(heap, (next_val, j, idx + 1))
                
            prev_j = j
            prev_d = d
            
        return count

    best_n = 2
    best_score = 0

    # Phase 1: Exhaustive search over "canonical" structures: n = 2^a * prod(first k odd primes)
    # We assume exponents of odd primes are 1 initially, as higher exponents increase n
    # without adding as many "new" odd divisors relative to the cost.
    # Actually, squarefree odd part is optimal for maximizing number of divisors for a given n?
    # No, n = p^k has k+1 divisors. n = p*q has 4 divisors.
    # For a fixed product bound, squarefree maximizes the number of distinct prime factors,
    # which maximizes the number of subsets (odd divisors).
    # So we assume odd part is squarefree.
    
    prefix_prod = [1]
    for p in odd_primes:
        prefix_prod.append(prefix_prod[-1] * p)
        
    max_k = 0
    for i, prod in enumerate(prefix_prod):
        if prod <= 10**18:
            max_k = i
        else:
            break

    # We will search k from 0 to max_k, and for each k, scan a.
    for k in range(0, max_k + 1):
        if time.time() > deadline - 10:
            break
            
        m = prefix_prod[k]
        if m > 10**18:
            break
            
        max_a = int(math.log2(10**18 / m))
        
        # For k=0, n=2^a. Score is 1 (pair 1,2).
        if k == 0:
            if max_a >= 1:
                n_cand = 2
                s = 1
                if s > best_score:
                    best_score = s
                    best_n = n_cand
            continue
            
        # For k >= 1
        # Scan a.
        for a in range(1, max_a + 1):
            if time.time() > deadline - 10:
                break
                
            n_cand = (2 ** a) * m
            if n_cand > 10**18:
                break
                
            # Use fast score
            factors = {2: a}
            # m is product of first k odd primes
            for i in range(1, k+1):
                factors[odd_primes[i-1]] = 1
                
            s = _fast_score(factors)
            
            if s > best_score:
                best_score = s
                best_n = n_cand

    # Phase 2: Local Search / Random Perturbation
    # The optimal n might not use the first k primes.
    # Swapping a prime for a slightly larger one might change the divisor ordering.
    # However, smaller primes generally keep divisors closer together,
    # which might help or hurt coprime adjacency.
    # Let's try swapping the largest prime in the set with a slightly larger one.
    
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
    
    # Candidate primes for swapping: primes up to 200
    candidate_primes = [p for p in primes if p > 2 and p < 200]
    
    steps = 0
    max_steps = 10000
    
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        if time.time() > deadline - 5:
            break
            
        move_type = random.random()
        new_factors = dict(current_factors)
        
        if move_type < 0.7:
            # Swap a prime
            odd_ps = [p for p in new_factors.keys() if p != 2]
            if not odd_ps:
                continue
            # Pick the largest odd prime to swap out? Or random?
            # Swapping the largest might allow a larger prime in, keeping product similar.
            # Or swapping a small one might allow a larger structure?
            # Let's try swapping a random odd prime.
            p_out = random.choice(odd_ps)
            e_out = new_factors[p_out]
            
            # Remove p_out
            if e_out == 1:
                del new_factors[p_out]
            else:
                new_factors[p_out] = e_out - 1
                
            # Add a new prime
            p_in = random.choice(candidate_primes)
            if p_in == p_out:
                continue
            new_factors[p_in] = new_factors.get(p_in, 0) + 1
            
        else:
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

        if not new_factors:
            continue
            
        n_new = _rebuild(new_factors)
        
        if n_new < 2 or n_new > 10**18:
            continue
            
        if n_new == current_n:
            continue
            
        s_new = _fast_score(new_factors)
        
        if s_new > current_score:
            current_score = s_new
            current_n = n_new
            current_factors = new_factors
            if current_score > best_score:
                best_score = current_score
                best_n = current_n

    return best_n

# EVOLVE-BLOCK-END
