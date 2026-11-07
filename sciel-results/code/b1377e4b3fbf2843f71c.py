import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 55.0  # Safety margin before 60s limit

def _get_prime_exponents(n_max):
    """Get prime factorization vectors for all numbers 2..n_max."""
    # Sieve for primes up to n_max
    is_prime = [True] * (n_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n_max + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n_max + 1) if is_prime[i]]
    
    vecs = {}
    for x in range(2, n_max + 1):
        v = [0] * len(primes)
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[x] = tuple(v)
    return primes, vecs

def _can_add(vec, sums):
    """Check if adding vec to the set of subset-sums maintains distinctness."""
    for s in sums:
        new_s = tuple(a + b for a, b in zip(s, vec))
        if new_s in sums:
            return False
    return True

def _add_to_sums(sums, vec):
    """Return new set of subset sums after adding vec."""
    new_sums = set(sums)
    for s in sums:
        new_sums.add(tuple(a + b for a, b in zip(s, vec)))
    return new_sums

def _greedy(candidates, vecs, primes):
    """Greedy: add candidates in given order if possible."""
    zero = tuple([0] * len(primes))
    sums = {zero}
    selected = []
    for x in candidates:
        if _can_add(vecs[x], sums):
            sums = _add_to_sums(sums, vecs[x])
            selected.append(x)
    return selected, sums

def _local_search(best_set, vecs, primes, deadline):
    """Try to improve by swapping elements out and in."""
    if not best_set:
        return best_set
    
    zero = tuple([0] * len(primes))
    current_set = list(best_set)
    current_sums = {zero}
    for x in current_set:
        current_sums = _add_to_sums(current_sums, vecs[x])
    
    candidates = list(range(2, 41))
    in_set = set(current_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each element in current_set to swap out
        for i, x_out in enumerate(list(current_set)):
            if time.time() > deadline:
                break
            # Remove x_out
            temp_set = current_set[:i] + current_set[i+1:]
            temp_sums = {zero}
            for x in temp_set:
                temp_sums = _add_to_sums(temp_sums, vecs[x])
            
            # Try adding each candidate not in current_set
            for x_in in candidates:
                if x_in in in_set:
                    continue
                if time.time() > deadline:
                    break
                if _can_add(vecs[x_in], temp_sums):
                    # This swap is valid
                    new_set = temp_set + [x_in]
                    new_sums = _add_to_sums(temp_sums, vecs[x_in])
                    # Check if this is better (same size but different composition might help future swaps)
                    # For now, accept any valid swap to explore
                    current_set = new_set
                    current_sums = new_sums
                    in_set = set(current_set)
                    improved = True
                    break
            if time.time() > deadline:
                break
    
    return current_set

def construct():
    start_time = time.time()
    deadline = _DEADLINE
    
    n_max = 40
    primes, vecs = _get_prime_exponents(n_max)
    candidates = list(range(2, n_max + 1))
    
    # Generate multiple orderings
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    primes_list = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    orders = [
        primes_list + composites,
        composites + primes_list,
        sorted(candidates, key=lambda x: (num_prime_factors(x), x)),
        sorted(candidates, key=lambda x: (-num_prime_factors(x), x)),
        sorted(candidates, key=lambda x: x),
        sorted(candidates, key=lambda x: -x),
    ]
    
    # Add some random orders
    for _ in range(20):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    best_set = []
    best_sums = None
    
    for order in orders:
        if time.time() > deadline:
            break
        result_set, result_sums = _greedy(order, vecs, primes)
        if len(result_set) > len(best_set):
            best_set = result_set
            best_sums = result_sums
    
    # Local search to try to improve
    if time.time() < deadline - 5:
        best_set = _local_search(best_set, vecs, primes, deadline)
    
    # Final check: try adding any remaining candidates
    if time.time() < deadline - 2:
        zero = tuple([0] * len(primes))
        final_sums = {zero}
        final_set = list(best_set)
        in_set = set(final_set)
        for x in final_set:
            final_sums = _add_to_sums(final_sums, vecs[x])
        
        for x in candidates:
            if x not in in_set and time.time() < deadline - 1:
                if _can_add(vecs[x], final_sums):
                    final_set.append(x)
                    final_sums = _add_to_sums(final_sums, vecs[x])
                    in_set.add(x)
        
        if len(final_set) > len(best_set):
            best_set = final_set
    
    # Ensure correct types
    return [int(x) for x in best_set]
