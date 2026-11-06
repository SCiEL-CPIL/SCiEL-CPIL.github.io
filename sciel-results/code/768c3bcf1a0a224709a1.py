"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def _is_p_smooth(n, primes_set, primes_list):
    """Check if n is P-smooth where P is the set of primes in primes_set."""
    temp = n
    for p in primes_list:
        if p * p > temp:
            break
        while temp % p == 0:
            if p not in primes_set:
                return False
            temp //= p
    if temp > 1:
        if temp not in primes_set:
            return False
    return True

def _greedy_clique(vertices, compatible, size):
    """Greedy maximal clique heuristic: pick vertices one by one that are
    compatible with all previously picked ones. Returns list of picked vertices."""
    selected = []
    for v in vertices:
        if len(selected) >= size:
            break
        ok = True
        for u in selected:
            if not compatible[u][v]:
                ok = False
                break
        if ok:
            selected.append(v)
    return selected

def construct():
    """Find 12 distinct positive integers <= 10000 minimizing the number of
    distinct primes dividing all pairwise sums of distinct elements."""
    
    N = 12
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL  # 20000
    
    all_primes = _sieve_primes(MAX_SUM)
    
    # Quick check: if all numbers are even, all sums are even,
    # so only prime factor is 2. This gives 1 distinct prime (optimal).
    # Verify this works:
    even_candidates = list(range(2, 2 * N + 1, 2))  # 2,4,6,...,24
    if len(even_candidates) >= N:
        # All pairwise sums of even numbers are even, hence 2-smooth.
        return even_candidates[:N]
    
    # Graph-based approach: for a prime set P, build compatibility graph
    # where edge (u,v) exists if u+v is P-smooth, then find clique of size N.
    
    # Try prime sets in increasing size
    for num_primes in range(1, 10):
        if num_primes > len(all_primes):
            break
        
        # Try the first num_primes primes (smallest, most likely to work)
        primes_tuple = tuple(all_primes[:num_primes])
        primes_set = set(primes_tuple)
        primes_list = list(primes_tuple)
        
        # Precompute which sums are P-smooth
        smooth = [False] * (MAX_SUM + 1)
        for s in range(2, MAX_SUM + 1):
            smooth[s] = _is_p_smooth(s, primes_set, primes_list)
        
        # Build compatibility: for each vertex u, which vertices v are compatible?
        # We'll use a dict-of-sets or just check on the fly in greedy.
        # For greedy, we need fast lookup: compatible[u][v]
        # But 10000x10000 is too big. Instead, in greedy we just check smooth[u+v].
        
        # Greedy: iterate through vertices in order, pick if compatible with all selected
        # Use a smarter ordering: sort by how many smooth-sum partners they have
        # But that's expensive. Just try natural order first.
        
        selected = []
        for a in range(1, MAX_VAL + 1):
            if len(selected) >= N:
                break
            ok = True
            for b in selected:
                s = a + b
                if s > MAX_SUM or not smooth[s]:
                    ok = False
                    break
            if ok:
                selected.append(a)
        
        if len(selected) >= N:
            # Verify
            result = selected[:N]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    s = result[i] + result[j]
                    if not smooth[s]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return result
        
        # Try a few alternative prime sets
        if num_primes <= 5:
            for swap_idx in range(num_primes, min(num_primes + 3, len(all_primes))):
                alt_tuple = tuple(all_primes[:num_primes-1] + [all_primes[swap_idx]])
                if alt_tuple == primes_tuple:
                    continue
                alt_set = set(alt_tuple)
                alt_list = list(alt_tuple)
                
                smooth2 = [False] * (MAX_SUM + 1)
                for s in range(2, MAX_SUM + 1):
                    smooth2[s] = _is_p_smooth(s, alt_set, alt_list)
                
                selected2 = []
                for a in range(1, MAX_VAL + 1):
                    if len(selected2) >= N:
                        break
                    ok = True
                    for b in selected2:
                        s = a + b
                        if s > MAX_SUM or not smooth2[s]:
                            ok = False
                            break
                    if ok:
                        selected2.append(a)
                
                if len(selected2) >= N:
                    result = selected2[:N]
                    valid = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            s = result[i] + result[j]
                            if not smooth2[s]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return result
    
    # Fallback: all even numbers
    return list(range(2, 2 * N + 1, 2))

# EVOLVE-BLOCK-END
