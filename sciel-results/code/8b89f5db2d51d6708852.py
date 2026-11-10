import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000000)
    prime_set = set(primes)
    
    # 2. Optimized tau_perp with bitmask-based gcd check
    def tau_perp_fast(prime_factors):
        # For squarefree n, each divisor corresponds to a subset of primes
        # Represent each divisor by its bitmask of which primes divide it
        # Two divisors are coprime iff their bitmasks are disjoint
        
        k = len(prime_factors)
        num_divs = 1 << k
        
        # Build list of (value, bitmask) for all divisors
        divs = []
        for mask in range(num_divs):
            val = 1
            for i in range(k):
                if mask & (1 << i):
                    val *= prime_factors[i]
            divs.append((val, mask))
        
        # Sort by value
        divs.sort()
        
        # Count adjacent coprime pairs using bitmask AND
        score = 0
        for i in range(num_divs - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                score += 1
        return score

    # 3. Heuristic: score estimate based on prime spacing
    # Smaller primes tend to create more "interleaving" in the sorted divisor list
    # which increases coprime adjacencies. Use log-spacing as a proxy.
    def score_heuristic(prime_factors):
        # Sort primes
        ps = sorted(prime_factors)
        # Reward having many small primes
        # Penalize large gaps between consecutive primes
        score = 0.0
        for i in range(len(ps)):
            score += 1.0 / math.log(ps[i] + 1)
        # Bonus for dense small primes
        if len(ps) >= 10:
            score += 0.5 * (len(ps) - 10)
        return score

    # 4. Initialize with product of smallest 15 primes
    max_k = 15
    current_primes = list(primes[:max_k])
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    best_n = current_n
    best_score = tau_perp_fast(current_primes)
    current_score = best_score
    
    # 5. Local search with multiple strategies
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        current_prime_set = set(current_primes)
        
        # Strategy 1: Single prime swap (original approach, optimized)
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                return best_n
                
            p_old = current_primes[idx]
            max_p_new = (10**18 * p_old) // current_n
            
            # Try a limited number of candidates: smallest available primes first
            # plus some random larger ones
            candidates = []
            for p_new in primes:
                if p_new in current_prime_set:
                    continue
                if p_new > max_p_new:
                    break
                candidates.append(p_new)
                if len(candidates) >= 50:  # Limit to 50 candidates per slot
                    break
            
            # Also add some random larger primes within range
            if max_p_new > 10000:
                for _ in range(10):
                    p_rand = random.randint(10000, min(max_p_new, 1000000))
                    # Find nearest prime >= p_rand
                    for p in primes:
                        if p >= p_rand and p <= max_p_new and p not in current_prime_set:
                            candidates.append(p)
                            break
            
            best_candidate = None
            best_candidate_score = current_score - 1
            
            for p_new in candidates:
                if p_new in current_prime_set:
                    continue
                if p_new > max_p_new:
                    continue
                
                new_primes = current_primes.copy()
                new_primes[idx] = p_new
                
                # Quick heuristic filter
                if score_heuristic(new_primes) < score_heuristic(current_primes) - 0.5:
                    continue
                
                s_new = tau_perp_fast(new_primes)
                
                if s_new > best_candidate_score:
                    best_candidate_score = s_new
                    best_candidate = p_new
            
            if best_candidate is not None:
                current_primes[idx] = best_candidate
                current_n = (current_n // p_old) * best_candidate
                current_score = best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
                improved = True
                break
        
        # Strategy 2: Pair swap (exchange two primes)
        if not improved and time.time() < deadline - 20.0:
            pair_indices = list(combinations(range(len(current_primes)), 2))
            random.shuffle(pair_indices)
            
            for i, j in pair_indices[:20]:  # Try up to 20 random pairs
                if time.time() > deadline - 10.0:
                    break
                    
                p_i = current_primes[i]
                p_j = current_primes[j]
                base = current_n // (p_i * p_j)
                
                # Try swapping p_i with a smaller prime and p_j with a larger prime
                # or both with different primes
                candidates_i = []
                candidates_j = []
                
                for p in primes:
                    if p in current_prime_set:
                        continue
                    if p * base * p_j <= 10**18:
                        candidates_i.append(p)
                        if len(candidates_i) >= 20:
                            break
                
                for p in primes:
                    if p in current_prime_set:
                        continue
                    if p * base * p_i <= 10**18:
                        candidates_j.append(p)
                        if len(candidates_j) >= 20:
                            break
                
                best_pair = None
                best_pair_score = current_score - 1
                
                for p_new_i in candidates_i[:10]:
                    for p_new_j in candidates_j[:10]:
                        if p_new_i == p_new_j:
                            continue
                        if p_new_i in current_prime_set or p_new_j in current_prime_set:
                            continue
                        
                        new_primes = current_primes.copy()
                        new_primes[i] = p_new_i
                        new_primes[j] = p_new_j
                        
                        s_new = tau_perp_fast(new_primes)
                        
                        if s_new > best_pair_score:
                            best_pair_score = s_new
                            best_pair = (p_new_i, p_new_j)
                
                if best_pair is not None:
                    current_primes[i] = best_pair[0]
                    current_primes[j] = best_pair[1]
                    current_n = base * best_pair[0] * best_pair[1]
                    current_score = best_pair_score
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                    improved = True
                    break

    # 6. Final polish: try removing one prime to see if score increases
    # (sometimes having fewer primes with better spacing can improve the score)
    if time.time() < deadline - 5.0:
        for idx in range(len(current_primes)):
            if time.time() > deadline - 5.0:
                break
            p_old = current_primes[idx]
            # Try removing this prime
            new_primes = [p for i, p in enumerate(current_primes) if i != idx]
            new_n = current_n // p_old
            if new_n >= 2 and new_n <= 10**18:
                s_new = tau_perp_fast(new_primes)
                if s_new > best_score:
                    best_score = s_new
                    best_n = new_n
    
    return best_n
