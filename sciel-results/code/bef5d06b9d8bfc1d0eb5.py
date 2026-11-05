import math
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    LIMIT = 10**18
    
    # Primes up to 97
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    prime_set = set(primes)
    
    def factorize(n):
        temp = n
        factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        return factors
    
    def calc_score(n):
        if n < 2:
            return 0
        factors = factorize(n)
        num_primes = len(factors)
        
        # Generate divisors with bitmask of which primes divide them
        # divs: list of (divisor_value, bitmask)
        divs = [(1, 0)]
        for i, (p, exp) in enumerate(factors):
            bit = 1 << i
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d_val, d_mask in divs:
                    new_divs.append((d_val * power, d_mask | bit))
            divs += new_divs
        
        divs.sort(key=lambda x: x[0])
        
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    # Phase 1: Generate candidates with high divisor count (tau)
    # We use a max-heap to keep the top K candidates by tau
    candidates = []
    K = 500 # Keep 500 top tau candidates
    
    def dfs(idx, n, max_exp, div_count):
        if time.time() > deadline - 20.0:
            return
        if idx >= len(primes):
            if len(candidates) < K:
                heapq.heappush(candidates, (-div_count, n))
            else:
                if div_count > -candidates[0][0]:
                    heapq.heapreplace(candidates, (-div_count, n))
            return
        
        p = primes[idx]
        curr = n
        exp = 0
        while curr <= LIMIT:
            exp += 1
            if exp > max_exp:
                break
            dfs(idx + 1, curr, exp, div_count * (exp + 1))
            curr *= p
            if curr > LIMIT:
                break

    # Start DFS
    # Max exponent for 2 is 60
    dfs(0, 1, 60, 1)
    
    # Extract candidates
    top_candidates = []
    while candidates:
        neg_tau, n = heapq.heappop(candidates)
        top_candidates.append(n)
        
    # Phase 2: Score the top candidates
    best_n = 2
    best_score = 0
    seen = set()
    
    # Evaluate primorial of first 15 primes as a strong baseline
    primorial = 1
    for p in primes[:15]:
        primorial *= p
        if primorial > LIMIT:
            primorial //= p
            break
    if primorial > 1:
        s = calc_score(primorial)
        if s > best_score:
            best_score = s
            best_n = primorial
        seen.add(primorial)

    scored_candidates = []
    for n in top_candidates:
        if time.time() > deadline - 10.0:
            break
        if n in seen:
            continue
        seen.add(n)
        s = calc_score(n)
        scored_candidates.append((s, n))
        if s > best_score:
            best_score = s
            best_n = n
            
    # Sort scored candidates by score
    scored_candidates.sort(reverse=True)
    
    # Phase 3: Local Search on the best candidates
    def local_search(n, max_iters=15):
        if time.time() > deadline - 2.0:
            return n, 0
        
        current_n = n
        current_score = calc_score(current_n)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            factors = factorize(current_n)
                
            best_neighbor_n = current_n
            best_neighbor_score = current_score
            
            # 1. Increase exponent
            for p, e in factors:
                if current_n * p <= LIMIT:
                    cand = current_n * p
                    if cand not in seen:
                        seen.add(cand)
                        s = calc_score(cand)
                        if s > best_neighbor_score:
                            best_neighbor_score = s
                            best_neighbor_n = cand
                            improved = True
            
            # 2. Decrease exponent
            for p, e in factors:
                if e > 0:
                    cand = current_n // p
                    if cand >= 2 and cand not in seen:
                        seen.add(cand)
                        s = calc_score(cand)
                        if s > best_neighbor_score:
                            best_neighbor_score = s
                            best_neighbor_n = cand
                            improved = True
            
            # 3. Replace prime p with any other prime q (not just adjacent)
            for i, (p, e) in enumerate(factors):
                if p in prime_set:
                    pi = primes.index(p)
                    base = current_n // (p ** e)
                    # Try replacing with nearby primes and some random ones
                    for ni in range(max(0, pi - 3), min(len(primes), pi + 4)):
                        if ni == pi:
                            continue
                        p_new = primes[ni]
                        cand = base * (p_new ** e)
                        if 1 < cand <= LIMIT and cand not in seen:
                            seen.add(cand)
                            s = calc_score(cand)
                            if s > best_neighbor_score:
                                best_neighbor_score = s
                                best_neighbor_n = cand
                                improved = True
            
            # 4. Add a new prime factor
            existing_primes = set(p for p, e in factors)
            for p in primes:
                if p not in existing_primes and current_n * p <= LIMIT:
                    cand = current_n * p
                    if cand not in seen:
                        seen.add(cand)
                        s = calc_score(cand)
                        if s > best_neighbor_score:
                            best_neighbor_score = s
                            best_neighbor_n = cand
                            improved = True

            if improved:
                current_n = best_neighbor_n
                current_score = best_neighbor_score
            else:
                break
                
        return current_n, current_score

    # Apply local search to top 10 unique scored candidates
    for s, n in scored_candidates[:10]:
        if time.time() > deadline - 5.0:
            break
        new_n, new_s = local_search(n)
        if new_s > best_score:
            best_score = new_s
            best_n = new_n

    return best_n
