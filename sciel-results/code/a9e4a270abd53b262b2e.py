import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    LIMIT = 10**18

    # Precompute primes up to a reasonable bound for the pool
    def sieve(n):
        is_prime = [True] * (n + 1)
        if n >= 0: is_prime[0] = False
        if n >= 1: is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    # We need primes that can be added to a product < 10^18.
    # The product of the first 15 primes is ~6*10^17.
    # We might want to swap some small primes for larger ones if it helps, 
    # but the greedy interval merging suggests we should build up.
    # Let's generate primes up to 1000.
    all_primes = sieve(2000)
    
    # Pool of candidate primes. 
    # We want to consider primes that are not too small (to keep divisors separated?) 
    # and not too large (to fit in budget).
    # Actually, let's just use all primes up to 2000 as candidates for each step.
    # But we must ensure the final product <= LIMIT.

    def fast_score(n):
        """Compute tau_perp(n) by generating all divisors, sorting, counting adjacent coprime pairs."""
        if n < 2:
            return 0
        # Factorize n
        temp = n
        factors = {}
        for p in all_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors[p] = cnt
        if temp > 1:
            factors[temp] = 1
        
        # Generate all divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        
        divs.sort()
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    # Greedy Interval Merging Strategy:
    # Start with n = 1 (divisors: [1]).
    # At each step, try adding a prime p from the pool.
    # The new divisors are D_old U {d * p for d in D_old}.
    # The sorted order is interleaved.
    # We want to maximize the number of NEW coprime pairs.
    # A new pair is formed between two divisors that were not adjacent before, 
    # or replaces an old boundary.
    # Actually, the simplest robust greedy is:
    # 1. Start with an empty set of primes.
    # 2. For k from 1 to max_k:
    #    Try all available primes p such that current_product * p <= LIMIT.
    #    For each candidate p, compute the score of the new set of primes.
    #    Select the p that gives the highest score.
    #    Add p to the set.
    # 3. Stop when no prime can be added or max_k reached.
    # 4. Then, try local search (swaps) to improve.
    
    # This is essentially a greedy construction.
    # To make it "Interval Merging" flavored, we note that adding a prime p
    # effectively doubles the number of divisors. The "quality" of the prime
    # is determined by how well it interleaves with existing divisors to create
    # coprime boundaries.
    
    # Optimization: We can pre-filter primes.
    # If we have product P, we can add p only if P*p <= LIMIT.
    
    current_primes = []
    current_product = 1
    current_score = 0
    
    # Max possible number of primes is 15 (primorial of 15 primes < 10^18).
    # But maybe fewer with better primes?
    # Let's allow up to 16 steps, but the product constraint will naturally stop it.
    
    for step in range(16):
        if time.time() > DEADLINE - 10:
            break
            
        best_p = None
        best_score_candidate = current_score - 1 # Must improve
        
        # Candidates: primes not in current_primes
        # To speed up, we can restrict to primes < LIMIT // current_product
        max_p_allowed = LIMIT // current_product
        
        # We should consider primes that are "reasonable".
        # Very small primes might create dense clusters.
        # Very large primes might be isolated.
        # Let's iterate through all_primes.
        
        # To avoid O(N^2) slow scoring if we do this for many steps,
        # we note that fast_score is O(2^k log 2^k).
        # For k=10, 1024 divisors. Very fast.
        # For k=15, 32768 divisors. Still fast (~0.05s).
        # So we can afford to evaluate all candidates at each step.
        
        candidates = [p for p in all_primes if p > max_p_allowed // 2 and p <= max_p_allowed and p not in current_primes]
        # Wait, why p > max_p_allowed // 2? No, just p <= max_p_allowed.
        # Let's just use all valid primes.
        candidates = [p for p in all_primes if p <= max_p_allowed and p not in current_primes]
        
        if not candidates:
            break
            
        # Sort candidates? Maybe smaller primes are more likely to fit?
        # Let's just iterate.
        
        # To save time, if we have many candidates, we can sample or use a heuristic.
        # But let's try full scan first. If too slow, we'll optimize.
        
        # Heuristic: The prime that creates the most coprime boundaries is likely
        # one that is "coprime" to many existing divisors? 
        # Since all primes are distinct, p is always coprime to any divisor not containing p.
        # The issue is adjacency in the sorted list.
        
        # Let's just compute the score for each candidate.
        # To speed up, we can cache scores? No, each set of primes is unique.
        
        # Optimization: 
        # If we have k primes, there are ~2000 candidates.
        # Scoring each takes ~0.05s (for k=15). 2000 * 0.05 = 100s. Too slow for 15 steps.
        # We need a faster way to evaluate the gain.
        
        # Alternative: 
        # The gain from adding p is the number of new coprime adjacencies.
        # This is hard to estimate without generating the list.
        # However, we can limit the candidate pool.
        # Primes that are "far apart" from existing primes in log scale might be better?
        # Or just use the first 100 primes in the pool?
        
        # Let's restrict candidates to the first 500 primes in the pool that fit.
        # And maybe prioritize smaller primes? Or larger?
        # Let's try a mix: sample 100 candidates from the valid range.
        
        # Actually, let's just use the first 100 valid primes.
        # The "best" primes are likely small ones (to keep product low) 
        # OR specific large ones that create gaps.
        # The primorial uses the smallest. 
        # Let's see if swapping a small prime for a larger one helps.
        # The greedy approach will naturally pick the best one.
        
        # To make it fast:
        # 1. If k < 10, do full scan of first 200 valid primes.
        # 2. If k >= 10, do full scan of first 50 valid primes.
        
        limit_candidates = 200 if len(current_primes) < 10 else 50
        candidates = candidates[:limit_candidates]
        
        for p in candidates:
            if time.time() > DEADLINE - 5:
                break
                
            new_primes = current_primes + [p]
            new_product = current_product * p
            
            # Quick check: is it worth computing?
            # If new_product > LIMIT: continue (already filtered)
            
            # Compute score
            # We can reuse the fast_score logic but pass the list of primes directly
            # to avoid factorization overhead.
            
            def score_from_primes(prime_list):
                divs = [1]
                for p in prime_list:
                    new_divs = []
                    for d in divs:
                        new_divs.append(d * p)
                    divs.extend(new_divs)
                divs.sort()
                score = 0
                prev = divs[0]
                for i in range(1, len(divs)):
                    curr = divs[i]
                    if math.gcd(prev, curr) == 1:
                        score += 1
                    prev = curr
                return score

            s = score_from_primes(new_primes)
            
            if s > best_score_candidate:
                best_score_candidate = s
                best_p = p
        
        if best_p is not None:
            current_primes.append(best_p)
            current_product *= best_p
            current_score = best_score_candidate
        else:
            break

    # Now we have a set of primes found by greedy.
    # Let's do a local search (hill climbing) to improve.
    # Try swapping one prime with another.
    
    best_n = current_product
    best_primes = current_primes[:]
    best_score_val = current_score
    
    # Local search
    # We have ~15 primes. 
    # Pool of swaps: any prime not in best_primes that fits.
    
    improved = True
    while improved and time.time() < DEADLINE - 10:
        improved = False
        for i in range(len(best_primes)):
            if time.time() > DEADLINE - 5:
                break
            p_old = best_primes[i]
            max_p_allowed = LIMIT // (best_n // p_old)
            
            # Candidates for swap
            # We want primes that are not in best_primes.
            # Let's look at primes up to 2000.
            candidates = [p for p in all_primes if p <= max_p_allowed and p not in best_primes]
            
            # Limit candidates for speed
            candidates = candidates[:100]
            
            for p_new in candidates:
                if time.time() > DEADLINE - 2:
                    break
                
                new_primes = best_primes[:]
                new_primes[i] = p_new
                new_product = best_n // p_old * p_new
                
                # Compute score
                def score_from_primes(prime_list):
                    divs = [1]
                    for p in prime_list:
                        new_divs = []
                        for d in divs:
                            new_divs.append(d * p)
                        divs.extend(new_divs)
                    divs.sort()
                    score = 0
                    prev = divs[0]
                    for i in range(1, len(divs)):
                        curr = divs[i]
                        if math.gcd(prev, curr) == 1:
                            score += 1
                        prev = curr
                    return score

                s = score_from_primes(new_primes)
                
                if s > best_score_val:
                    best_score_val = s
                    best_n = new_product
                    best_primes = new_primes
                    improved = True
                    # Break out of candidate loop? 
                    # If we find an improvement, we can continue to see if a better swap exists at this index?
                    # Or just accept and restart?
                    # Accepting and restarting is safer.
                    break
            
            if improved:
                # Restart the outer loop? 
                # Or just continue to next i?
                # If we changed best_primes, the index i might be different now? 
                # No, we just swapped one element. The list length is same.
                # But the values changed. 
                # It's better to restart the whole local search to find the new local optimum.
                break
    
    # Final verification
    if best_n < 2 or best_n > LIMIT:
        return 2
    
    # One last check: is the primorial better?
    # The greedy might have gotten stuck.
    # Let's compare with the standard primorial of first 15 primes.
    
    primorial_primes = all_primes[:15]
    primorial_n = 1
    for p in primorial_primes:
        primorial_n *= p
    
    if primorial_n <= LIMIT:
        def score_from_primes(prime_list):
            divs = [1]
            for p in prime_list:
                new_divs = []
                for d in divs:
                    new_divs.append(d * p)
                divs.extend(new_divs)
            divs.sort()
            score = 0
            prev = divs[0]
            for i in range(1, len(divs)):
                curr = divs[i]
                if math.gcd(prev, curr) == 1:
                    score += 1
                prev = curr
            return score
        
        p_score = score_from_primes(primorial_primes)
        if p_score > best_score_val:
            best_n = primorial_n
            best_score_val = p_score
    
    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
