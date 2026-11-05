"""Optimized strategy: Greedy prime addition based on marginal tau_perp gain."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Generate sufficient primes
    # Product of first 16 primes is ~1.1e19, so max distinct primes for n<=1e18 is 15.
    # We generate a few extra just in case, though we will stop when product exceeds limit.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(200)
    
    def calc_score(divs):
        """Calculate tau_perp score for a sorted list of divisors."""
        # divs must be sorted
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Start with n=1, divisors=[1]
    current_divs = [1]
    current_n = 1
    current_score = 0 # Score for n=1 is 0 (no pairs)
    
    # Greedy addition of primes
    # We consider primes in increasing order.
    # For each prime p, if current_n * p <= 10^18, we calculate the new score.
    # If it improves the score, we add it.
    # Note: It is possible that skipping a small prime allows adding a larger one later?
    # No, because primes are independent in terms of divisibility structure for squarefree numbers.
    # Adding a prime p multiplies the divisor count by 2.
    # The "cost" is the product value.
    # Smaller primes are "cheaper" in product terms.
    # Also, the marginal gain in tau_perp depends on the interaction with existing divisors.
    # However, for squarefree numbers, adding a prime p to n results in new divisors {d, d*p}.
    # The sorted list is interleaved.
    # Let's just iterate through primes in ascending order and add if beneficial and fits.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        new_n = current_n * p
        if new_n > 10**18:
            # If the smallest remaining prime doesn't fit, larger ones won't either.
            break
            
        # Calculate new divisors
        # New divisors are old divisors and old divisors * p
        # We need to merge them and sort.
        # Since current_divs is sorted, and current_divs * p is sorted, we can merge.
        old_divs = current_divs
        new_divs_part = [d * p for d in old_divs]
        
        # Merge two sorted lists
        merged = []
        i, j = 0, 0
        while i < len(old_divs) and j < len(new_divs_part):
            if old_divs[i] < new_divs_part[j]:
                merged.append(old_divs[i])
                i += 1
            else:
                merged.append(new_divs_part[j])
                j += 1
        if i < len(old_divs):
            merged.extend(old_divs[i:])
        if j < len(new_divs_part):
            merged.extend(new_divs_part[j:])
        
        # Calculate score for merged list
        new_score = calc_score(merged)
        
        # Accept if score improves
        # Since we are building from 1, and adding primes generally increases complexity,
        # is it always beneficial?
        # For n=1, score=0.
        # Add 2: divs [1,2]. Score: gcd(1,2)=1 -> 1. Improvement.
        # Add 3 to 2: divs [1,2,3,6]. Score: (1,2)1, (2,3)1, (3,6)3. Score 2. Improvement.
        # It seems adding primes always increases or maintains score?
        # Actually, adding a prime p introduces new adjacencies.
        # Let's just take it if it improves. If it ties, we might want to take smaller primes?
        # But we are iterating in ascending order.
        # Let's verify if score can ever decrease.
        # Divisors of n: D. Divisors of n*p: D union D*p.
        # Sorted merge.
        # New pairs are formed between elements of D and D*p.
        # It is highly likely the score increases.
        # Let's just accept if new_score >= current_score.
        # Actually, strictly greater is safer to avoid unnecessary bloat if score doesn't improve,
        # but for squarefree, it almost always improves.
        
        if new_score > current_score:
            current_divs = merged
            current_n = new_n
            current_score = new_score
        # If new_score == current_score, we might still want to add it?
        # No, it consumes product budget without gaining score.
        # But wait, if we skip it, can we add a larger prime?
        # Yes, but larger primes are "more expensive".
        # So if adding p doesn't help, adding q>p is also unlikely to help more per unit cost?
        # Actually, the structure of coprime adjacencies might differ.
        # But given the tight constraint, greedy ascending is a strong heuristic.
        
    # The current program (score 21.9) found a non-squarefree number.
    # My greedy squarefree strategy might be stuck at a lower score if non-squarefree is better.
    # Let's check the score of the best squarefree number.
    # Product of first 15 primes: 614889782588491410.
    # Let's estimate its score.
    # The current program's local search improved from squarefree seed.
    # This implies non-squarefree is better.
    # Therefore, a pure squarefree greedy approach might be inferior to the current program.
    # I need to adapt.
    
    # Revised Strategy:
    # 1. Greedy squarefree construction to get a strong seed.
    # 2. Then, perform a limited local search on exponents, but optimized.
    # The current program's local search is slow because it recalculates full divisor list and sorts every time.
    # Can we update the score incrementally?
    # Increasing exponent of p_i from e to e+1:
    # Divisors are multiplied by p_i.
    # This is complex.
    
    # Alternative:
    # Use the current program's logic but optimize the `calc_score` call.
    # Or, realize that the "marginal gain" of adding a prime is the key.
    # What if we allow exponents > 1 in the greedy phase?
    # No, that complicates the merge.
    
    # Let's stick to the observation: "Local search from a strong squarefree seed ... allows finding non-squarefree configurations".
    # The bottleneck in the current program is `calc_score` which does O(tau(n) * log(tau(n))) sort + O(tau(n)) gcd.
    # For tau(n) ~ 100k, this is ~100k * 17 ~ 1.7M ops.
    # In Python, 1.7M ops is ~0.05s?
    # If we do 1000 steps, that's 50s.
    # We have 145s.
    # The current program does a full pass of 20 primes * 2 moves = 40 evaluations per pass.
    # If each eval is 0.05s, a pass is 2s.
    # We can do ~70 passes.
    # Maybe the current program is just not efficient enough in `calc_score`.
    
    # Let's optimize `calc_score` by using `numpy` for gcd?
    # `numpy.gcd` is vectorized.
    # But sorting is still a bottleneck.
    # Can we avoid sorting?
    # Divisors generated by iterative multiplication are not sorted.
    # We must sort.
    
    # Let's try a different approach for the local search:
    # Instead of trying all primes, try only the "active" primes (exponent > 0).
    # And use a smarter neighbor selection.
    
    # Actually, let's look at the inspiration 1 (score 21.9184). It's the same as the current program.
    # The prompt says "Current Program (score 21.9184)".
    # So I am trying to beat 21.9184.
    
    # Key insight: The current program uses `math.gcd` in a Python loop.
    # I can use `numpy` to vectorize the gcd calculation.
    # `np.gcd` on two arrays is much faster than a Python loop.
    # Sorting is still needed, but `numpy.sort` is faster than `list.sort` for large lists?
    # `list.sort` is Timsort (C implementation), very fast.
    # `numpy.sort` is also C.
    # The main gain is in the gcd loop.
    
    # Let's implement a version that:
    # 1. Uses the same greedy squarefree seed.
    # 2. Uses a local search but with vectorized scoring.
    
    import numpy as np
    
    def calc_score_fast(exps, primes_list):
        # Reconstruct n
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= primes_list[i] ** e
        if n > 10**18 or n < 2:
            return -1
            
        # Generate divisors
        # This part is hard to vectorize easily because the number of divisors varies.
        # We can generate them as a list of lists, then flatten.
        divs = [1]
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes_list[i]
            p_pows = [p ** k for k in range(e + 1)]
            # Cartesian product
            new_divs = []
            for d in divs:
                for pp in p_pows:
                    new_divs.append(d * pp)
            divs = new_divs
            
        divs_np = np.array(divs, dtype=np.int64)
        divs_np.sort()
        
        # Vectorized gcd
        if len(divs_np) < 2:
            return 0
            
        # gcd of adjacent elements
        # np.gcd(divs_np[:-1], divs_np[1:])
        # This creates two arrays of size tau(n)-1.
        g = np.gcd(divs_np[:-1], divs_np[1:])
        score = np.sum(g == 1)
        
        return int(score)

    # Re-initialize with the greedy squarefree seed
    current_divs = [1]
    current_n = 1
    current_score = 0
    seed_exps = [0] * len(primes)
    
    for i, p in enumerate(primes):
        if time.time() > deadline:
            break
        new_n = current_n * p
        if new_n > 10**18:
            break
            
        # Calculate score for squarefree addition
        # We can do this efficiently for squarefree numbers.
        # But for simplicity, let's just use the fast calc_score for the seed phase too?
        # No, seed phase is small.
        
        # Quick score check for squarefree merge
        old_divs = current_divs
        new_divs_part = [d * p for d in old_divs]
        merged = []
        i_idx, j_idx = 0, 0
        while i_idx < len(old_divs) and j_idx < len(new_divs_part):
            if old_divs[i_idx] < new_divs_part[j_idx]:
                merged.append(old_divs[i_idx])
                i_idx += 1
            else:
                merged.append(new_divs_part[j_idx])
                j_idx += 1
        if i_idx < len(old_divs):
            merged.extend(old_divs[i_idx:])
        if j_idx < len(new_divs_part):
            merged.extend(new_divs_part[j_idx:])
            
        # Score merged
        # For small lists, Python loop is fine.
        s = 0
        for k in range(len(merged) - 1):
            if math.gcd(merged[k], merged[k+1]) == 1:
                s += 1
                
        if s > current_score:
            current_divs = merged
            current_n = new_n
            current_score = s
            seed_exps[i] = 1
            
    # Now Local Search with Fast Score
    best_exps = seed_exps[:]
    best_score = current_score
    
    # Limit search to first 20 primes
    max_primes = 20
    while len(best_exps) < max_primes:
        best_exps.append(0)
    best_exps = best_exps[:max_primes]
    
    # Re-evaluate best_score with fast method to ensure consistency
    best_score = calc_score_fast(best_exps, primes[:max_primes])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(max_primes):
            # Try increasing
            if best_exps[i] + 1 <= 20:
                new_exps = best_exps[:]
                new_exps[i] += 1
                s = calc_score_fast(new_exps, primes[:max_primes])
                if s > best_score:
                    best_score = s
                    best_exps = new_exps
                    improved = True
            # Try decreasing
            if best_exps[i] > 0:
                new_exps = best_exps[:]
                new_exps[i] -= 1
                s = calc_score_fast(new_exps, primes[:max_primes])
                if s > best_score:
                    best_score = s
                    best_exps = new_exps
                    improved = True
                    
            if time.time() > deadline:
                break
                
    # Final n
    final_n = 1
    for i in range(max_primes):
        if best_exps[i] > 0:
            final_n *= primes[i] ** best_exps[i]
            
    if final_n > 10**18 or final_n < 2:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
        
    return final_n
# EVOLVE-BLOCK-END
