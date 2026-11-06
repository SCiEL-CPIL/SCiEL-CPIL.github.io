import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    LIMIT = 10**18

    # Precompute primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    # We need primes up to a reasonable bound. 
    # For n <= 10^18, the largest prime factor can be large, but for squarefree 
    # with many factors, they are small.
    # However, we might want to swap in larger primes to optimize adjacency.
    all_primes = sieve(1000)
    
    def calc_tau_perp_exact(n):
        """Compute tau_perp(n) exactly. Used only for final verification."""
        if n < 2:
            return 0
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

    def estimate_score(prime_set):
        """
        Structural heuristic: 
        For squarefree n, the divisors are subset products.
        The number of coprime adjacent pairs is related to the 'separation' of primes.
        If primes are very close, divisors cluster. If primes are spaced, 
        there are more 'gaps' where coprime transitions occur.
        
        A better proxy: 
        The number of divisors is 2^k.
        In the worst case (primes very close), many adjacent divisors share factors.
        In the best case (primes very spaced), more adjacent divisors are coprime.
        
        Actually, for squarefree numbers, the structure is fixed by the relative magnitudes.
        Let's use a simpler proxy: prefer sets with more primes, and among same count,
        prefer those with larger gaps between log(primes) to maximize structural diversity?
        Or simply: maximize k (number of primes), then minimize product?
        
        Wait, the task is to maximize tau_perp. 
        For squarefree n with k primes, tau_perp is roughly proportional to k or 2^k?
        Let's assume maximizing k is the primary driver.
        """
        return len(prime_set)

    best_score = -1
    best_n = 2
    
    # Strategy:
    # 1. Find the maximum number of distinct primes k such that product(p1...pk) <= LIMIT.
    # 2. Among all subsets of size k, k-1, k-2, find the one that maximizes the exact score.
    #    To do this efficiently, we don't check all 2^30 subsets.
    #    We use a greedy/beam search or random sampling focused on 'good' structures.
    
    # Determine max k for squarefree using smallest primes
    max_k_sq = 0
    prod = 1
    for i, p in enumerate(all_primes):
        if prod * p <= LIMIT:
            prod *= p
            max_k_sq = i + 1
        else:
            break
    
    # The max_k_sq is likely around 15 (product of first 15 primes is ~6e17)
    # Let's verify: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    # 2*... * 53 = 32589158477190044730 > 10^18. So max_k_sq is 15.
    
    # We will search for the best subset of size 15, 14, 13.
    # Since 2^15 = 32768 divisors, exact calc is fast (~ms).
    # But we can't check all C(30, 15) subsets.
    
    # Heuristic: 
    # Start with the smallest primes (min product, max k).
    # Then try swapping larger primes for slightly larger ones to see if structure improves.
    # Why? Maybe having primes that are "incommensurate" helps?
    # Actually, for squarefree, the sorted order is determined by the values.
    # If primes are p1 < p2 < ... < pk.
    # Divisors are all subset products.
    # Adjacent divisors d_i, d_{i+1}.
    # If we replace a prime p_j with a larger p'_j, the divisors involving p_j shift up.
    # This might create more gaps or merge gaps.
    
    # Let's do a local search starting from the minimal prime set.
    
    initial_set = all_primes[:max_k_sq]
    current_n = 1
    for p in initial_set:
        current_n *= p
    
    current_score = calc_tau_perp_exact(current_n)
    best_score = current_score
    best_n = current_n
    
    # Local Search: Swap one prime in the set with a larger prime not in the set.
    # Constraint: New product must be <= LIMIT.
    
    random.seed(42)
    tried_swaps = set()
    
    # Pool of candidate primes for swapping (larger than max in set, but not too large)
    # If we swap a small prime p_i with a large prime P, the product increases.
    # To stay under LIMIT, we must swap a small prime for a slightly larger one,
    # or swap multiple primes.
    
    # Actually, since we are at max_k (15), we can't add more primes.
    # We can only swap.
    # If we swap p_i (small) with P (large), product increases by factor P/p_i.
    # We have "slack" = LIMIT / current_n.
    # slack is roughly 10^18 / 6e17 ~ 1.6.
    # So we can only swap p_i with P if P <= p_i * 1.6.
    # This is a very tight constraint.
    
    # Let's check the slack.
    # Product of first 15 primes:
    p15 = 47
    prod15 = 614889782588491410
    slack = LIMIT / prod15
    # slack is ~1.62
    
    # So for each prime p_i in the set, we can only swap it with primes P in (p_i, p_i * slack].
    # This limits the search space significantly, which is good.
    
    # Let's refine the local search to be systematic.
    # For each prime p_i in current_set:
    #   Find all primes P such that p_i < P <= p_i * slack AND P not in current_set.
    #   Try swapping p_i with P.
    #   If score improves, accept.
    
    current_set = set(initial_set)
    
    # Precompute primes up to max_prime_in_set * slack
    max_p_in_set = max(current_set)
    max_search_prime = int(max_p_in_set * slack) + 100
    all_primes_extended = sieve(max_search_prime)
    
    # Iterate local search
    for _ in range(1000): # Max iterations
        if time.time() > DEADLINE - 5:
            break
            
        improved = False
        # Try all possible single swaps
        for p_i in list(current_set):
            # Find candidates
            min_p = p_i
            max_p_allowed = p_i * slack
            candidates = [p for p in all_primes_extended if min_p < p <= max_p_allowed and p not in current_set]
            
            for p_new in candidates:
                # Calculate new n
                new_n = current_n // p_i * p_new
                if new_n > LIMIT or new_n < 2:
                    continue
                
                # Check if this swap was tried recently? 
                # For now, just evaluate.
                score_new = calc_tau_perp_exact(new_n)
                
                if score_new > best_score:
                    best_score = score_new
                    best_n = new_n
                    current_n = new_n
                    current_set.remove(p_i)
                    current_set.add(p_new)
                    improved = True
                    break # Restart search with new best? Or continue?
                    # Let's restart to be safe and converge to local optimum.
            if improved:
                break
        
        if not improved:
            # Try random swaps from a wider pool? 
            # Or try swapping 2 primes?
            # Given tight slack, 2-prime swaps are rare.
            # Let's try a random restart with a different initial set?
            # E.g., start with k=14 primes and see if we can add a 15th?
            # No, max_k is fixed by the smallest primes.
            
            # Maybe we should explore k=14 more deeply?
            # Sometimes a set of 14 primes with better structure beats 15 primes with bad structure?
            # Unlikely, as 2^15 divisors vs 2^14. The score scales with divisor count roughly.
            # But let's check k=14 just in case.
            
            break
    
    # Also, let's try to see if we can improve by using exponents.
    # If we use p^2, we lose the ability to use another prime.
    # Usually squarefree is better for maximizing divisor count within limit.
    # But maybe p^2 creates more coprime pairs?
    # Divisors of p^2: 1, p, p^2.
    # Divisors of p*q: 1, p, q, pq.
    # For p^2: pairs (1,p) gcd p, (p,p^2) gcd p. Score 0.
    # For p*q: pairs (1,p) gcd 1, (p,q) gcd 1 (if p!=q), (q,pq) gcd q. Score 2.
    # So squarefree is generally better for coprime pairs.
    
    # Final verification of best_n
    final_score = calc_tau_perp_exact(best_n)
    if final_score > best_score:
        best_score = final_score
    
    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
