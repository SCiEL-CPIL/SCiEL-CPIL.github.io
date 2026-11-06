import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin below 150s
    
    # The optimal n is expected to use the smallest primes.
    # Product of first 15 primes: 2*3*5*...*47 ~ 6e17.
    # Product of first 16 primes: *53 ~ 3.4e19 > 10^18.
    # So k is at most 15.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_val = 10**18
    
    best_n = 1
    best_score = -1
    
    # Cache for divisor generation if needed, but direct calc is likely faster for one-shot.
    
    def tau_perp(n):
        """
        Calculate the score tau_perp(n).
        """
        if n == 1:
            return 0
            
        # Factorize n using the known primes (since n is generated from them)
        divisors = [1]
        temp = n
        
        for p in primes:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                # Extend divisors
                current_divs = list(divisors)
                factor = 1
                for _ in range(count):
                    factor *= p
                    for d in current_divs:
                        divisors.append(d * factor)
            
            # If temp becomes 1, we have factored n completely
            if temp == 1:
                break
        
        # If temp > 1, n has a prime factor larger than our list (should not happen for our candidates)
        # But for safety, handle it if needed. In our construction, n is always from 'primes'.
        
        divisors.sort()
        
        score = 0
        # Optimization: use math.gcd
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
                
        return score

    def get_max_tau_suffix(idx, e_max, current_n):
        """
        Estimate the maximum possible tau for the remaining primes.
        This is an upper bound to allow pruning.
        """
        if idx >= len(primes):
            return 1
        
        # Loose upper bound: assume we can use e_max for all remaining primes
        # This is very loose but fast.
        # Better: Greedy assignment of max possible exponents.
        # But for speed, let's use a slightly tighter heuristic.
        # The max tau is achieved by smooth exponents.
        # We'll just return (e_max + 1) ** (len(primes) - idx)
        # This is an overestimate, which is safe for pruning (we might miss pruning, but won't prune incorrectly).
        return (e_max + 1) ** (len(primes) - idx)

    def dfs(idx, max_exp, current_n, current_tau):
        nonlocal best_n, best_score
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # If we have included at least one prime, it's a valid candidate
        # We evaluate all candidates found.
        if current_n > 1:
            # Heuristic check: if current_n is large, tau is likely high.
            # Just evaluate directly.
            score = tau_perp(current_n)
            if score > best_score:
                best_score = score
                best_n = current_n
        
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Determine max exponent for this prime
        # current_n * p^e <= max_val
        # p^e <= max_val // current_n
        limit = max_val // current_n
        
        e_max_possible = 0
        temp_product = 1
        while temp_product * p <= limit and e_max_possible < max_exp:
            temp_product *= p
            e_max_possible += 1
            
        # Try exponents from e_max_possible down to 1
        # Trying higher exponents first might find good candidates sooner,
        # but we need to explore all valid non-increasing sequences.
        # Actually, to maintain non-increasing exponents, e_next <= e_current.
        # So for this prime (idx), we can choose e from 1 to min(e_max_possible, max_exp).
        
        for e in range(1, min(e_max_possible, max_exp) + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            
            # Pruning: Check if we can possibly beat best_score
            # Upper bound on score is tau(n) - 1
            # Max tau for this branch is next_tau * max_possible_tau_suffix
            max_future_tau = get_max_tau_suffix(idx + 1, e, next_n)
            upper_bound_score = next_tau * max_future_tau - 1
            
            if upper_bound_score > best_score:
                dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    # Max exponent for 2 is log2(10^18) ~ 59
    # We pass max_exp = 60 to allow any reasonable start.
    dfs(0, 60, 1, 1)
    
    return best_n

if __name__ == "__main__":
    # Optional: run a quick self-test
    n = construct()
    # print(f"Best n: {n}")
    # print(f"Score: {tau_perp(n)}")
    pass
