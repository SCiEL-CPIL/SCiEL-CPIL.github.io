import math
import time
import numpy as np

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def tau_perp_fast(n, primes_list):
        """Fast computation of tau_perp using NumPy for gcd."""
        # Factorize
        factors = []
        temp = n
        for p in primes_list:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
            if temp == 1:
                break
        
        if not factors:
            return 0, []
        
        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
        
        divs.sort()
        divs_arr = np.array(divs, dtype=np.int64)
        
        if len(divs_arr) < 2:
            return 0, divs
        
        # Compute gcd of adjacent pairs
        gcds = np.array([math.gcd(int(divs_arr[i]), int(divs_arr[i+1])) for i in range(len(divs_arr)-1)])
        score = int(np.sum(gcds == 1))
        
        return score, divs
    
    def tau_perp_python(n, primes_list):
        """Python fallback for tau_perp."""
        factors = []
        temp = n
        for p in primes_list:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
            if temp == 1:
                break
        
        if not factors:
            return 0
        
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
        
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    # Phase 1: Generate candidates with DFS
    # Key insight: optimal structure has many primes with exponent 1
    # We'll generate candidates and score them with a heuristic first
    
    candidates = []  # (heuristic_score, tau, n, num_distinct_primes)
    
    def dfs(idx, max_exp, current_n, current_tau, num_primes):
        if time.time() - start_time > 80:  # Reserve time for scoring
            return
        
        if current_n > 1:
            # Heuristic: prioritize many distinct primes, moderate tau
            # Score heuristic: num_primes * 10 + log(tau)
            heuristic = num_primes * 10 + math.log(current_tau + 1)
            candidates.append((heuristic, current_tau, current_n, num_primes))
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
        
        # Try exponents: for later primes, prioritize e=1
        if idx >= 8:
            e_values = [1] + list(range(e_max_possible, 1, -1))
        else:
            e_values = list(range(e_max_possible, 0, -1))
        
        for e in e_values:
            if e > max_exp:
                continue
            next_n = current_n * (p ** e)
            if next_n > max_val:
                continue
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau, num_primes + 1)
            
            if time.time() - start_time > 80:
                return
    
    # Start DFS
    dfs(0, 60, 1, 1, 0)
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    
    # Sort by heuristic score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Score top candidates
    best_score = -1
    best_n = candidates[0][2]
    
    scored_count = 0
    for heuristic, tau, n, num_primes in candidates:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        try:
            score, _ = tau_perp_fast(n, primes)
        except:
            score = tau_perp_python(n, primes)
        
        if score > best_score:
            best_score = score
            best_n = n
        
        scored_count += 1
        
        # Early stopping: if we've scored many and score is decreasing
        if scored_count > 500 and score < best_score * 0.7:
            break
    
    # Phase 3: Local search around best_n
    if time.time() - start_time < TIME_LIMIT - 15:
        # Get factorization of best_n
        best_factors = {}
        temp = best_n
        for p in primes:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                best_factors[p] = count
            if temp == 1:
                break
        
        local_candidates = set()
        
        # Try adjusting each prime's exponent
        for p in list(best_factors.keys()):
            e = best_factors[p]
            # Try increasing
            new_n = best_n * p
            if new_n <= max_val:
                local_candidates.add(new_n)
            # Try decreasing
            if e > 1:
                new_n = best_n // p
                if new_n >= 2:
                    local_candidates.add(new_n)
        
        # Try adding new small primes
        for p in primes:
            if p not in best_factors:
                new_n = best_n * p
                if new_n <= max_val:
                    local_candidates.add(new_n)
        
        # Try removing a prime
        for p in list(best_factors.keys()):
            if best_factors[p] >= 1:
                new_n = best_n // p
                if new_n >= 2:
                    local_candidates.add(new_n)
        
        # Evaluate local candidates
        for n in local_candidates:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            try:
                score, _ = tau_perp_fast(n, primes)
            except:
                score = tau_perp_python(n, primes)
            if score > best_score:
                best_score = score
                best_n = n
    
    return best_n
