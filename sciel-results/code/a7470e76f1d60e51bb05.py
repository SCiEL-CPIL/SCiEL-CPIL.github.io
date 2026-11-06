import math
import time

def construct():
    start_time = time.time()
    time_limit = 145
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    candidates = set()
    
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 50:
            return
        if current_n > 1:
            candidates.add((current_tau, current_n))
        if idx >= len(primes):
            return
        p = primes[idx]
        limit_n = max_val // current_n
        e_max = 0
        temp = 1
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            if time.time() - start_time > 50:
                return

    dfs(0, 60, 1, 1)
    
    pk = 1
    for i, p in enumerate(primes):
        pk *= p
        if pk > max_val:
            break
        candidates.add((2**(i+1), pk))
        for a in range(1, 12):
            n = pk * (2 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
        for a in range(1, 8):
            n = pk * (3 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
        for a in range(1, 6):
            n = pk * (5 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
        for a in range(1, 5):
            n = pk * (7 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
        if i >= 2:
            pk_no_last = pk // primes[i]
            for a in range(2, 14):
                n = pk_no_last * (2 ** a)
                if n <= max_val and n > 1:
                    candidates.add((2**i * (a+1), n))
            for a in range(2, 10):
                n = pk_no_last * (3 ** a)
                if n <= max_val and n > 1:
                    candidates.add((2**i * (a+1), n))
            for a in range(2, 7):
                n = pk_no_last * (5 ** a)
                if n <= max_val and n > 1:
                    candidates.add((2**i * (a+1), n))
            if i >= 4:
                pk_no_last2 = pk_no_last // primes[i-1]
                for a in range(2, 16):
                    n = pk_no_last2 * (2 ** a)
                    if n <= max_val and n > 1:
                        candidates.add((2**(i-1) * (a+1), n))
                for a in range(2, 12):
                    n = pk_no_last2 * (3 ** a)
                    if n <= max_val and n > 1:
                        candidates.add((2**(i-1) * (a+1), n))

    def generate_shifted_candidates(base_primes, max_shifts=3):
        base_n = 1
        for p in base_primes:
            base_n *= p
        if base_n > max_val:
            return
        exponents = [1] * len(base_primes)

        def shift_search(idx, remaining_budget, current_exps, current_n):
            if time.time() - start_time > 60:
                return
            n = 1
            for i, p in enumerate(base_primes):
                n *= p ** current_exps[i]
            if n <= max_val and n > 1:
                tau = 1
                for e in current_exps:
                    tau *= (e + 1)
                candidates.add((tau, n))
            if idx >= len(base_primes):
                return
            p = base_primes[idx]
            max_e = current_exps[idx] + remaining_budget + 2
            for e in range(current_exps[idx], max_e + 1):
                if current_n * (p ** (e - current_exps[idx])) > max_val:
                    break
                new_exps = current_exps[:]
                new_exps[idx] = e
                new_n = current_n * (p ** (e - current_exps[idx]))
                new_budget = remaining_budget - (e - current_exps[idx])
                shift_search(idx + 1, new_budget, new_exps, new_n)
            if current_exps[idx] > 1:
                for e in range(1, current_exps[idx]):
                    new_exps = current_exps[:]
                    new_exps[idx] = e
                    new_n = current_n // (p ** (current_exps[idx] - e))
                    new_budget = remaining_budget + (current_exps[idx] - e)
                    shift_search(idx + 1, new_budget, new_exps, new_n)

        shift_search(0, 0, exponents, base_n)

    for num_primes in [13, 14, 15]:
        if num_primes <= len(primes):
            generate_shifted_candidates(primes[:num_primes])

    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    candidate_list = list(candidates)
    candidate_list.sort(key=lambda x: x[0], reverse=True)

    def factorize(n):
        factors = []
        temp_n = n
        for p in primes:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                factors.append((p, count))
            if temp_n == 1:
                break
        if temp_n > 1:
            factors.append((temp_n, 1))
        return factors

    def tau_perp(n):
        factors = factorize(n)
        if not factors:
            return 0
        k = len(factors)
        
        # Generate all divisors
        divs = [1]
        for p, e in factors:
            powers = [1]
            for _ in range(e):
                powers.append(powers[-1] * p)
            new_divs = []
            for d in divs:
                for pw in powers:
                    new_divs.append(d * pw)
            divs = new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Evaluate top candidates first
    best_score = -1
    best_n = candidate_list[0][1]
    
    evaluated = set()
    max_eval = min(len(candidate_list), 2000)  # Limit evaluations to ensure time
    
    for i in range(max_eval):
        if time.time() - start_time > time_limit - 10:
            break
        tau, n = candidate_list[i]
        if n in evaluated:
            continue
        evaluated.add(n)
        
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    # More aggressive local search with multiple restarts
    if time.time() - start_time < time_limit - 5:
        def local_search(n, max_iterations=1000):
            factors = factorize(n)
            best_local = n
            best_local_score = tau_perp(n)
            
            for _ in range(max_iterations):
                if time.time() - start_time > time_limit - 3:
                    break
                improved = False
                # Try all single exponent changes
                for i, (p, e) in enumerate(factors):
                    # Increase
                    if e < 60:
                        new_n = n * p
                        if new_n <= max_val:
                            new_score = tau_perp(new_n)
                            if new_score > best_local_score:
                                best_local_score = new_score
                                best_local = new_n
                                factors = factorize(new_n)
                                improved = True
                                break
                    # Decrease
                    if e > 1:
                        new_n = n // p
                        if new_n > 1:
                            new_score = tau_perp(new_n)
                            if new_score > best_local_score:
                                best_local_score = new_score
                                best_local = new_n
                                factors = factorize(new_n)
                                improved = True
                                break
                # Try adding a new prime factor
                if not improved:
                    for p in primes:
                        if n % p != 0:
                            new_n = n * p
                            if new_n <= max_val:
                                new_score = tau_perp(new_n)
                                if new_score > best_local_score:
                                    best_local_score = new_score
                                    best_local = new_n
                                    factors = factorize(new_n)
                                    improved = True
                                    break
                if not improved:
                    break
            return best_local, best_local_score
        
        improved_n, improved_score = local_search(best_n)
        if improved_score > best_score:
            best_score = improved_score
            best_n = improved_n
        
        # Try local search from other top candidates
        if time.time() - start_time < time_limit - 5:
            for i in range(1, min(5, len(candidate_list))):
                if time.time() - start_time > time_limit - 3:
                    break
                tau, n = candidate_list[i]
                if n in evaluated:
                    continue
                improved_n2, improved_score2 = local_search(n, max_iterations=200)
                if improved_score2 > best_score:
                    best_score = improved_score2
                    best_n = improved_n2
    
    return best_n
