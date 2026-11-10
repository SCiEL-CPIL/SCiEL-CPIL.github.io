def construct():
    import math
    import time
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s limit

    def _sieve(limit):
        """Return list of primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    def _get_divisors(factors):
        """Get all divisors given list of (prime, exponent) pairs."""
        divs = [1]
        for p, e in factors:
            powers = [p**i for i in range(e + 1)]
            new_divs = []
            for d in divs:
                for pw in powers:
                    new_divs.append(d * pw)
            divs = new_divs
        divs.sort()
        return divs

    def _tau_perp(factors):
        """Count adjacent coprime pairs in sorted divisors of n with given prime factorization."""
        divs = _get_divisors(factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count

    def _product(factors):
        prod = 1
        for p, e in factors:
            prod *= p**e
            if prod > 10**18:
                return prod
        return prod

    # Generate primes up to a reasonable limit
    all_primes = _sieve(300)

    # Initial solution: first 15 primes with exponent 1
    max_k = 0
    prod = 1
    for i, p in enumerate(all_primes):
        if prod * p <= 10**18:
            prod *= p
            max_k = i + 1
        else:
            break

    # Start with squarefree: first max_k primes, each exponent 1
    best_factors = [(p, 1) for p in all_primes[:max_k]]
    best_score = _tau_perp(best_factors)
    best_n = _product(best_factors)

    # Cache for tau_perp evaluations to avoid recomputation
    tau_cache = {}

    def _tau_perp_cached(factors_tuple):
        key = factors_tuple
        if key in tau_cache:
            return tau_cache[key]
        score = _tau_perp(list(factors_tuple))
        if len(tau_cache) < 100000:  # Limit cache size
            tau_cache[key] = score
        return score

    def try_modify(factors, idx, new_exp):
        """Try changing exponent of prime at idx to new_exp. Return (score, factors, prod) or (None, None, None)."""
        if new_exp < 1:
            # Remove this prime
            new_factors = list(factors)
            new_factors.pop(idx)
            if not new_factors:
                return None, None, None
            new_prod = _product(new_factors)
            if new_prod > 10**18:
                return None, None, None
            new_score = _tau_perp_cached(tuple(new_factors))
            return new_score, new_factors, new_prod
        new_factors = list(factors)
        new_factors[idx] = (new_factors[idx][0], new_exp)
        new_prod = _product(new_factors)
        if new_prod > 10**18:
            return None, None, None
        new_score = _tau_perp_cached(tuple(new_factors))
        return new_score, new_factors, new_prod

    def try_add_prime(factors, new_p):
        """Try adding a new prime with exponent 1."""
        if any(p == new_p for p, e in factors):
            return None, None, None
        new_factors = list(factors) + [(new_p, 1)]
        new_prod = _product(new_factors)
        if new_prod > 10**18:
            return None, None, None
        new_score = _tau_perp_cached(tuple(new_factors))
        return new_score, new_factors, new_prod

    def try_swap_prime(factors, idx, new_p):
        """Try swapping prime at idx with new_p, keeping same exponent."""
        if any(p == new_p for p, e in factors):
            return None, None, None
        new_factors = list(factors)
        old_exp = factors[idx][1]
        new_factors[idx] = (new_p, old_exp)
        new_prod = _product(new_factors)
        if new_prod > 10**18:
            return None, None, None
        new_score = _tau_perp_cached(tuple(new_factors))
        return new_score, new_factors, new_prod

    # Local search: iterate through all possible moves
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for idx in range(len(best_factors)):
            if time.time() > deadline - 5.0:
                break
            p, e = best_factors[idx]
            # Try changing exponent
            for new_e in range(1, e + 3):
                if new_e == e:
                    continue
                new_score, new_factors, new_prod = try_modify(best_factors, idx, new_e)
                if new_score is not None and new_score > best_score:
                    best_score = new_score
                    best_factors = new_factors
                    best_n = new_prod
                    improved = True
                    break
            # Try removing
            if time.time() > deadline - 5.0:
                break
            new_score, new_factors, new_prod = try_modify(best_factors, idx, 0)
            if new_score is not None and new_score > best_score:
                best_score = new_score
                best_factors = new_factors
                best_n = new_prod
                improved = True
                break
            # Try swapping with another prime
            if time.time() > deadline - 5.0:
                break
            for new_p in all_primes:
                if new_p == p:
                    continue
                new_score, new_factors, new_prod = try_swap_prime(best_factors, idx, new_p)
                if new_score is not None and new_score > best_score:
                    best_score = new_score
                    best_factors = new_factors
                    best_n = new_prod
                    improved = True
                    break
            if time.time() > deadline - 5.0:
                break
        # Try adding a new prime
        if improved or time.time() < deadline - 5.0:
            for new_p in all_primes:
                if time.time() > deadline - 5.0:
                    break
                new_score, new_factors, new_prod = try_add_prime(best_factors, new_p)
                if new_score is not None and new_score > best_score:
                    best_score = new_score
                    best_factors = new_factors
                    best_n = new_prod
                    improved = True
                    break

    # Second phase: try 2-swap moves (swap two primes simultaneously)
    if time.time() < deadline - 10.0:
        current_factors = list(best_factors)
        for i in range(len(current_factors)):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, len(current_factors)):
                if time.time() > deadline - 10.0:
                    break
                p_i, e_i = current_factors[i]
                p_j, e_j = current_factors[j]
                # Try swapping p_i with a larger prime and p_j with a smaller prime
                for new_p_i in all_primes:
                    if time.time() > deadline - 10.0:
                        break
                    if new_p_i == p_i or new_p_i == p_j:
                        continue
                    for new_p_j in all_primes:
                        if time.time() > deadline - 10.0:
                            break
                        if new_p_j == p_i or new_p_j == p_j or new_p_j == new_p_i:
                            continue
                        # Check if both new primes are available
                        test_factors = list(current_factors)
                        test_factors[i] = (new_p_i, e_i)
                        test_factors[j] = (new_p_j, e_j)
                        # Check for duplicates
                        primes_in_test = [f[0] for f in test_factors]
                        if len(set(primes_in_test)) != len(primes_in_test):
                            continue
                        new_prod = _product(test_factors)
                        if new_prod > 10**18:
                            continue
                        new_score = _tau_perp_cached(tuple(test_factors))
                        if new_score is not None and new_score > best_score:
                            best_score = new_score
                            best_factors = test_factors
                            best_n = new_prod
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Final local search pass with the potentially improved solution
    if improved and time.time() < deadline - 5.0:
        improved2 = True
        while improved2 and time.time() < deadline - 5.0:
            improved2 = False
            for idx in range(len(best_factors)):
                if time.time() > deadline - 5.0:
                    break
                p, e = best_factors[idx]
                for new_e in range(1, e + 3):
                    if new_e == e:
                        continue
                    new_score, new_factors, new_prod = try_modify(best_factors, idx, new_e)
                    if new_score is not None and new_score > best_score:
                        best_score = new_score
                        best_factors = new_factors
                        best_n = new_prod
                        improved2 = True
                        break
                if time.time() > deadline - 5.0:
                    break
                new_score, new_factors, new_prod = try_modify(best_factors, idx, 0)
                if new_score is not None and new_score > best_score:
                    best_score = new_score
                    best_factors = new_factors
                    best_n = new_prod
                    improved2 = True
                    break
                for new_p in all_primes:
                    if new_p == p:
                        continue
                    new_score, new_factors, new_prod = try_swap_prime(best_factors, idx, new_p)
                    if new_score is not None and new_score > best_score:
                        best_score = new_score
                        best_factors = new_factors
                        best_n = new_prod
                        improved2 = True
                        break
                if time.time() > deadline - 5.0:
                    break
            if improved2 or time.time() < deadline - 5.0:
                for new_p in all_primes:
                    if time.time() > deadline - 5.0:
                        break
                    new_score, new_factors, new_prod = try_add_prime(best_factors, new_p)
                    if new_score is not None and new_score > best_score:
                        best_score = new_score
                        best_factors = new_factors
                        best_n = new_prod
                        improved2 = True
                        break

    return best_n
