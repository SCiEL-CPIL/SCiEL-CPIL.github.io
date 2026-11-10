def construct():
    import math
    import time
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

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

    def _get_divisors(prime_set):
        """Get all divisors of a squarefree number given its prime factors."""
        divs = [1]
        for p in prime_set:
            new_divs = []
            for d in divs:
                new_divs.append(d)
                new_divs.append(d * p)
            divs = new_divs
        divs.sort()
        return divs

    def _tau_perp(prime_set):
        """Count adjacent coprime pairs in sorted divisors of squarefree n."""
        divs = _get_divisors(prime_set)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count

    def _product(prime_set):
        prod = 1
        for p in prime_set:
            prod *= p
            if prod > 10**18:
                return prod
        return prod

    # Generate primes up to a reasonable limit
    # Product of first 15 primes ≈ 6.15e17, first 16 ≈ 3.25e19 > 1e18
    # We need primes up to maybe 200 to allow swaps
    all_primes = _sieve(300)

    # Determine how many of the smallest primes we can fit
    prod = 1
    max_k = 0
    for i, p in enumerate(all_primes):
        if prod * p <= 10**18:
            prod *= p
            max_k = i + 1
        else:
            break

    # Initial solution: first max_k primes
    initial_set = all_primes[:max_k]
    best_score = _tau_perp(initial_set)
    best_n = prod
    best_set = list(initial_set)

    def try_swap(current_set, idx, new_p):
        """Try swapping prime at idx with new_p, return (new_score, new_set) if valid."""
        if new_p in current_set:
            return None, None
        new_set = list(current_set)
        new_set[idx] = new_p
        new_prod = _product(new_set)
        if new_prod > 10**18:
            return None, None
        new_score = _tau_perp(new_set)
        return new_score, new_set

    # Local search: try swapping one prime at a time
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for idx in range(len(best_set)):
            if time.time() > deadline - 5.0:
                break
            orig = best_set[idx]
            for new_p in all_primes:
                if new_p == orig:
                    continue
                if new_p in best_set:
                    continue
                new_score, new_set = try_swap(best_set, idx, new_p)
                if new_score is not None and new_score > best_score:
                    best_score = new_score
                    best_n = _product(new_set)
                    best_set = new_set
                    improved = True
                    break

    # Try 2-swap moves: swap two primes simultaneously
    if time.time() < deadline - 10.0:
        for i in range(len(best_set)):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, len(best_set)):
                if time.time() > deadline - 10.0:
                    break
                for new_p1 in all_primes:
                    if time.time() > deadline - 10.0:
                        break
                    if new_p1 == best_set[i] or new_p1 == best_set[j]:
                        continue
                    if new_p1 in best_set:
                        continue
                    for new_p2 in all_primes:
                        if new_p2 == best_set[i] or new_p2 == best_set[j]:
                            continue
                        if new_p2 == new_p1:
                            continue
                        if new_p2 in best_set:
                            continue
                        new_set = list(best_set)
                        new_set[i] = new_p1
                        new_set[j] = new_p2
                        new_prod = _product(new_set)
                        if new_prod > 10**18:
                            continue
                        new_score = _tau_perp(new_set)
                        if new_score > best_score:
                            best_score = new_score
                            best_n = new_prod
                            best_set = new_set
                            break
                    if time.time() > deadline - 10.0:
                        break

    # Also try adding a prime if possible (increase k)
    if time.time() < deadline - 5.0:
        for j in range(len(all_primes)):
            if time.time() > deadline - 5.0:
                break
            if all_primes[j] in best_set:
                continue
            new_set = list(best_set) + [all_primes[j]]
            new_prod = best_n * all_primes[j]
            if new_prod > 10**18:
                continue
            new_score = _tau_perp(new_set)
            if new_score > best_score:
                best_score = new_score
                best_n = new_prod
                best_set = new_set

    # Try removing a prime if it improves score (unlikely but check)
    if time.time() < deadline - 5.0:
        for i in range(len(best_set)):
            if time.time() > deadline - 5.0:
                break
            new_set = list(best_set)
            new_set.pop(i)
            new_prod = _product(new_set)
            new_score = _tau_perp(new_set)
            if new_score > best_score:
                best_score = new_score
                best_n = new_prod
                best_set = new_set

    return best_n
