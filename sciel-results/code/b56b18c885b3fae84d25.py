import math
import time
from math import gcd

def _get_primes(limit):
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

_PRIMES = _get_primes(300)

def _generate_divisors(factorization):
    """Generate all divisors from list of (prime, exponent) pairs."""
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp_from_divisors(divs):
    """Count adjacent coprime pairs in sorted divisor list."""
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _tau_perp_fast(factorization):
    """Compute tau_perp from factorization."""
    divs = _generate_divisors(factorization)
    return _tau_perp_from_divisors(divs)

def _eval_candidate(factorization, LIMIT):
    """Evaluate a candidate factorization. Returns (n, score) or None if invalid."""
    n = 1
    for p, e in factorization:
        n *= p ** e
    if n < 2 or n > LIMIT:
        return None
    num_divs = 1
    for _, e in factorization:
        num_divs *= (e + 1)
    if num_divs > 500000:
        return None
    score = _tau_perp_fast(factorization)
    return (n, score)

def _search():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    LIMIT = 10 ** 18

    primes = _PRIMES
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Max k: product of first k primes <= LIMIT
    max_k = 0
    prod = 1
    for i, p in enumerate(primes):
        if prod * p > LIMIT:
            break
        prod *= p
        max_k = i + 1

    best_n = 1
    best_score = 0
    best_factorization = []

    def try_candidate(factorization):
        nonlocal best_n, best_score, best_factorization
        result = _eval_candidate(factorization, LIMIT)
        if result:
            n, score = result
            if score > best_score:
                best_score = score
                best_n = n
                best_factorization = factorization[:]

    # Phase 1: Evaluate primorial baselines
    for k in range(2, max_k + 1):
        if time.time() > deadline - 40:
            break
        factorization = [(primes[i], 1) for i in range(k)]
        try_candidate(factorization)

    # Phase 2: Systematic exponent search for top k values
    # Focus on k near max_k, search exponents on first few primes
    for k in range(max_k, max(8, max_k - 6), -1):
        if time.time() > deadline - 30:
            break

        base_primes = primes[:k]
        base_product = 1
        for p in base_primes:
            base_product *= p

        if base_product > LIMIT:
            continue

        remaining_budget = LIMIT // base_product

        # Search exponents for first 4 primes (2, 3, 5, 7)
        # Use iterative deepening with pruning
        def search_exponents(idx, current_mult, extra_exps):
            if time.time() > deadline - 10:
                return

            if idx == 4:
                factorization = []
                for i in range(k):
                    if i < 4:
                        factorization.append((primes[i], 1 + extra_exps[i]))
                    else:
                        factorization.append((primes[i], 1))
                try_candidate(factorization)
                return

            p = primes[idx]
            max_extra = 0
            temp = remaining_budget // current_mult
            while temp >= p:
                temp //= p
                max_extra += 1

            # Limit search depth for efficiency
            max_extra = min(max_extra, 20)

            # Try higher exponents first (more likely to be better)
            for extra in range(max_extra, -1, -1):
                new_mult = current_mult * (p ** extra)
                if new_mult > remaining_budget:
                    continue
                search_exponents(idx + 1, new_mult, extra_exps + [extra])

        search_exponents(0, 1, [])

    # Phase 3: Extended exponent search for 5 primes
    for k in range(max_k, max(9, max_k - 4), -1):
        if time.time() > deadline - 20:
            break

        base_primes = primes[:k]
        base_product = 1
        for p in base_primes:
            base_product *= p

        if base_product > LIMIT:
            continue

        remaining_budget = LIMIT // base_product

        # Search exponents for first 5 primes (2, 3, 5, 7, 11)
        def search_exponents5(idx, current_mult, extra_exps):
            if time.time() > deadline - 10:
                return

            if idx == 5:
                factorization = []
                for i in range(k):
                    if i < 5:
                        factorization.append((primes[i], 1 + extra_exps[i]))
                    else:
                        factorization.append((primes[i], 1))
                try_candidate(factorization)
                return

            p = primes[idx]
            max_extra = 0
            temp = remaining_budget // current_mult
            while temp >= p:
                temp //= p
                max_extra += 1

            max_extra = min(max_extra, 15)

            for extra in range(max_extra, -1, -1):
                new_mult = current_mult * (p ** extra)
                if new_mult > remaining_budget:
                    continue
                search_exponents5(idx + 1, new_mult, extra_exps + [extra])

        search_exponents5(0, 1, [])

    # Phase 4: Local search - try replacing individual primes with nearby ones
    if best_factorization:
        for iteration in range(300):
            if time.time() > deadline - 15:
                break

            improved = False
            current_k = len(best_factorization)

            for pos in range(current_k):
                if time.time() > deadline - 15:
                    break

                current_prime = best_factorization[pos][0]
                cur_idx = prime_to_idx.get(current_prime, 0)

                # Try replacing with nearby primes (wider window)
                search_start = max(0, cur_idx - 30)
                search_end = min(cur_idx + 30, len(primes))

                for new_idx in range(search_start, search_end):
                    if new_idx == cur_idx:
                        continue
                    new_prime = primes[new_idx]

                    new_factorization = []
                    valid = True
                    for j, (p, e) in enumerate(best_factorization):
                        if j == pos:
                            new_factorization.append((new_prime, e))
                        else:
                            if p == new_prime:
                                valid = False
                                break
                            new_factorization.append((p, e))

                    if not valid:
                        continue

                    result = _eval_candidate(new_factorization, LIMIT)
                    if result:
                        n, score = result
                        if score > best_score:
                            best_score = score
                            best_n = n
                            best_factorization = new_factorization[:]
                            improved = True
                            break

                if improved:
                    break

            if not improved:
                break

    # Phase 5: Try adding exponents to existing best (finer search)
    if best_factorization:
        base_k = len(best_factorization)
        for exp_pos in range(min(8, base_k)):
            if time.time() > deadline - 8:
                break

            p, e = best_factorization[exp_pos]
            # Try increasing this exponent
            for extra_exp in range(1, 10):
                new_factorization = []
                for i, (pp, ee) in enumerate(best_factorization):
                    if i == exp_pos:
                        new_factorization.append((pp, ee + extra_exp))
                    else:
                        new_factorization.append((pp, ee))

                result = _eval_candidate(new_factorization, LIMIT)
                if result:
                    n, score = result
                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_factorization = new_factorization[:]

    # Phase 6: Try dropping one prime and increasing exponents on others
    if best_factorization and len(best_factorization) > 10:
        for drop_pos in range(len(best_factorization)):
            if time.time() > deadline - 5:
                break

            reduced_factorization = []
            for i, (p, e) in enumerate(best_factorization):
                if i != drop_pos:
                    reduced_factorization.append((p, e))

            # Try increasing exponents on the first 3 primes of the reduced set
            for e1 in range(1, 6):
                for e2 in range(1, 5):
                    for e3 in range(1, 4):
                        if time.time() > deadline - 5:
                            break

                        new_factorization = []
                        for i, (p, e) in enumerate(reduced_factorization):
                            if i == 0:
                                new_factorization.append((p, e + e1))
                            elif i == 1:
                                new_factorization.append((p, e + e2))
                            elif i == 2:
                                new_factorization.append((p, e + e3))
                            else:
                                new_factorization.append((p, e))

                        result = _eval_candidate(new_factorization, LIMIT)
                        if result:
                            n, score = result
                            if score > best_score:
                                best_score = score
                                best_n = n
                                best_factorization = new_factorization[:]

    return best_n

def construct():
    return _search()
