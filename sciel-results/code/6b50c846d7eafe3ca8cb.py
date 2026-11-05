def construct():
    import math
    import time
    import heapq
    from itertools import product

    start_time = time.time()
    deadline = start_time + 145.0

    # Primes up to ~200
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    LIMIT = 10**18

    def calc_score(n):
        if n < 2: return 0
        divs = [1]
        temp = n
        i = 0
        while i < len(primes) and primes[i] * primes[i] <= temp:
            p = primes[i]
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            i += 1
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        divs.sort()
        score = 0
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j+1]) == 1:
                score += 1
        return score

    def calc_score_from_factorization(factorization):
        if not factorization or all(e == 0 for e in factorization.values()):
            return 0
        divs = [1]
        for p, exp in sorted(factorization.items()):
            if exp == 0:
                continue
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        divs.sort()
        score = 0
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j+1]) == 1:
                score += 1
        return score

    def get_n_from_factorization(factorization, base_product=1):
        n = base_product
        for p, exp in factorization.items():
            if exp > 0:
                for _ in range(exp):
                    if n > LIMIT // p:
                        return None
                    n *= p
        return n if n >= 2 and n <= LIMIT else None

    # Start with primorial of first 15 primes
    primorial = 1
    primorial_factorization = {}
    for i, p in enumerate(primes[:15]):
        primorial *= p
        primorial_factorization[p] = 1

    best_n = primorial
    best_score = calc_score(primorial)
    best_factorization = dict(primorial_factorization)

    # Beam search over exponent assignments
    # State: (negative_estimate, n, factorization)
    beam_size = 200
    beam = [(-best_score, best_n, dict(best_factorization))]

    # Primes we can vary: first 12 primes with exponents 0-5
    variable_primes = primes[:12]
    max_exp = 5

    # Generate candidates by varying exponents of first 6 primes
    # Keep other primes at exponent 1
    other_primes = primes[6:15]
    base_product = 1
    base_factorization = {}
    for p in other_primes:
        base_product *= p
        base_factorization[p] = 1

    # Generate initial beam
    seen = {best_n}
    for exps in product(range(max_exp + 1), repeat=6):
        if time.time() > deadline - 30.0:
            break
        factorization = dict(base_factorization)
        n = base_product
        valid = True
        for i, p in enumerate(variable_primes[:6]):
            exp = exps[i]
            if exp > 0:
                factorization[p] = exp
                for _ in range(exp):
                    if n > LIMIT // p:
                        valid = False
                        break
                    n *= p
                if not valid:
                    break
        if not valid or n < 2 or n > LIMIT or n in seen:
            continue
        seen.add(n)
        # Estimate: use divisor count as proxy
        tau = 1
        for e in factorization.values():
            tau *= (e + 1)
        est = tau ** 0.6
        if len(beam) < beam_size:
            heapq.heappush(beam, (-est, n, factorization))
        else:
            if est > -beam[0][0]:
                heapq.heapreplace(beam, (-est, n, factorization))

    # Evaluate beam
    evaluated = set()
    while beam and time.time() < deadline - 20.0:
        neg_est, n, factorization = heapq.heappop(beam)
        if n in evaluated:
            continue
        evaluated.add(n)
        s = calc_score_from_factorization(factorization)
        if s > best_score:
            best_score = s
            best_n = n
            best_factorization = dict(factorization)

    # Local search: try changing one exponent at a time
    for iteration in range(5):
        if time.time() > deadline - 10.0:
            break
        improved = False
        for p in variable_primes:
            if time.time() > deadline - 10.0:
                break
            old_exp = best_factorization.get(p, 0)
            for new_exp in range(max(0, old_exp - 2), min(max_exp, old_exp + 2) + 1):
                if new_exp == old_exp:
                    continue
                trial_factorization = dict(best_factorization)
                if new_exp == 0:
                    trial_factorization.pop(p, None)
                else:
                    trial_factorization[p] = new_exp
                trial_n = get_n_from_factorization(trial_factorization, base_product)
                if trial_n is None or trial_n in evaluated:
                    continue
                trial_score = calc_score_from_factorization(trial_factorization)
                if trial_score > best_score:
                    best_score = trial_score
                    best_n = trial_n
                    best_factorization = trial_factorization
                    improved = True
                    evaluated.add(trial_n)
                    break
        if not improved:
            break

    # Try adding more primes
    for p in primes[15:25]:
        if time.time() > deadline - 5.0:
            break
        trial_factorization = dict(best_factorization)
        trial_factorization[p] = 1
        trial_n = get_n_from_factorization(trial_factorization, base_product)
        if trial_n is None or trial_n in evaluated:
            continue
        trial_score = calc_score_from_factorization(trial_factorization)
        if trial_score > best_score:
            best_score = trial_score
            best_n = trial_n
            best_factorization = trial_factorization
            evaluated.add(trial_n)

    return best_n
