import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    K = len(primes)
    LOG_LIMIT = math.log(10**18)

    # Precompute prime powers up to a reasonable max exponent
    # Max exponent for 2: 2^59 ~ 5.7e17, so 59 is max
    # For larger primes, max exponent is smaller
    max_exponents = []
    prime_powers = []  # prime_powers[i] = [1, p, p^2, ...]
    for i, p in enumerate(primes):
        powers = [1]
        exp = 0
        while powers[-1] * p <= 10**18:
            powers.append(powers[-1] * p)
            exp += 1
        prime_powers.append(powers)
        max_exponents.append(exp)

    def calc_score(exponents):
        # Check product limit using logs for speed
        log_n = 0
        for e in exponents:
            if e > 0:
                log_n += e * math.log(primes[exponents.index(e) if e != 0 else 0])  # This is wrong, let me fix
        # Actually, let me redo this properly
        log_n = 0
        for i, e in enumerate(exponents):
            if e > 0:
                log_n += e * math.log(primes[i])
        if log_n > LOG_LIMIT:
            return -1

        # Generate divisors
        divs = [1]
        for i, e in enumerate(exponents):
            if e == 0:
                continue
            p = primes[i]
            powers = prime_powers[i][:e + 1]
            new_divs = []
            for d in divs:
                for pow_p in powers:
                    new_divs.append(d * pow_p)
            divs = new_divs

        divs.sort()

        # Count coprime adjacent pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score

    # Initial state: All 1s (squarefree with all 15 primes)
    current_exp = [1] * K
    current_score = calc_score(current_exp)
    best_exp = current_exp.copy()
    best_score = current_score

    # Deterministic local search with multiple restarts
    def local_search(exp, score, time_limit):
        nonlocal best_exp, best_score
        current = exp.copy()
        cur_score = score
        no_improve = 0
        max_no_improve = 10000

        while time.time() < time_limit and no_improve < max_no_improve:
            improved = False
            # Try all single-exponent changes
            for i in range(K):
                for delta in [-1, 1]:
                    new_exp = current.copy()
                    new_exp[i] += delta
                    if new_exp[i] < 0 or new_exp[i] > max_exponents[i]:
                        continue
                    new_score = calc_score(new_exp)
                    if new_score < 0:
                        continue
                    if new_score > cur_score:
                        current = new_exp
                        cur_score = new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_exp = current.copy()
                        break
                if improved:
                    break
            if not improved:
                # Try double moves
                for i in range(K):
                    for j in range(i + 1, K):
                        for di in [-1, 1]:
                            for dj in [-1, 1]:
                                new_exp = current.copy()
                                new_exp[i] += di
                                new_exp[j] += dj
                                if new_exp[i] < 0 or new_exp[i] > max_exponents[i]:
                                    continue
                                if new_exp[j] < 0 or new_exp[j] > max_exponents[j]:
                                    continue
                                new_score = calc_score(new_exp)
                                if new_score < 0:
                                    continue
                                if new_score > cur_score:
                                    current = new_exp
                                    cur_score = new_score
                                    improved = True
                                    if cur_score > best_score:
                                        best_score = cur_score
                                        best_exp = current.copy()
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                no_improve += 1
            else:
                no_improve = 0

        return current, cur_score

    # Run local search from squarefree start
    current_exp, current_score = local_search(current_exp, current_score, deadline)

    # Try a few different initial configurations
    initial_configs = [
        [1] * K,  # Squarefree
        [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Extra 2
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Extra 3
        [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Extra 5
        [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # Extra large prime
    ]

    for init in initial_configs:
        if time.time() > deadline:
            break
        score = calc_score(init)
        if score < 0:
            continue
        if score > current_score:
            current_exp, current_score = local_search(init, score, deadline)

    # Construct n from best_exp
    n = 1
    for i, e in enumerate(best_exp):
        n *= prime_powers[i][e]

    if n > 10**18:
        # Fallback to squarefree primorial of first 15 primes
        return 614889782588491410

    if n < 2:
        return 2

    return n
