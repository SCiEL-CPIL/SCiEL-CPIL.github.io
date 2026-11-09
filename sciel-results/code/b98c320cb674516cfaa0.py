<<<<<<< SEARCH
    # ---- Phase 3: random restarts / simulated annealing for a few iterations ----
    if time_ok(25):
        # Get current best as starting point
        # Find a k-sized set near best_n's prime factorization
        def factorize(n):
            res = []
            d = 2
            while d * d <= n:
                if n % d == 0:
                    res.append(d)
                    while n % d == 0:
                        n //= d
                d += 1 if d == 2 else 2
            if n > 1:
                res.append(n)
            return res

        # Use best_n's primes as a seed
        try:
            seed_primes = factorize(best_n)
        except Exception:
            seed_primes = PRIMES[:8]

        k0 = len(seed_primes)
        for restart in range(50):
            if not time_ok(20):
                break
            # Start from a random-ish set of size k0 or k0±1
            k = k0 + rng.choice([-1, 0, 0, 1])
            if k < 2 or k > 16:
                continue
            # Random subset of primes, biased toward small but expanded pool
            idx_pool = list(range(0, min(len(PRIMES), 100)))
            rng.shuffle(idx_pool)
            chosen = sorted(PRIMES[i] for i in idx_pool[:k])
            if _product(chosen) > MAX_N:
                # Fallback to smallest k
                chosen = PRIMES[:k]
                if _product(chosen) > MAX_N:
                    continue
            cur = list(chosen)
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_score = eval_set(cur)

            # Simulated annealing with wider neighborhood
            temp = 2.0
            for it in range(150):
                if not time_ok(20):
                    break
                i = rng.randrange(k)
                removed = cur[i]
                idx = PRIMES.index(removed)
                # Candidate new primes - expanded neighborhood
                cands = []
                for j in range(max(0, idx - 5), min(len(PRIMES), idx + 60)):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        continue
                    cands.append(np)
                if not cands:
                    continue
                np = rng.choice(cands)
                new_list = cur[:i] + [np] + cur[i+1:]
                new_list.sort()
                divs = _divisors_sorted(new_list)
                s = _tau_perp(divs)
                new_n = _product(new_list)
                # Accept if better, or with probability based on temperature
                if s > cur_score:
                    cur = new_list
                    cur_n = new_n
                    cur_set = set(cur)
                    cur_score = s
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                else:
                    # Accept worse moves with decreasing probability
                    delta = cur_score - s
                    if delta < 10 and rng.random() < math.exp(-delta / temp):
                        cur = new_list
                        cur_n = new_n
                        cur_set = set(cur)
                        cur_score = s
                temp *= 0.98
=======
    # ---- Phase 3: random restarts / simulated annealing for a few iterations ----
    if time_ok(25):
        def factorize(n):
            res = []
            d = 2
            while d * d <= n:
                if n % d == 0:
                    res.append(d)
                    while n % d == 0:
                        n //= d
                d += 1 if d == 2 else 2
            if n > 1:
                res.append(n)
            return res

        try:
            seed_primes = factorize(best_n)
        except Exception:
            seed_primes = PRIMES[:8]

        k0 = len(seed_primes)
        for restart in range(80):
            if not time_ok(20):
                break
            k = k0 + rng.choice([-1, 0, 0, 1])
            if k < 2 or k > 16:
                continue
            idx_pool = list(range(0, min(len(PRIMES), 120)))
            rng.shuffle(idx_pool)
            chosen = sorted(PRIMES[i] for i in idx_pool[:k])
            if _product(chosen) > MAX_N:
                chosen = PRIMES[:k]
                if _product(chosen) > MAX_N:
                    continue
            cur = list(chosen)
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_score = eval_set(cur)

            temp = 2.0
            for it in range(200):
                if not time_ok(20):
                    break
                i = rng.randrange(k)
                removed = cur[i]
                idx = PRIMES.index(removed)
                cands = []
                for j in range(max(0, idx - 8), min(len(PRIMES), idx + 80)):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        continue
                    cands.append(np)
                if not cands:
                    continue
                np = rng.choice(cands)
                new_list = cur[:i] + [np] + cur[i+1:]
                new_list.sort()
                s, new_n = _score_set(new_list)
                if s > cur_score:
                    cur = new_list
                    cur_n = new_n
                    cur_set = set(cur)
                    cur_score = s
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                else:
                    delta = cur_score - s
                    if delta < 10 and rng.random() < math.exp(-delta / temp):
                        cur = new_list
                        cur_n = new_n
                        cur_set = set(cur)
                        cur_score = s
                temp *= 0.98
>>>>>>> REPLACE

