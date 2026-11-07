"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80
    N = 256

    # Sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N+1) if is_prime[i]]
    prime_set = set(primes)

    # Factorization
    fac_cache = {}
    def get_fac(n):
        if n in fac_cache:
            return fac_cache[n]
        f = {}
        d = 2
        tmp = n
        while d * d <= tmp:
            while tmp % d == 0:
                f[d] = f.get(d, 0) + 1
                tmp //= d
            d += 1
        if tmp > 1:
            f[tmp] = f.get(tmp, 0) + 1
        fac_cache[n] = f
        return f

    for i in range(2, N+1):
        get_fac(i)

    def a_divides_bc(a, b, c):
        for p, e in fac_cache[a].items():
            if fac_cache[b].get(p, 0) + fac_cache[c].get(p, 0) < e:
                return False
        return True

    def is_valid(S):
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            if is_prime[a]:
                for j in range(n):
                    if j != i and S_list[j] % a == 0:
                        return False
            else:
                for j in range(n):
                    if j == i:
                        continue
                    b = S_list[j]
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if a_divides_bc(a, b, S_list[k]):
                            return False
        return True

    best = set(primes)
    best_size = len(best)

    all_nums = list(range(2, N+1))
    composites = [x for x in all_nums if not is_prime[x]]

    import random
    rng = random.Random(42)

    def can_add_to(x, S):
        for a in S:
            for b in S:
                if a == b:
                    continue
                if a_divides_bc(a, x, b) or a_divides_bc(x, a, b) or a_divides_bc(b, a, x):
                    return False
        return True

    def can_coexist(x1, x2, S):
        for c in S:
            if a_divides_bc(x1, x2, c) or a_divides_bc(x2, x1, c) or a_divides_bc(c, x1, x2):
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur = set(best)

        # Phase 1: add composite, remove up to 2
        for x in composites:
            if x in cur:
                continue
            if time.time() > deadline:
                break
            bad = set()
            for a in cur:
                for b in cur:
                    if a == b:
                        continue
                    if a_divides_bc(a, x, b) or a_divides_bc(x, a, b) or a_divides_bc(b, a, x):
                        bad.add(a)
                        bad.add(b)
                        break
            if len(bad) <= 2 and len(cur) - len(bad) + 1 > best_size:
                new_set = (cur - bad) | {x}
                if is_valid(new_set):
                    best = new_set
                    best_size = len(best)
                    cur = new_set
                    improved = True

        # Phase 2: remove 1, add 2
        if time.time() < deadline:
            for a in list(cur):
                if time.time() > deadline:
                    break
                base = cur - {a}
                addable = []
                for x in composites:
                    if x in base:
                        continue
                    if can_add_to(x, base):
                        addable.append(x)
                if len(addable) >= 2:
                    addable = addable[:10]
                    for i in range(len(addable)):
                        for j in range(i+1, len(addable)):
                            if time.time() > deadline:
                                break
                            x1, x2 = addable[i], addable[j]
                            if can_coexist(x1, x2, base):
                                new_set = base | {x1, x2}
                                if len(new_set) > best_size:
                                    if is_valid(new_set):
                                        best = new_set
                                        best_size = len(best)
                                        cur = new_set
                                        improved = True

        # Phase 3: random restarts if stuck
        if not improved and time.time() < deadline - 5:
            for _ in range(20):
                if time.time() > deadline:
                    break
                perturbed = set(best)
                to_rm = rng.sample(list(perturbed), min(rng.randint(1,3), len(perturbed)))
                for a in to_rm:
                    perturbed.discard(a)
                for _ in range(rng.randint(1,3)):
                    x = rng.choice(composites)
                    if x not in perturbed:
                        perturbed.add(x)
                if len(perturbed) > best_size and is_valid(perturbed):
                    best = perturbed
                    best_size = len(best)
                    improved = True
                    break

    if not is_valid(best):
        return list(primes)
    return sorted(best)

# EVOLVE-BLOCK-END
