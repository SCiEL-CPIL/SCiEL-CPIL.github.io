import time
from fractions import Fraction
import random

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    primes = _sieve(1000)
    CAP = 100000
    random.seed(42)

    best_L = 0
    best_classes = []

    def compute_L(covered):
        L = 0
        for i in range(1, CAP + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def build_classes(prime_res):
        return [[p, prime_res[p]] for p in sorted(prime_res.keys())]

    def make_coverage(prime_res):
        covered = bytearray(CAP + 1)
        for p, r in prime_res.items():
            s = r
            if s == 0:
                s = p
            for k in range(s, CAP + 1, p):
                covered[k] = 1
        return covered

    def greedy_init(deadline):
        nonlocal best_L, best_classes
        covered = bytearray(CAP + 1)
        used = set()
        budget = Fraction(1)
        prime_res = {}

        n = 1
        while n <= CAP:
            if time.time() > deadline:
                break
            if covered[n]:
                n += 1
                continue

            best_p = None
            best_r = 0
            best_score = -1

            window = 3000
            hi = min(n + window, CAP)

            for p in primes:
                if p in used:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue

                r = n % p
                cnt = 0
                k = n
                while k <= hi:
                    if not covered[k]:
                        cnt += 1
                    k += p

                score = cnt * p
                if score > best_score or (score == best_score and (best_p is None or p < best_p)):
                    best_score = score
                    best_p = p
                    best_r = r

            if best_p is None:
                break

            used.add(best_p)
            budget -= Fraction(1, best_p)
            prime_res[best_p] = best_r

            s = best_r
            if s == 0:
                s = best_p
            for k in range(s, CAP + 1, best_p):
                covered[k] = 1

            n += 1

        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)
        return L

    def local_search(classes, deadline):
        nonlocal best_L, best_classes
        if not classes:
            return

        prime_res = {p: r for p, r in classes}
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used

        covered = make_coverage(prime_res)
        L = compute_L(covered)

        if L >= CAP:
            if L > best_L:
                best_L = L
                best_classes = build_classes(prime_res)
            return

        no_improve = 0
        max_no_improve = 100

        while time.time() < deadline and no_improve < max_no_improve:
            gap = L + 1
            improved = False

            # Move 1: Reassign residue of an existing prime
            candidates = list(prime_res.items())
            random.shuffle(candidates)
            for p, r in candidates:
                if time.time() > deadline:
                    return
                new_r = gap % p
                if new_r == r:
                    continue
                new_cov = covered[:]
                s_new = new_r
                if s_new == 0:
                    s_new = p
                for k in range(s_new, CAP + 1, p):
                    new_cov[k] = 1
                L2 = compute_L(new_cov)
                if L2 > L:
                    prime_res[p] = new_r
                    covered = new_cov
                    L = L2
                    no_improve = 0
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_classes = build_classes(prime_res)
                    if L >= CAP:
                        return
                    break

            # Move 2: Add an unused prime
            if not improved:
                unused = [p for p in primes if p not in used]
                random.shuffle(unused)
                for p in unused:
                    if time.time() > deadline:
                        return
                    if remaining < Fraction(1, p):
                        continue
                    r = gap % p
                    new_cov = covered[:]
                    s = r
                    if s == 0:
                        s = p
                    for k in range(s, CAP + 1, p):
                        new_cov[k] = 1
                    L2 = compute_L(new_cov)
                    if L2 > L:
                        prime_res[p] = r
                        used.add(p)
                        remaining -= Fraction(1, p)
                        covered = new_cov
                        L = L2
                        no_improve = 0
                        improved = True
                        if L > best_L:
                            best_L = L
                            best_classes = build_classes(prime_res)
                        if L >= CAP:
                            return
                        break

            # Move 3: Swap - remove one prime, add another
            if not improved:
                for p_remove in list(used):
                    if time.time() > deadline:
                        return
                    old_r = prime_res[p_remove]
                    del prime_res[p_remove]
                    used.remove(p_remove)
                    remaining += Fraction(1, p_remove)

                    cov_no_p = make_coverage(prime_res)
                    L_no_p = compute_L(cov_no_p)
                    gap2 = L_no_p + 1

                    for p_add in primes:
                        if p_add in used:
                            continue
                        if remaining < Fraction(1, p_add):
                            continue
                        r = gap2 % p_add
                        new_cov = cov_no_p[:]
                        s = r
                        if s == 0:
                            s = p_add
                        for k in range(s, CAP + 1, p_add):
                            new_cov[k] = 1
                        L2 = compute_L(new_cov)
                        if L2 > L:
                            prime_res[p_add] = r
                            used.add(p_add)
                            remaining -= Fraction(1, p_add)
                            covered = new_cov
                            L = L2
                            no_improve = 0
                            improved = True
                            if L > best_L:
                                best_L = L
                                best_classes = build_classes(prime_res)
                            if L >= CAP:
                                return
                            break

                    if not improved:
                        # Revert
                        prime_res[p_remove] = old_r
                        used.add(p_remove)
                        remaining -= Fraction(1, p_remove)
                        covered = make_coverage(prime_res)
                        L = compute_L(covered)
                        break

                if not improved:
                    no_improve += 1

        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

    # Phase 1: Greedy
    greedy_init(deadline - 20)

    # Phase 2: Local search
    if time.time() < deadline - 3 and best_classes:
        local_search(best_classes, deadline)

    if not best_classes:
        small = [p for p in primes if p <= 97]
        best_classes = [[p, (p * (i+1)) % p or p] for i, p in enumerate(small)]

    return best_classes
