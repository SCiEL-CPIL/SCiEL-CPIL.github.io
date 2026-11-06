import time
from fractions import Fraction

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
    deadline = start_time + 110  # safety margin before 150s kill

    primes = _sieve(1000)
    CAP = 100000

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

    def greedy_init(window, eff, deadline):
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

                if eff:
                    score = cnt * p
                else:
                    score = cnt

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

    # Phase 1: Multiple greedy strategies with different windows and efficiency metrics
    strategies = []
    for w in [500, 1000, 2000, 3000, 5000, 8000, 12000, 20000]:
        for eff in [True, False]:
            strategies.append((w, eff))

    # Sort to try mid-range windows with efficiency first
    strategies.sort(key=lambda x: (not x[1], abs(x[0] - 3000)))

    for w, eff in strategies:
        if time.time() > deadline - 10:
            break
        greedy_init(w, eff, deadline)

    # Phase 2: Local search refinement
    if time.time() < deadline - 5 and best_classes:
        prime_res = {p: r for p, r in best_classes}
        used = set(prime_res.keys())
        budget_used = sum(Fraction(1, p) for p in prime_res)
        remaining = Fraction(1) - budget_used

        covered = make_coverage(prime_res)
        L = compute_L(covered)

        if L < CAP:
            no_improve = 0
            max_no_improve = 30

            while time.time() < deadline - 2 and no_improve < max_no_improve:
                gap = L + 1
                improved = False

                # Move 1: Try reassigning residue of an existing prime to cover gap
                candidates = list(prime_res.items())
                import random
                random.shuffle(candidates)
                candidates = candidates[:15]

                for p, r in candidates:
                    if time.time() > deadline - 2:
                        return
                    new_r = gap % p
                    if new_r == r:
                        continue

                    # Try this reassignment
                    old_r = r
                    new_cov = covered[:]
                    
                    # Unmark old residue (simplified: just rebuild from scratch for correctness)
                    # Actually, let's just try adding new primes first as it's safer
                    
                    pass  # Skip complex reassignment for now

                # Move 2: Add an unused prime with residue covering gap
                unused = [p for p in primes if p not in used and remaining >= Fraction(1, p)]
                random.shuffle(unused)
                unused = unused[:8]

                for p in unused:
                    if time.time() > deadline - 2:
                        return
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

                # Move 3: Try swapping - remove one prime and add another
                if not improved:
                    for p_remove in list(used):
                        if time.time() > deadline - 2:
                            return
                        # Try removing p_remove and see if we can add a better prime
                        # This is complex; skip for now to ensure speed
                        
                        pass

                if not improved:
                    no_improve += 1

        if L > best_L:
            best_L = L
            best_classes = build_classes(prime_res)

    if not best_classes:
        # Fallback: Use small primes to ensure valid return
        small = [p for p in primes if p <= 97]
        total_cost = sum(Fraction(1, p) for p in small)
        while total_cost > Fraction(1) and small:
            small.pop()
            total_cost = sum(Fraction(1, p) for p in small)
        best_classes = [[p, 0] for p in small]

    return best_classes
