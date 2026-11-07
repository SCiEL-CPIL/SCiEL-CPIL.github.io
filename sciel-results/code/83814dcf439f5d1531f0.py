import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    # ---------- precompute number properties ----------
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def num_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # ---------- core incremental machinery ----------
    def get_all_sums(S):
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums

    def check_divisibility_free(sums):
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            ai = sorted_sums[i]
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % ai == 0:
                    return False
        return True

    def can_add(S, x, current_sums):
        """Return (ok, new_sums) where new_sums are the sums that involve x.
        Fast rejection: check new sums against old sums AND among themselves."""
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)

        # duplicate-sum check (two disjoint subsets with same sum -> invalid)
        if current_sums & new_sums:
            return False, None

        # divisibility: new vs old
        for s_new in new_sums:
            for s_old in current_sums:
                if s_new > s_old:
                    if s_new % s_old == 0:
                        return False, None
                else:
                    if s_old % s_new == 0:
                        return False, None

        # divisibility: new vs new
        nl = list(new_sums)
        for i in range(len(nl)):
            a = nl[i]
            for j in range(i + 1, len(nl)):
                b = nl[j]
                if a > b:
                    if a % b == 0:
                        return False, None
                else:
                    if b % a == 0:
                        return False, None

        return True, new_sums

    # ---------- candidate orders ----------
    candidates = list(range(1, 257))
    primes = [x for x in candidates if is_prime(x)]
    few_factors = sorted([x for x in candidates if num_factors(x) <= 2], reverse=True)

    def biased_order():
        # large prime-ish numbers first (proven prior), random tie-break
        scored = [( -num_factors(x), -x, random.random(), x) for x in candidates]
        scored.sort()
        return [x for _, _, _, x in scored]

    # ---------- branch and bound ----------
    best_set = []
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    def branch_and_bound(order, limit):
        S = []
        current_sums = set()
        used = set()

        def rec(i):
            nonlocal best_size
            if time.time() > limit:
                return
            # upper bound: can we possibly beat best?
            if len(S) + (256 - len(used)) <= best_size:
                return
            if len(S) > best_size:
                update_best(S)
            if i >= len(order):
                return
            x = order[i]
            if x in used:
                rec(i + 1)
                return
            ok, new_sums = can_add(S, x, current_sums)
            if ok:
                S.append(x)
                old = current_sums
                current_sums = old | new_sums
                used.add(x)
                rec(i + 1)
                used.remove(x)
                current_sums = old
                S.pop()
            rec(i + 1)

        rec(0)

    def greedy(order, limit):
        S = []
        current_sums = set()
        for x in order:
            if time.time() > limit:
                break
            ok, new_sums = can_add(S, x, current_sums)
            if ok:
                S.append(x)
                current_sums = current_sums | new_sums
        update_best(S)
        return S

    # ---------- run search ----------
    # 1) a couple of strong deterministic biased orders
    for _ in range(3):
        if time.time() > deadline - 20.0:
            break
        branch_and_bound(biased_order(), deadline - 20.0)

    # 2) many randomized greedy restarts (cheap, broad coverage)
    while time.time() < deadline - 10.0:
        order = biased_order()
        greedy(order, deadline - 10.0)

    # 3) local search on the best set: swap one element out, add another
    if best_set and time.time() < deadline - 2.0:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            base = best_set.copy()
            improved = False
            for i in range(len(base)):
                if time.time() > deadline - 2.0:
                    break
                trial = base[:i] + base[i + 1:]
                trial_sums = get_all_sums(trial)
                used = set(trial)
                # try to add a better element
                add_candidates = [x for x in candidates if x not in used]
                random.shuffle(add_candidates)
                for x in add_candidates:
                    if time.time() > deadline - 2.0:
                        break
                    ok, new_sums = can_add(trial, x, trial_sums)
                    if ok:
                        new_set = trial + [x]
                        if len(new_set) > len(base):
                            update_best(new_set)
                            base = new_set.copy()
                            improved = True
                        break
            if not improved:
                break

    # ---------- verify and return ----------
    if best_size >= 2:
        all_sums = get_all_sums(best_set)
        if not check_divisibility_free(all_sums):
            # fallback to a guaranteed-valid small set
            best_set = [11, 13, 17]
            best_size = 3
    elif best_size < 3:
        best_set = [11, 13, 17]
        best_size = 3

    return [int(x) for x in best_set]
