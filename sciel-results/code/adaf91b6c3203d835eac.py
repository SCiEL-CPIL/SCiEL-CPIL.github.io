import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 84.0  # safety margin under 90s wall-clock

    def full_check(S):
        """Check if set S is valid: for all a<b<=c in S, a does not divide b+c."""
        elems = sorted(S)
        m = len(elems)
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                for k in range(j, m):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if adding x to set S keeps it valid. S is a set, x not in S.
        Correctly handles the three roles of x: smallest (a), middle (b), largest (c),
        including the b=c case (which only matters when b=c=x, i.e. x is middle)."""
        if x in S:
            return True
        S_list = sorted(S)

        # Case 1: a = x, b,c in S with x < b <= c, x | (b+c)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False

        # Case 2: b = x, a,c in S with a < x <= c, a | (x+c)
        # (covers b=c=x when c=x, since x <= x)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Case 3: c = x, a,b in S with a < b <= x, a | (b+x)
        for a in S_list:
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False

        return True

    # Base solution: upper tail [171, 256] (86 elements)
    best_set = set(range(171, 257))
    best_size = len(best_set)

    # Divisor-count priority: smaller numbers with few divisors create fewer conflicts.
    # Precompute for the candidate pool [1,170].
    def divisor_count(k):
        return sum(1 for d in range(1, k + 1) if k % d == 0)

    cand_pool = list(range(1, 171))
    div_counts = {k: divisor_count(k) for k in cand_pool}
    by_div = sorted(cand_pool, key=lambda k: div_counts[k])

    def greedy(order):
        S = set(range(171, 257))
        for x in order:
            if time.time() > deadline - 6:
                break
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S

    # Multiple greedy passes with different orderings.
    orders = [
        list(range(170, 0, -1)),
        list(range(1, 171)),
        by_div,
    ]
    # A few random shuffles of the divisor-priority and natural orders.
    for _ in range(3):
        o = list(by_div)
        random.shuffle(o)
        orders.append(o)
    for _ in range(2):
        o = list(range(1, 171))
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline - 8:
            break
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Local search: remove a few random elements, then greedily re-add (fill) to repair.
    for iteration in range(400):
        if time.time() > deadline - 4:
            break

        S_trial = set(best_set)
        S_list = list(S_trial)
        random.shuffle(S_list)
        num_remove = random.randint(1, 3)
        for i in range(min(num_remove, len(S_list))):
            S_trial.remove(S_list[i])

        # Re-add candidates in divisor-priority order to repair.
        candidates = [x for x in by_div if x not in S_trial]
        for x in candidates:
            if time.time() > deadline - 4:
                break
            if can_add(S_trial, x):
                S_trial.add(x)

        if len(S_trial) > best_size:
            best_size = len(S_trial)
            best_set = set(S_trial)

    # Final validation; fall back to the known-valid tail if anything is off.
    if full_check(best_set) and len(best_set) > 0:
        return list(best_set)
    return list(range(171, 257))
