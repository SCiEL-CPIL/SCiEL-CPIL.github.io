import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 87.0

    def make_set(lst):
        s = 0
        for x in lst:
            s |= (1 << (x - 1))
        return s

    def is_valid_full(s):
        elems = [i + 1 for i in range(n) if (s >> i) & 1]
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    continue
                if not ((s >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        continue
                    if ((s >> (ac - 1)) & 1) and ((s >> (bc - 1)) & 1):
                        return False
        return True

    def can_add(s, x):
        if (s >> (x - 1)) & 1:
            return True
        # Check if adding x creates a violation
        # For any pair (a, b) in S where a+b is in S, check if x+a and x+b are both in S
        # Or if x+a is in S and x+b is in S and a+b is in S
        elems = [i + 1 for i in range(n) if (s >> i) & 1]
        L = len(elems)
        for i in range(L):
            a = elems[i]
            xa = x + a
            if xa > n:
                continue
            if not ((s >> (xa - 1)) & 1):
                continue
            for j in range(i + 1, L):
                b = elems[j]
                xb = x + b
                if xb > n:
                    continue
                if not ((s >> (xb - 1)) & 1):
                    continue
                ab = a + b
                if ab == x or (ab <= n and ((s >> (ab - 1)) & 1)):
                    return False
        return True

    # Start with seed: 65-128
    best_s = make_set(range(65, 129))
    best_size = 64

    # Multiple greedy passes with different random orderings
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        trial_s = best_s
        candidates = list(range(1, 65))
        random.seed(trial * 7919 + 42)
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline - 30:
                break
            if can_add(trial_s, x):
                trial_s |= (1 << (x - 1))

        # Try adding any remaining element
        for x in range(1, n + 1):
            if time.time() > deadline - 30:
                break
            if (trial_s >> (x - 1)) & 1:
                continue
            if can_add(trial_s, x):
                trial_s |= (1 << (x - 1))

        sz = bin(trial_s).count('1')
        if sz > best_size:
            best_s = trial_s
            best_size = sz

    # Local search on best solution
    for iteration in range(30000):
        if time.time() > deadline - 5:
            break

        in_set = [i + 1 for i in range(n) if (best_s >> i) & 1]
        out_set = [i + 1 for i in range(n) if not ((best_s >> i) & 1)]

        if not out_set:
            break

        improved = False

        # Try single adds
        random.shuffle(out_set)
        for x in out_set:
            if can_add(best_s, x):
                best_s |= (1 << (x - 1))
                best_size += 1
                improved = True
                break
        if improved:
            continue

        # Try remove-one-add-two
        if len(in_set) >= 1 and len(out_set) >= 2:
            random.shuffle(in_set)
            random.shuffle(out_set)
            for rem in in_set[:8]:
                if time.time() > deadline - 5:
                    break
                s2 = best_s & ~(1 << (rem - 1))
                added = 0
                for add in out_set:
                    if add == rem:
                        continue
                    if can_add(s2, add):
                        s2 |= (1 << (add - 1))
                        added += 1
                    if added >= 2:
                        break
                if added >= 2:
                    best_s = s2
                    best_size = bin(best_s).count('1')
                    improved = True
                    break

        # Try remove-two-add-three
        if len(in_set) >= 2 and len(out_set) >= 3:
            random.shuffle(in_set)
            random.shuffle(out_set)
            for i in range(min(5, len(in_set))):
                rem1 = in_set[i]
                for j in range(i + 1, min(5, len(in_set))):
                    rem2 = in_set[j]
                    s2 = best_s & ~(1 << (rem1 - 1)) & ~(1 << (rem2 - 1))
                    added = 0
                    for add in out_set:
                        if add == rem1 or add == rem2:
                            continue
                        if can_add(s2, add):
                            s2 |= (1 << (add - 1))
                            added += 1
                        if added >= 3:
                            break
                    if added >= 3:
                        best_s = s2
                        best_size = bin(best_s).count('1')
                        improved = True
                        break
                if improved:
                    break

        if not improved and iteration > 10000:
            break

    # Final verification
    if not is_valid_full(best_s):
        best_s = make_set(range(65, 129))
        best_size = 64

    result = [i + 1 for i in range(n) if (best_s >> i) & 1]
    return result
