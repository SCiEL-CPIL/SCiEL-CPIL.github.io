def construct():
    import time
    from math import gcd
    import random

    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Precompute gcd table
    gcd_table = [[0] * 129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(a, 129):
            g = gcd(a, b)
            gcd_table[a][b] = g
            gcd_table[b][a] = g

    def get_gcd(a, b):
        return gcd_table[a][b]

    def num_distinct_pf(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def total_pf(n):
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

    def num_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count

    all_numbers = list(range(1, 129))

    def can_add(selected, x):
        """Return True if adding x to selected keeps the no-equal-gcd-triple property."""
        for i in range(len(selected)):
            a = selected[i]
            ga = get_gcd(a, x)
            for j in range(i + 1, len(selected)):
                b = selected[j]
                gb = get_gcd(b, x)
                gab = get_gcd(a, b)
                if ga == gb == gab:
                    return False
        return True

    def greedy(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected

    def is_valid_set(selected):
        n = len(selected)
        for i in range(n):
            a = selected[i]
            for j in range(i + 1, n):
                b = selected[j]
                for k in range(j + 1, n):
                    if get_gcd(a, b) == get_gcd(a, c if False else k) == get_gcd(b, selected[k]):
                        pass
        # simpler explicit
        for i in range(n):
            a = selected[i]
            for j in range(i + 1, n):
                b = selected[j]
                for k in range(j + 1, n):
                    c = selected[k]
                    if get_gcd(a, b) == get_gcd(a, c) == get_gcd(b, c):
                        return False
        return True

    best = []

    # Ordering 1: by decreasing distinct prime factors
    order1 = sorted(all_numbers, key=lambda x: (-num_distinct_pf(x), -total_pf(x), -num_divisors(x), x))
    best = greedy(order1)

    # Ordering 2: by decreasing total prime factors
    order2 = sorted(all_numbers, key=lambda x: (-total_pf(x), -num_distinct_pf(x), x))
    r2 = greedy(order2)
    if len(r2) > len(best):
        best = r2

    # Ordering 3: by decreasing number of divisors
    order3 = sorted(all_numbers, key=lambda x: (-num_divisors(x), x))
    r3 = greedy(order3)
    if len(r3) > len(best):
        best = r3

    # Natural and reversed
    r4 = greedy(all_numbers)
    if len(r4) > len(best):
        best = r4
    r5 = greedy(list(reversed(all_numbers)))
    if len(r5) > len(best):
        best = r5

    # Random orderings
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        order = all_numbers[:]
        random.shuffle(order)
        r = greedy(order)
        if len(r) > len(best):
            best = r

    def local_search(selected, unselected, time_budget):
        """Improve via: add-anywhere, remove-1-add-1 (enabling growth), remove-1-add-2."""
        selected = list(selected)
        selected_set = set(selected)
        unselected = [x for x in all_numbers if x not in selected_set]

        # Pass A: add any element that fits
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for x in unselected[:]:
                if can_add(selected, x):
                    selected.append(x)
                    selected_set.add(x)
                    unselected.remove(x)
                    improved = True

        # Pass B: remove-1-add-1 swap that enables a later add (net +1)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for y in selected[:]:
                if time.time() > time_budget:
                    break
                temp = [s for s in selected if s != y]
                # try to add any x to temp
                for x in unselected:
                    if can_add(temp, x):
                        new_sel = temp + [x]
                        new_set = set(new_sel)
                        # check if we can now add another z (net +1)
                        for z in unselected:
                            if z == x:
                                continue
                            if can_add(new_sel, z):
                                selected = new_sel + [z]
                                selected_set = set(selected)
                                unselected = [u for u in all_numbers if u not in selected_set]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

        # Pass C: remove-1-add-2 (net +1)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for y in selected[:]:
                if time.time() > time_budget:
                    break
                temp = [s for s in selected if s != y]
                for i in range(len(unselected)):
                    x1 = unselected[i]
                    if not can_add(temp, x1):
                        continue
                    new_sel = temp + [x1]
                    for j in range(len(unselected)):
                        if i == j:
                            continue
                        x2 = unselected[j]
                        if can_add(new_sel, x2):
                            selected = new_sel + [x2]
                            selected_set = set(selected)
                            unselected = [u for u in all_numbers if u not in selected_set]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return selected

    # Local search on best
    best = local_search(best, [x for x in all_numbers if x not in set(best)], deadline - 3)

    # A couple of local searches on other good greedy results
    for r in (r2, r3, r4, r5):
        if time.time() > deadline - 5:
            break
        cand = local_search(r, [x for x in all_numbers if x not in set(r)], deadline - 3)
        if len(cand) > len(best):
            best = cand

    # Final fill: add anything that fits
    selected_set = set(best)
    for x in all_numbers:
        if x not in selected_set and can_add(best, x):
            best.append(x)
            selected_set.add(x)

    # Guarantee non-empty
    if not best:
        best = [1]

    return best
