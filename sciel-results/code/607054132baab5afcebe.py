def construct():
    import time
    start = time.time()
    deadline = start + 80  # Safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def _check_distinct(subset):
        """Check if all 2^n subset products of 'subset' are distinct."""
        n = len(subset)
        seen = set()
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= subset[i]
            if prod in seen:
                return False
            seen.add(prod)
        return True

    # Known valid 16-element set using prime powers with superincreasing exponents
    base = [8, 16, 32, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    base_set = set(base)

    # Try to add one more element to reach 17
    candidates = [x for x in range(2, 41) if x not in base_set]
    for extra in candidates:
        if time.time() > deadline:
            break
        test = base + [extra]
        if _check_distinct(test):
            return [int(x) for x in test]

    # Try swapping one element out and two in (net +1)
    for i in range(len(base)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(base)):
            if time.time() > deadline:
                break
            removed = {base[i], base[j]}
            remaining_base = [x for x in base if x not in removed]
            for a in candidates:
                if time.time() > deadline:
                    break
                for b in candidates:
                    if b <= a:
                        continue
                    if a in removed or b in removed:
                        continue
                    test = remaining_base + [a, b]
                    if _check_distinct(test):
                        return [int(x) for x in test]

    # Fallback: return the base set
    return [int(x) for x in base]
