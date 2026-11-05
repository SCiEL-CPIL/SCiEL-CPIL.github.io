"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time

    start = time.time()
    deadline = 80.0

    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')

    def greedy_complete(remaining, used, terms):
        """Complete the fraction greedily, return the list or None if fails."""
        result = list(terms)
        r = remaining
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in used:
                # Try next larger denominator
                found = False
                for dd in range(d + 1, 10000001):
                    if dd not in used:
                        d = dd
                        found = True
                        break
                if not found:
                    return None
            result.append(d)
            used.add(d)
            r -= Fraction(1, d)
            if len(result) > 256:
                return None
        return result

    def dfs(remaining, used, terms, max_den):
        nonlocal best_solution, best_max_den
        if time.time() > deadline:
            return
        if remaining == 0:
            if terms:
                mx = max(terms)
                if mx < best_max_den:
                    best_max_den = mx
                    best_solution = list(terms)
            return
        if remaining < 0 or len(terms) >= 256:
            return

        min_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if min_d < 1:
            min_d = 1
        start_d = max(min_d, (max(used) + 1) if used else 1)

        # Limit the search range for efficiency
        # We only care about improving best_max_den
        upper = min(max_den, best_max_den - 1) if best_solution else max_den

        for d in range(start_d, upper + 1):
            if time.time() > deadline:
                return
            if d in used:
                continue
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                continue
            # Pruning: if new_rem > 0, we need at least one more term
            # The minimum possible additional sum with denominators > d is bounded
            # Simple prune: if new_rem > sum of 1/(d+1) + 1/(d+2) + ... (harmonic tail)
            # But that's complex. Simple prune: if new_rem >= 1/(d+1), we might still fail
            # but let's just try
            used.add(d)
            terms.append(d)
            dfs(new_rem, used, terms, upper)
            terms.pop()
            used.remove(d)

    # Try first denominators from 7 to 20, with increasing max_den bounds
    for first_d in range(7, 21):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, first_d)
        if rem < 0:
            continue
        used = {first_d}
        terms = [first_d]

        # Try greedy completion first
        sol = greedy_complete(rem, used, terms)
        if sol:
            mx = max(sol)
            if mx < best_max_den:
                best_max_den = mx
                best_solution = sol

        # Try DFS with a bounded max_den
        # Start with a reasonable bound
        for max_den in range(first_d + 1, 500):
            if time.time() > deadline:
                break
            if max_den >= best_max_den:
                break
            dfs(rem, set(used), list(terms), max_den)
            if best_solution and best_max_den <= max_den:
                break

    if best_solution is not None:
        return best_solution

    # Fallback
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
