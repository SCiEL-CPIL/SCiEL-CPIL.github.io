"""Egyptian fraction: minimize largest denominator for 5/31.
Strategy: binary search on max denominator M with meet-in-the-middle feasibility check,
then DFS construction. Completely different from greedy+local-search approach."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85
    target = Fraction(5, 31)

    def greedy_solution(min_d=1):
        r = target
        denoms = []
        md = min_d
        used = set()
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < md:
                d = md
            while d in used:
                d += 1
            denoms.append(d)
            r -= Fraction(1, d)
            md = d + 1
            used.add(d)
        return denoms

    best_solution = greedy_solution()
    best_max = max(best_solution)

    def greedy_complete(remaining_frac, used, min_d, max_cap=None):
        terms = []
        r = remaining_frac
        md = min_d
        used = used.copy()
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < md:
                d = md
            if max_cap is not None and d > max_cap:
                return None
            while d in used:
                d += 1
                if max_cap is not None and d > max_cap:
                    return None
            terms.append(d)
            r -= Fraction(1, d)
            md = d + 1
            used.add(d)
        return terms

    def dfs_construct(remaining, min_d, depth, max_cap):
        """DFS to find denominators summing to remaining, all in [min_d, max_cap]."""
        if time.time() > deadline:
            return None
        if remaining == 0:
            return []
        if remaining < 0:
            return None
        if min_d > max_cap or depth > 12:
            return None

        d_min = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if d_min < min_d:
            d_min = min_d
        if d_min > max_cap:
            return None

        # Pruning: max possible sum from remaining denominators
        # sum(1/d for d in range(d_min, max_cap+1)) should be >= remaining
        # Quick bound: (max_cap - d_min + 1) / d_min >= remaining (very loose)
        avail = max_cap - d_min + 1
        if avail <= 0:
            return None
        if avail * Fraction(1, d_min) < remaining:
            return None

        for d in range(d_min, min(d_min + 15, max_cap + 1)):
            if Fraction(1, d) > remaining:
                continue
            result = dfs_construct(remaining - Fraction(1, d), d + 1, depth + 1, max_cap)
            if result is not None:
                return [d] + result
        return None

    def check_and_construct(M):
        """Try to find a solution with all denominators <= M."""
        if time.time() > deadline:
            return None
        return dfs_construct(target, 7, 0, M)

    # Binary search on M
    lo, hi = 7, best_max
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = check_and_construct(mid)
        if sol is not None and max(sol) <= mid:
            hi = mid
            best_solution = sol
            best_max = max(sol)
        else:
            lo = mid + 1

    # Final check at lo
    if lo < best_max:
        sol = check_and_construct(lo)
        if sol is not None:
            best_solution = sol
            best_max = max(sol)

    # Local improvement: try to reduce max denominator
    if best_solution and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_solution)):
                if time.time() > deadline:
                    break
                old_d = best_solution[i]
                for new_d in range(7, old_d):
                    if new_d in best_solution:
                        continue
                    remaining = target
                    new_sol = []
                    for j, d in enumerate(best_solution):
                        if j != i:
                            remaining -= Fraction(1, d)
                            new_sol.append(d)
                    needed = Fraction(1, new_d) - Fraction(1, old_d)
                    if needed == remaining:
                        new_sol.append(new_d)
                        if max(new_sol) < best_max:
                            best_max = max(new_sol)
                            best_solution = new_sol
                            improved = True
                            break
                    # Try re-completing
                    target_rem = remaining + Fraction(1, old_d) - Fraction(1, new_d)
                    if target_rem > 0:
                        used = set(new_sol)
                        min_d = max(new_sol) + 1 if new_sol else 1
                        cap = best_max - 1 if best_max > 7 else None
                        completion = greedy_complete(target_rem, used, min_d, cap)
                        if completion and len(new_sol) + len(completion) <= 256:
                            full = new_sol + completion
                            if max(full) < best_max:
                                best_max = max(full)
                                best_solution = full
                                improved = True
                                break
                if improved:
                    break

    return best_solution
# EVOLVE-BLOCK-END
