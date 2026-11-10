import math
import time

N = 479001600  # 12!
PRIMES = (2, 3, 5, 7, 11)
TARGET_EXP = (10, 5, 2, 1, 1)
MAX_FACTORS = 12
DEADLINE = 78.0


def _exponent_vector(n):
    e = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            e[i] += 1
            n //= p
    return tuple(e)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def time_up():
        return time.time() > deadline

    # Divisors of N that are > 12, sorted ascending
    divs = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            for d in (i, N // i):
                if d > 12:
                    divs.append(d)
    divs = sorted(set(divs))
    L = len(divs)
    exps = [_exponent_vector(d) for d in divs]

    # Prefix products for lower-bound pruning in DFS
    prefix = [1] * (L + 1)
    for i in range(L):
        prefix[i + 1] = prefix[i] * divs[i]

    best_factors = None
    best_max = None

    # DFS to extract a witness: distinct divisors (indices increasing) with
    # product exactly N, all <= M (enforced by caller via index limit).
    # Returns True on success, storing in best_factors.
    def dfs(start_idx, max_idx, cur_exp, cur_prod, path):
        nonlocal best_factors
        if time_up():
            return False
        if cur_prod == N:
            if cur_exp == TARGET_EXP and len(path) >= 1:
                best_factors = list(path)
                best_max = path[-1]
                return True
            return False
        if cur_prod > N:
            return False
        for i in range(start_idx, max_idx + 1):
            d = divs[i]
            if cur_prod * d > N:
                break
            e = exps[i]
            if (cur_exp[0] + e[0] > TARGET_EXP[0] or
                    cur_exp[1] + e[1] > TARGET_EXP[1] or
                    cur_exp[2] + e[2] > TARGET_EXP[2] or
                    cur_exp[3] + e[3] > TARGET_EXP[3] or
                    cur_exp[4] + e[4] > TARGET_EXP[4]):
                continue
            # Lower bound: can we still reach N with remaining distinct divisors?
            # We need product multiplier = N // cur_prod. The smallest we can
            # add with (max_idx+1 - (i+1)) more picks is bounded; just rely on
            # the cur_prod*d <= N and exponent checks (space is tiny).
            path.append(d)
            if dfs(i + 1, max_idx,
                   (cur_exp[0] + e[0], cur_exp[1] + e[1],
                    cur_exp[2] + e[2], cur_exp[3] + e[3],
                    cur_exp[4] + e[4]),
                   cur_prod * d, path):
                path.pop()
                return True
            path.pop()
        return False

    # Scan M ascending. For each M, consider divisors <= M (indices 0..hi).
    # Use bitmask DP to test feasibility: reachable exponent vectors.
    # State: set of exponent vectors achievable. Represent as set of tuples.
    for hi in range(L):
        if time_up():
            break
        M = divs[hi]
        # Feasibility via DP over subsets is heavy; use set-of-exponent-vectors
        # reachable by picking distinct divisors among [0..hi].
        # Start: {zero vector, product 1}. Track (exp, prod) but prod is
        # determined by exp? No—different divisors can share exp? No, exp
        # vector uniquely determines the divisor (since divisors of N). So
        # exp vector <=> divisor. Good: state = set of exponent vectors.
        # But we need product == N i.e. exp == TARGET_EXP, AND distinct
        # divisors. Since each exp vector maps to exactly one divisor, a set
        # of distinct divisors = a set of distinct exp vectors.
        # DP: reachable = set of exp vectors formable as sum of distinct
        # chosen exp vectors (each used at most once), each <= hi.
        # This is subset-sum over exp vectors. Number of distinct reachable
        # exp vectors <= number of divisors = 792. Manageable.
        reachable = {(0, 0, 0, 0, 0)}
        for i in range(hi + 1):
            e = exps[i]
            new = set()
            for v in reachable:
                nv = (v[0] + e[0], v[1] + e[1], v[2] + e[2],
                      v[3] + e[3], v[4] + e[4])
                if (nv[0] <= TARGET_EXP[0] and nv[1] <= TARGET_EXP[1] and
                        nv[2] <= TARGET_EXP[2] and nv[3] <= TARGET_EXP[3] and
                        nv[4] <= TARGET_EXP[4]):
                    new.add(nv)
            reachable |= new
            if TARGET_EXP in reachable:
                break
        if TARGET_EXP in reachable:
            # Feasible with max factor M. Extract witness via DFS restricted
            # to indices <= hi.
            best_factors = None
            if dfs(0, hi, (0, 0, 0, 0, 0), 1, []):
                break
            # If DFS failed (shouldn't), continue to next M
        # Early stop: M can't be less than 13, and 13 is minimal possible max
        if best_max is not None and best_max == 13:
            break

    if best_factors is None:
        return [N]
    return best_factors
