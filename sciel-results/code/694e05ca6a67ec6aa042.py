import time
from math import factorial

TARGET = factorial(12)  # 479001600
DEADLINE = 78.0  # safety margin before 90s kill

# Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7 * 11
PRIMES = [2, 3, 5, 7, 11]
EXP_MAX = [10, 5, 2, 1, 1]

def _get_all_divisors():
    divs = []
    def gen(idx, cur):
        if idx == len(PRIMES):
            divs.append(cur)
            return
        p = PRIMES[idx]
        v = cur
        for _ in range(EXP_MAX[idx] + 1):
            gen(idx + 1, v)
            v *= p
    gen(0, 1)
    divs.sort()
    return divs

ALL_DIVS = _get_all_divisors()
DIV_SET = set(ALL_DIVS)
# Precompute index map for divisors <= some bound (we'll build per-B lists)

def _build_valid(B):
    """Return list of divisors d with 12 < d <= B, sorted."""
    # Use bisect to slice ALL_DIVS
    import bisect
    lo = bisect.bisect_right(ALL_DIVS, 12)
    hi = bisect.bisect_right(ALL_DIVS, B)
    return ALL_DIVS[lo:hi]

def _feasible(B, deadline):
    """
    Check if 12! can be factored into distinct integers in (12, B].
    Meet-in-the-middle:
      Split divisors into low (13..B//2) and high (B//2+1..B).
      Enumerate all subset-products from low that divide TARGET.
      Enumerate all subset-products from high that divide TARGET.
      For each high-subset product h, need a low-subset product l = TARGET // h,
      and l must be achievable by a low subset.
    Also allow single-factor case: B >= TARGET (trivially true if B large).
    """
    if B >= TARGET:
        return True  # [TARGET] is valid
    if B <= 12:
        return False

    valid = _build_valid(B)
    n = len(valid)
    if n == 0:
        return False

    # Single factor check
    if TARGET in DIV_SET and TARGET <= B:
        return True

    # Split
    mid = B // 2
    low = [d for d in valid if d <= mid]
    high = [d for d in valid if d > mid]

    # We need distinct factors. Since low and high are disjoint ranges,
    # any low-subset and high-subset are automatically distinct from each other.
    # Within each half, subsets use distinct elements by construction.

    # Enumerate all subset-products from low that divide TARGET
    # and track the set of such products.
    low_products = set()
    # Also track: for a given product, we just need existence.
    # But we also need the actual subset to be valid (distinct, which is guaranteed).
    # However, we must ensure the subset has at least 1 element if product > 1.
    # Empty subset gives product 1.

    # Use iterative enumeration
    # For performance, prune subsets whose product doesn't divide TARGET
    # or exceeds TARGET.
    def _enumerate_subsets(arr, deadline):
        """Yield (product, mask) for subsets where product divides TARGET and product <= TARGET."""
        results = set()
        n = len(arr)
        # Use recursive generation with pruning
        def rec(idx, prod, mask):
            if time.time() > deadline:
                return
            if prod > TARGET:
                return
            if TARGET % prod != 0:
                # Can't recover by multiplying more (all factors > 1)
                # Actually we can't prune here because we haven't added all elements.
                # But if prod doesn't divide TARGET, multiplying by more divisors
                # won't help since all factors divide TARGET. Actually if prod | TARGET
                # fails, then prod * x | TARGET for any x | TARGET is not guaranteed.
                # Better: only continue if prod | TARGET.
                return
            if idx == n:
                results.add(prod)
                return
            # Skip arr[idx]
            rec(idx + 1, prod, mask)
            # Take arr[idx]
            new_prod = prod * arr[idx]
            if new_prod <= TARGET and TARGET % new_prod == 0:
                rec(idx + 1, new_prod, mask | (1 << idx))
        rec(0, 1, 0)
        return results

    # To keep it fast, limit subset size implicitly by product constraint.
    # Enumerate low products
    t0 = time.time()
    low_prods = _enumerate_subsets(low, deadline)
    if time.time() > deadline:
        return False  # assume infeasible, will be caught by binary search

    # For high, we don't need to store all; we can enumerate and check complement.
    # But for symmetry and to handle the case where high has many elements,
    # let's also enumerate high products and check intersection.
    high_prods = _enumerate_subsets(high, deadline)
    if time.time() > deadline:
        return False

    # Now check: exists h in high_prods, l in low_prods such that l * h == TARGET
    # and l != 1 or h != 1 (i.e., at least one factor total)
    # Since l * h = TARGET, and TARGET > 1, at least one is > 1.
    # We need the union of the two subsets to be non-empty.
    # l = 1 means empty low subset; h = 1 means empty high subset.
    # We need NOT (l == 1 and h == 1), but l*h = TARGET > 1 guarantees that.

    # Check intersection: for each h in high_prods, l = TARGET // h must be in low_prods
    for h in high_prods:
        if TARGET % h != 0:
            continue
        l = TARGET // h
        if l in low_prods:
            # Verify we're using distinct factors: since low and high are disjoint
            # value ranges, and within each subset elements are distinct, we're fine.
            # Also need l >= 1 and h >= 1, which is true.
            # But we must ensure the combined set has distinct integers > 12.
            # By construction, all elements are in valid (12, B], and low/high are disjoint.
            return True
    return False

def construct():
    global DEADLINE
    deadline = time.time() + DEADLINE

    # Binary search on B
    # Lower bound: at least 13 (smallest valid factor)
    # Upper bound: TARGET (single factor)
    lo = 13
    hi = TARGET

    best_B = TARGET  # fallback

    # First, check if the trivial single-factor solution is the only option
    # We'll binary search for the minimum feasible B.

    # To speed up, we can do exponential search first if needed, but TARGET is fixed.
    # Just binary search directly.
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if _feasible(mid, deadline):
            hi = mid
            best_B = mid
        else:
            lo = mid + 1

    # Now we have best_B = minimum feasible max factor.
    # We need to actually construct the factors.
    # Let's do a DFS with the bound best_B to find an actual decomposition.
    # This DFS should be fast because best_B is tight.

    valid = _build_valid(best_B)
    n = len(valid)

    best_factors = []

    def dfs(idx, rem, last, current):
        if time.time() > deadline:
            return False
        if rem == 1:
            if current:
                best_factors.extend(current)
                return True
            return False

        # We need to pick factors from valid[idx:] that are > last
        # Actually, we should pick from valid starting at idx, but ensure > last.
        # Since valid is sorted, we can start from idx.
        # But we need distinct, so we pick each at most once.

        # Pruning: if rem is a divisor of TARGET and rem > last and rem <= best_B
        # and rem not in current, we could take it as the last factor.
        # But we need to ensure we can fill with distinct factors.

        # Simple approach: try each valid[i] for i >= idx where valid[i] > last
        for i in range(idx, n):
            if time.time() > deadline:
                return False
            d = valid[i]
            if d <= last:
                continue
            if d > best_B:
                break
            if rem % d != 0:
                continue
            current.append(d)
            if dfs(i + 1, rem // d, d, current):
                return True
            current.pop()
        return False

    if not dfs(0, TARGET, 12, []):
        # Fallback: return [TARGET]
        return [TARGET]

    return best_factors
