"""Rewritten program: maximum independent set on conflict graph, with divisor-of-512 structural shortcut."""
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_valid_subset(s):
    """Check that every pair has lcm <= 512."""
    n = len(s)
    for i in range(n):
        ai = s[i]
        for j in range(i + 1, n):
            if _lcm(ai, s[j]) > 512:
                return False
    return True

def _divisors_of_512():
    """All positive divisors of 512 = 2^9, i.e. {1,2,4,...,512}."""
    d = []
    p = 1
    while p <= 512:
        d.append(p)
        p *= 2
    return d

def construct():
    start = time.time()
    deadline = 60.0  # safety margin under 90s kill

    # Structural observation: the set of all divisors of 512 is a valid subset,
    # because for any a, b | 512, lcm(a, b) | 512, so lcm(a, b) <= 512.
    # This gives 10 elements (1,2,4,...,512). We can do far better.

    # The true maximum is achieved by taking ALL numbers whose prime-power
    # content fits within 512's factorization. Since 512 = 2^9, any number
    # composed only of powers of 2 (i.e., divisors of 512) trivially works,
    # but we can also include odd numbers and mixed numbers as long as
    # lcm with every other chosen element stays <= 512.

    # Strategy: build the conflict graph on [1,512], where edge (a,b) exists
    # iff lcm(a,b) > 512. Then find a large independent set.
    #
    # Key structural insight for a strong initial solution:
    # Any number a with a | 512 has gcd(a, 512) = a, so lcm(a, b) = a*b/gcd(a,b).
    # For the full divisor set D = {1,2,...,512} (powers of 2), it's valid.
    #
    # We can extend: include odd numbers too. If a is odd and a <= 512, then
    # lcm(a, 2^k) = a * 2^k (since gcd(a, 2^k)=1). This is <= 512 iff a * 2^k <= 512.
    # So for a given odd a, the largest power of 2 we can pair with it is 2^(9 - ceil(log2(a))).
    #
    # A very large valid set: all numbers a in [1,512] such that a divides some
    # number <= 512... that's all of them. The real constraint is pairwise lcm.
    #
    # Let's use a greedy + local search approach on the conflict graph.
    # But first, let's get a strong seed: all divisors of 512 (10 elements)
    # plus odd numbers that fit.

    # Actually, the simplest large valid set: take all a in [1,512] where a is
    # a divisor of 512. That's 10. We can do much better.
    #
    # Better seed: take all numbers a where a | 512 OR (a is odd and a <= 512).
    # Wait, that's not valid either because lcm(3, 256) = 768 > 512.
    #
    # Correct approach: use the conflict graph and greedy independent set.

    N = 512
    # Build conflict adjacency: conflict[a][b] = True if lcm(a,b) > 512
    # Use a set of sets for efficiency
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if _lcm(a, b) > 512:
                conflicts[a].add(b)
                conflicts[b].add(a)
        if time.time() > deadline - 10:
            break

    # Greedy minimum-degree independent set
    available = set(range(1, N + 1))
    independent_set = []

    while available:
        # Pick vertex with minimum degree in the conflict graph restricted to available
        min_deg = len(available)
        min_v = None
        for v in available:
            deg = len(conflicts[v] & available)
            if deg < min_deg:
                min_deg = deg
                min_v = v
                if min_deg == 0:
                    break

        if min_v is None:
            break

        independent_set.append(min_v)
        available.discard(min_v)
        # Remove all neighbors of min_v from available
        available -= conflicts[min_v]

        if time.time() > deadline - 5:
            break

    # Local search: try to add more elements by swapping
    # Try adding each removed element if it doesn't conflict with the set
    current_set = set(independent_set)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        # Try to add any element not in current_set that has no conflicts with current_set
        for a in range(1, N + 1):
            if a in current_set:
                continue
            if not (conflicts[a] & current_set):
                current_set.add(a)
                improved = True
        # Try swaps: remove one, add one
        if not improved:
            for a in list(current_set):
                for b in range(1, N + 1):
                    if b in current_set:
                        continue
                    new_set = current_set - {a} | {b}
                    # Check if b conflicts with any element in new_set
                    if not (conflicts[b] & (new_set - {b})):
                        # Also check that removing a and adding b doesn't create issues
                        # (it won't, since we only need pairwise lcm <= 512)
                        current_set = new_set
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try 2-for-1 swaps
                for a in list(current_set):
                    for b in list(current_set):
                        if a >= b:
                            continue
                        for c in range(1, N + 1):
                            if c in current_set:
                                continue
                            new_set = current_set - {a, b} | {c}
                            if not (conflicts[c] & (new_set - {c})):
                                current_set = new_set
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

    result = sorted(current_set)

    # Validate
    if not _is_valid_subset(result):
        # Fallback to safe divisor set
        result = _divisors_of_512()

    return result
