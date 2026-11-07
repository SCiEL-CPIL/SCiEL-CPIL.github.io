"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # 210 = 2*3*5*7, squarefree. We need distinct ints in (210, 10210]
    # whose product * 210 is a perfect square.
    # Equivalently, the squarefree part of the product of selected numbers must be 210.
    # For a number n, sf(n) is its squarefree kernel. We need prod(sf(a_i)) = 210
    # (since sf is multiplicative up to squares, and 210 is squarefree).
    # Actually: sf(210 * prod(a_i)) = 1 iff sf(prod(a_i)) = sf(210) = 210.
    # And sf(prod(a_i)) = prod(sf(a_i)) mod squares, but since each sf(a_i) is squarefree,
    # the product's squarefree part is the "xor" of prime exponents mod 2.
    # For the total to be exactly 210=2*3*5*7, we need the product of sf(a_i) to be
    # 210 times a perfect square. But since each sf(a_i) is squarefree, their product
    # being 210 * (square) means each prime in {2,3,5,7} appears odd times total,
    # and no other primes appear.
    # Simplest: product of sf(a_i) = 210 exactly (each prime appears exactly once).

    best_max = 10211
    best_set = None

    # Strategy: enumerate all valid sf-multisets whose product is 210,
    # then for each, greedily assign the smallest distinct numbers.
    # sf values can only use primes {2,3,5,7}.
    # The sf values form a partition of the multiset {2,3,5,7} into groups.
    # Each group's product is an sf value.

    primes = [2, 3, 5, 7]

    def get_partitions(elements):
        if not elements:
            yield []
            return
        first = elements[0]
        for partition in get_partitions(elements[1:]):
            for i in range(len(partition)):
                new_partition = partition[:i] + [partition[i] + [first]] + partition[i+1:]
                yield new_partition
            yield partition + [[first]]

    # For a given sf value d, find the smallest number n = d * k^2 with n > 210
    def min_num_for_sf(d, exclude=None):
        """Find smallest n = d*k^2 > 210, n <= 10210, n not in exclude."""
        if exclude is None:
            exclude = set()
        k = math.isqrt(210 // d) + 1
        # Ensure d*k^2 > 210
        while d * k * k <= 210:
            k += 1
        while d * k * k <= 10210:
            n = d * k * k
            if n not in exclude:
                return n
            k += 1
        return None

    # Enumerate all partitions of {2,3,5,7}
    for partition in get_partitions(primes):
        sf_values = []
        for subset in partition:
            prod = 1
            for p in subset:
                prod *= p
            sf_values.append(prod)

        # For each sf value, assign the smallest available number
        numbers = []
        used = set()
        valid = True
        for sf in sf_values:
            n = min_num_for_sf(sf, used)
            if n is None:
                valid = False
                break
            numbers.append(n)
            used.add(n)

        if valid and len(numbers) >= 1:
            max_val = max(numbers)
            if max_val < best_max:
                best_max = max_val
                best_set = numbers[:]

        if time.time() > deadline:
            break

    # Also consider: can we use numbers with sf values that aren't pure divisors of 210?
    # E.g., sf=6 and sf=35: 6*35=210. Already covered by partition [2,3],[5,7].
    # What about using the same sf twice? sf=6, sf=6, sf=35/6... no, must be integer.
    # sf=2, sf=2, sf=52.5 - not integer. So no.
    # What about 3 numbers where two have the same sf? sf=2, sf=2, sf=52.5 - no.
    # The product must be exactly 210, and each sf is squarefree using only {2,3,5,7}.
    # If we use sf=2 twice, product has 2^2, so we'd need to cancel it, but sfs are squarefree
    # so we can't have negative exponents. Thus each prime can appear in at most one sf value
    # if we want the product to be exactly 210. Wait, no: sf=6 and sf=10: 6*10=60, not 210.
    # sf=2 and sf=105: 2*105=210. sf=105=3*5*7. That's partition [2],[3,5,7].
    # So we're covering all cases.

    # Additional search: try pairs directly for robustness
    # Factor pairs of 210: (d, 210/d)
    for d in range(1, 211):
        if 210 % d == 0:
            d2 = 210 // d
            n1 = min_num_for_sf(d)
            n2 = min_num_for_sf(d2, {n1} if n1 else None)
            if n1 is not None and n2 is not None and n1 != n2:
                max_val = max(n1, n2)
                if max_val < best_max:
                    best_max = max_val
                    best_set = [n1, n2]

        if time.time() > deadline:
            break

    # Fallback
    if best_set is None:
        best_set = [840]
        best_max = 840

    # Verify
    product = 210
    for x in best_set:
        product *= x
    if not _is_perfect_square(product):
        best_set = [840]
        best_max = 840

    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
