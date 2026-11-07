# EVOLVE-BLOCK-START
import math
from itertools import product as iproduct

def construct():
    N = 210
    MAX_VAL = 10210

    # Step 1: Precompute squarefree kernel for every integer in [211, 10210]
    # Using a sieve-like approach for efficiency
    sf = [0] * (MAX_VAL + 1)
    sf[1] = 1
    for i in range(2, MAX_VAL + 1):
        if sf[i] == 0:
            # i is prime
            sf[i] = i
            for j in range(i * i, MAX_VAL + 1, i):
                if sf[j] == 0:
                    sf[j] = i
                else:
                    # j already has some kernel; multiply by i if i not already in kernel
                    # Actually, we need to be more careful. Let's use a different approach.
                    pass

    # Actually, let's use a cleaner method: for each number, compute its squarefree kernel
    # by trial division (since MAX_VAL is only ~10210, this is fine)
    sf = [0] * (MAX_VAL + 1)
    sf[1] = 1
    for x in range(2, MAX_VAL + 1):
        n = x
        kernel = 1
        d = 2
        while d * d <= n:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt % 2 == 1:
                kernel *= d
            d += 1
        if n > 1:
            kernel *= n
        sf[x] = kernel

    # Step 2: The squarefree kernel of 210 is 210 itself (2*3*5*7, all to power 1)
    # We need a subset of integers in (210, 10210] such that the product of their
    # squarefree kernels equals 210.
    #
    # Represent each kernel as a bitmask over primes {2,3,5,7}:
    #   bit 0: prime 2
    #   bit 1: prime 3
    #   bit 2: prime 5
    #   bit 3: prime 7
    # A kernel k has a bitmask mask(k) where bit i is set iff the corresponding prime
    # appears to an odd power in k.
    #
    # We need the XOR (bitwise XOR) of the masks of selected numbers to equal mask(210).
    # mask(210) = mask(2*3*5*7) = all four bits set = 0b1111 = 15.
    #
    # Additionally, the product of the kernels must equal 210, not just have the same
    # squarefree part. Since each kernel is squarefree and the product of squarefree
    # numbers is squarefree iff no prime appears in more than one kernel, we need the
    # kernels to be pairwise coprime AND their product to be exactly 210.
    #
    # So we need a partition of the prime set {2,3,5,7} into groups, where each group
    # corresponds to one selected number's kernel, and each kernel must be a divisor
    # of 210 (since the product must be 210).
    #
    # The divisors of 210 that are > 1 and squarefree: 2,3,5,6,7,10,14,15,21,30,35,42,70,105,210
    #
    # We need to find, for each such divisor d of 210 (d>1), the smallest integer x in
    # (210, 10210] with sf[x] == d. Then we need to select a collection of these x's
    # such that the product of their d's is exactly 210, and minimize the maximum x.
    #
    # This is a small combinatorial problem. Let's enumerate all ways to write 210 as
    # a product of distinct squarefree divisors > 1 of 210 (order doesn't matter, and
    # we can use at most one of each divisor since they must be pairwise coprime in a
    # way that their product is 210).
    #
    # Actually, the condition is: the product of the selected kernels equals 210, and
    # each kernel is a squarefree divisor of 210. This means the kernels form a
    # multiplicative partition of 210 into squarefree factors > 1.

    # Find the smallest x > 210 with sf[x] == d for each d | 210, d > 1
    min_x_for_kernel = {}
    for d in range(2, 211):
        if 210 % d == 0:
            # d is a divisor of 210
            # Find smallest x in (210, MAX_VAL] with sf[x] == d
            for x in range(211, MAX_VAL + 1):
                if sf[x] == d:
                    min_x_for_kernel[d] = x
                    break
            else:
                min_x_for_kernel[d] = None  # not found

    # Now find all multiplicative partitions of 210 into squarefree factors > 1
    # We'll use recursion
    def find_partitions(n, min_factor, current, results):
        if n == 1:
            results.append(tuple(sorted(current)))
            return
        for f in range(min_factor, n + 1):
            if n % f == 0:
                # f must be squarefree and a divisor of 210
                if 210 % f == 0:
                    # Check f is squarefree: f divides 210 and gcd(f, 210//f) ... 
                    # Actually, since f divides 210 and 210 is squarefree, f is automatically squarefree
                    current.append(f)
                    find_partitions(n // f, f, current, results)
                    current.pop()

    partitions = []
    find_partitions(210, 2, [], partitions)

    # For each partition, compute the maximum of min_x_for_kernel[d] for d in partition
    # Choose the partition with the smallest such maximum
    best_max = float('inf')
    best_partition = None
    for part in partitions:
        max_val = 0
        valid = True
        for d in part:
            if min_x_for_kernel.get(d) is None:
                valid = False
                break
            if min_x_for_kernel[d] > max_val:
                max_val = min_x_for_kernel[d]
        if valid and max_val < best_max:
            best_max = max_val
            best_partition = part

    if best_partition is None:
        # Fallback
        return [4 * N]

    # Select the numbers
    selected = [min_x_for_kernel[d] for d in best_partition]
    selected.sort()

    # Verify
    prod = 1
    for num in selected:
        prod *= num
    total = prod * N
    root = math.isqrt(total)
    if root * root != total:
        # Fallback
        return [4 * N]

    return selected
# EVOLVE-BLOCK-END
