"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import reduce
from collections import defaultdict

def _squarefree_part(n):
    """Compute the squarefree part of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def _is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    # The squarefree part of 210 is 210 itself (2*3*5*7, all to power 1)
    target_sf = 210
    
    # First try single element: 210 * x is square => x = 210 * k^2
    # x > 210 => k >= 1, x <= 10210 => k^2 <= 48.6 => k <= 6
    best_result = None
    best_max = 10211
    
    # Single element candidates
    for k in range(1, 7):
        x = 210 * k * k
        if 210 < x <= 10210:
            if _is_perfect_square(210 * x):
                if x < best_max:
                    best_max = x
                    best_result = [x]
    
    # Try to find smaller max with multiple elements
    # We need distinct integers in (210, M] whose squarefree parts multiply to 210
    # Search M from 211 upward
    
    # Precompute squarefree parts for numbers in range
    # For efficiency, search incrementally
    
    # Try pairs: sf(a) * sf(b) = 210
    # Try triples: sf(a) * sf(b) * sf(c) = 210
    # etc.
    
    # Let's search for the minimum M such that there exist distinct a1,...,ak in (210, M]
    # with sf(a1)*...*sf(ak) = 210
    
    # Generate all numbers > 210 up to some bound with their squarefree parts
    # and look for subsets whose sf-product is 210
    
    # Since 210 = 2*3*5*7, the sf-parts of selected numbers must have prime factors
    # only from {2,3,5,7} and their product must be exactly 210.
    
    # Possible sf values that can appear: divisors of 210 that are > 1
    # sf values: 2,3,5,6,7,10,14,15,21,30,35,42,70,105,210
    
    # For a number n, sf(n) must divide 210 (since product of sfs = 210)
    # Actually no - sf(n) can be any squarefree number, but the product must be 210.
    # So each sf(n) must be a divisor of 210? No, they just need to multiply to 210.
    # Each sf(n) must be squarefree and the product of all sfs = 210.
    
    # Let me enumerate possible sf values and find numbers with those sf values
    
    # For each squarefree divisor d of 210, find numbers n in (210, 10210] with sf(n) = d
    # Then find a subset of these numbers (distinct) whose sf product = 210
    
    # Squarefree divisors of 210:
    sf_divisors = []
    for d in range(1, 211):
        if d * d > 210:
            break
        if 210 % d == 0:
            sf_divisors.append(d)
            if d != 210 // d:
                sf_divisors.append(210 // d)
    sf_divisors = sorted(set(sf_divisors))
    # Only keep squarefree ones
    sf_divisors = [d for d in sf_divisors if _squarefree_part(d) == d]
    
    # For each sf divisor, find the smallest number > 210 with that sf
    # n = sf * k^2, and we need sf * k^2 > 210
    sf_to_min_num = {}
    for d in sf_divisors:
        if d == 1:
            continue
        # n = d * k^2 > 210 => k^2 > 210/d => k > sqrt(210/d)
        k_min = math.isqrt(math.ceil(210 / d))
        if k_min * k_min <= 210 / d:
            k_min += 1
        # Actually: d * k^2 > 210 => k^2 > 210/d
        k_min = math.isqrt(int(210 / d)) + 1
        while d * k_min * k_min <= 210:
            k_min += 1
        n_min = d * k_min * k_min
        if n_min <= 10210:
            sf_to_min_num[d] = n_min
    
    # Now we need to find a multiset of sf values (with distinct numbers) that multiply to 210
    # and minimize the max of the corresponding numbers
    
    # Since we want distinct numbers, and for each sf value d we can have multiple numbers
    # (d*k^2 for different k), we should pick the smallest available number for each sf.
    
    # Let's find all combinations of sf values that multiply to 210
    # Use DFS
    
    def find_sf_combinations(target, available_sfs, current_comb, results):
        if target == 1:
            results.append(list(current_comb))
            return
        for i, sf in enumerate(available_sfs):
            if target % sf == 0:
                current_comb.append(sf)
                find_sf_combinations(target // sf, available_sfs[i:], current_comb, results)
                current_comb.pop()
    
    # We can use each sf value at most... well, we need distinct numbers, so we can use
    # the same sf value multiple times as long as we have distinct numbers with that sf.
    # For simplicity, let's first try using each sf at most once, then relax.
    
    # Actually, let's think differently. The product of sfs must be 210 = 2*3*5*7.
    # Each sf is squarefree. So we're partitioning the prime factors {2,3,5,7} among
    # the selected numbers' sf values.
    
    # Number of ways to partition 4 primes into groups: Bell number B(4) = 15
    # But we also need the numbers to be distinct and > 210.
    
    # Let me enumerate all partitions of {2,3,5,7} into nonempty subsets
    primes = [2, 3, 5, 7]
    
    def get_partitions(elements):
        if not elements:
            yield []
            return
        first = elements[0]
        for partition in get_partitions(elements[1:]):
            # Add first to each existing subset
            for i in range(len(partition)):
                new_partition = partition[:i] + [partition[i] + [first]] + partition[i+1:]
                yield new_partition
            # Add first as new subset
            yield partition + [[first]]
    
    best_max_found = best_max
    best_set_found = best_result
    
    for partition in get_partitions(primes):
        # Each subset gives an sf value = product of primes in subset
        sf_values = [reduce(lambda a, b: a * b, subset) for subset in partition]
        
        # For each sf value, find the smallest number > 210 with that sf
        # and ensure all numbers are distinct
        numbers = []
        valid = True
        used_numbers = set()
        
        for sf in sf_values:
            # Find smallest n = sf * k^2 > 210, not in used_numbers
            k = math.isqrt(int(210 / sf)) + 1
            while sf * k * k <= 210:
                k += 1
            found = False
            while k * k * sf <= 10210:
                n = sf * k * k
                if n not in used_numbers:
                    numbers.append(n)
                    used_numbers.add(n)
                    found = True
                    break
                k += 1
            if not found:
                valid = False
                break
        
        if valid and len(numbers) >= 1:
            max_val = max(numbers)
            if max_val < best_max_found:
                best_max_found = max_val
                best_set_found = numbers[:]
        
        if time.time() > deadline:
            break
    
    # Also try using more numbers (some with sf=1 won't help, but we can have
    # redundant factors that cancel)
    # Actually, if sf(n) = 1, it doesn't change the product, so it's useless.
    # What about having extra pairs? E.g., two numbers with the same sf value.
    # sf(a) * sf(b) = d * d = d^2, which is a square, so it doesn't change the sf product.
    # So adding pairs with the same sf doesn't help.
    
    # What about 3 numbers where sfs multiply to a square times 210? No, we need exactly 210.
    
    # Let me also try: can we use a number with sf that's not a divisor of 210?
    # If sf(a) = d and d doesn't divide 210, then for the product to be 210,
    # we'd need another number with sf that has the extra factors. But since all sfs
    # are squarefree, the product's prime factors are the union (with parity).
    # For the product to be exactly 210 = 2*3*5*7, each prime must appear an odd
    # number of times total, and no other primes can appear.
    # So each sf can only contain primes from {2,3,5,7}, and the XOR (mod 2) of
    # exponents for each prime must be 1.
    
    # This is equivalent to: the sf values form a multiset where the product is 210.
    # Which is what we're already doing.
    
    # Verify the best result
    if best_set_found is None:
        best_set_found = [840]  # fallback
        best_max_found = 840
    
    # Final verification
    product = 210
    for x in best_set_found:
        product *= x
    if not _is_perfect_square(product):
        # Fallback to single element
        best_set_found = [840]
        best_max_found = 840
    
    # Ensure types are correct
    result = [int(x) for x in best_set_found]
    return result

# EVOLVE-BLOCK-END
