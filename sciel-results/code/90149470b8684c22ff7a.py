"""Rewritten program for the suite task; stronger algorithm from scratch."""
import itertools
import random

# EVOLVE-BLOCK-START

def _is_valid_set(nums):
    """Check if the set of distinct integers satisfies the divisibility-incomparable subset sum condition."""
    # Compute all nonempty subset sums
    n = len(nums)
    sums = set()
    # Use iterative approach to generate all subset sums
    # For n up to 30, 2^30 is too large, but we can prune or use a smarter check.
    # However, for validation of candidate sets we generate all sums.
    # If n is large, this might be slow, but we'll only validate small candidates or use pruning.
    # Actually, for n=30, 2^30 ~ 1e9 which is too slow.
    # We need a smarter validation or only validate small sets during search.
    # For the final construct, we'll return a precomputed valid set or a small valid set.
    # Let's implement a check that works for small n, and for larger n we rely on the construction being correct.
    
    # For now, implement full check for n <= 20, and for larger n we assume validity if constructed properly.
    if n > 20:
        # Use a different strategy: check pairwise divisibility of all subset sums is expensive.
        # Instead, we'll just return True and rely on the construction method.
        # But we need to be sure. Let's implement a check that works up to n=25 or so.
        # Actually, let's just do the full check but with early termination.
        pass
    
    sums = set()
    # Generate all nonempty subset sums
    for i in range(1, 1 << n):
        s = 0
        for j in range(n):
            if i & (1 << j):
                s += nums[j]
        if s in sums:
            return False  # Duplicate sum
        sums.add(s)
    
    # Check no two distinct sums divide one another
    sum_list = sorted(sums)
    for i in range(len(sum_list)):
        for j in range(i + 1, len(sum_list)):
            if sum_list[j] % sum_list[i] == 0:
                return False
    return True

def _try_build_set(max_size, max_val, seed=42):
    """Try to build a valid set of distinct integers in 1..max_val with size up to max_size."""
    random.seed(seed)
    
    # Start with a base set and try to add elements
    # Greedy approach: start with empty, add elements that maintain validity
    # But validity check is expensive for large sets.
    
    # Alternative: use a precomputed set that we know works, or search for one.
    # Let's try a heuristic: use numbers that are pairwise coprime and large enough
    # that subset sums don't divide each other.
    
    # A known approach: use powers of 2 times a large prime, or just use numbers in a range
    # where all subset sums are in a range where no one divides another.
    
    # Simple approach: pick numbers from a range [L, 2L) so that all subset sums of k elements
    # are in [kL, k*2L). If we ensure that for any two subset sums s1 < s2, s2/s1 < 2,
    # then s2 cannot be a multiple of s1 (since the next multiple would be 2*s1).
    # This works if all numbers are in [L, 2L) and we limit the number of elements.
    
    # Actually, a cleaner approach: use numbers that are all in (M/2, M] for some M.
    # Then any subset sum of k elements is in (k*M/2, k*M].
    # For two subset sums s1 (with k1 elements) and s2 (with k2 elements), 
    # if k2 > k1, then s2/s1 > (k2*M/2)/(k1*M) = k2/(2*k1).
    # If k2 >= 2*k1, then s2/s1 > 1, but we need s2/s1 < 2 to prevent divisibility.
    # This doesn't directly help.
    
    # Let's just try a random search with a limited number of elements.
    # For now, return a small valid set.
    
    # Precomputed valid set: [31, 32, 34] works (from the seed).
    # Let's try to find a larger one.
    
    # Try to build a set of size 4, 5, etc.
    for target_size in range(4, max_size + 1):
        # Try random sets
        for _ in range(100):
            candidate = random.sample(range(1, max_val + 1), target_size)
            if _is_valid_set(candidate):
                return candidate
    
    # If we couldn't find a larger set, return the known valid set
    return [31, 32, 34]

def construct():
    """Return a list of distinct integers in 1..256 satisfying the condition, maximizing size."""
    # Use the precomputed or searched set
    result = _try_build_set(30, 256, seed=42)
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in result]
    
    # Verify the result is valid
    if not _is_valid_set(result):
        # Fallback to known valid set
        result = [31, 32, 34]
    
    return result

# EVOLVE-BLOCK-END
