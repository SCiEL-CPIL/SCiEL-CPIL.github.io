"""Program for: Large sets with distinct subset products.

Uses a constructive approach based on prime factorization.
We partition the candidate integers into groups based on their prime factors.
For each prime, we select a subset of its powers such that their exponents
form a set with distinct subset sums. This ensures that the combined set
has distinct subset products.
"""

# EVOLVE-BLOCK-START

def _max_subset_with_distinct_sums(exponents):
    """
    Find the largest subset of `exponents` such that all subset sums are distinct.
    We use a simple greedy/backtracking search for small sets.
    """
    n = len(exponents)
    best_subset = []
    best_size = 0
    
    # Try all subsets? 2^n is small for n<=5.
    # But we want the largest.
    # We can just check all subsets of size k from n down to 1.
    for k in range(n, 0, -1):
        found = False
        # itertools.combinations
        import itertools
        for combo in itertools.combinations(exponents, k):
            # Check if subset sums are distinct
            sums = set()
            valid = True
            # Generate all 2^k subset sums
            # Use bitmask
            for mask in range(1 << k):
                s = 0
                for i in range(k):
                    if mask & (1 << i):
                        s += combo[i]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if valid:
                best_subset = list(combo)
                best_size = k
                found = True
                break
        if found:
            break
            
    return best_subset

def construct():
    """Return a list of distinct integers in 2..40 maximizing count such that
    all subset products are distinct."""
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    result = []
    
    for p in primes:
        # Find powers of p <= 40
        powers = []
        exp = 1
        val = p
        while val <= 40:
            powers.append(exp)
            exp += 1
            val *= p
            
        # Find max subset of exponents with distinct subset sums
        best_exps = _max_subset_with_distinct_sums(powers)
        
        # Add corresponding numbers
        for e in best_exps:
            result.append(p ** e)
            
    # Verify the result
    def _is_valid_set(numbers):
        if not numbers:
            return True
        if len(numbers) < 2:
            return True
        
        # Check distinct subset products
        # We can just generate all products and check for duplicates
        # k <= 16, so 2^16 = 65536 products. This is fast.
        products = set()
        k = len(numbers)
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= numbers[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    if not _is_valid_set(result):
        # Fallback to primes if construction failed (should not happen)
        return primes
        
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
