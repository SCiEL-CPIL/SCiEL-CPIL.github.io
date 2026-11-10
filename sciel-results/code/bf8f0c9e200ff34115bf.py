import itertools
from functools import lru_cache

def construct():
    # We want the largest subset of {2..40} whose subset products are all distinct.
    # Equivalent: no non-empty subset has product 1 (impossible here) and no two distinct subsets have same product.
    # Greedy approach: try to add numbers in an order that maximizes the chance of keeping all subset products distinct.
    # 
    # Key insight: if we represent each number by its prime factorization (exponent vector), then subset products are distinct
    # iff the exponent vectors are linearly independent over the rationals (i.e., no non-trivial integer linear combination gives zero
    # with coefficients in {-1,0,1} corresponding to subset differences).
    #
    # However, a simpler sufficient condition: if we pick numbers such that no number divides another, and more strongly,
    # the prime exponent vectors are "superincreasing" in some sense, we can guarantee distinctness.
    #
    # Better approach: use backtracking with pruning. We have 39 numbers to choose from (2..40). We want the largest subset.
    # 
    # Actually, let's think about what makes subset products distinct. If we have a set S, then for any two distinct subsets A, B of S,
    # prod(A) != prod(B). This is equivalent to: for any non-empty multiset difference (A\B, B\A), the product of elements in A\B
    # is not equal to the product of elements in B\A.
    #
    # A well-known construction: use powers of 2. {2, 4, 8, 16, 32} gives 5 numbers. But we can do better by mixing primes.
    #
    # Let's use a greedy algorithm with a twist: sort candidates by some heuristic and greedily add if it preserves distinctness.
    # To maximize the count, we should try different orderings or use a smarter selection.
    #
    # Alternative: use the fact that if all numbers are prime, then subset products are distinct (by unique factorization).
    # Primes in 2..40: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes.
    # Can we add composite numbers? If we add a composite, say 4=2^2, then we need to check that no subset of existing primes
    # multiplies to 4 (impossible since 4 requires 2^2 and we only have one 2). But wait, if we have 2 and 4, then {4} and {2,2}... 
    # but we can't use 2 twice. The subsets are sets, not multisets. So {4} has product 4, and {2} has product 2. No conflict.
    # But {2,4} has product 8, and if we had 8, then {8} and {2,4} both give 8. So we can't have 2, 4, and 8 together.
    #
    # So the constraint is: for any element x in the set, x must not be expressible as a product of a subset of the other elements.
    # More generally, no subset product should equal another subset product.
    #
    # Let's try a backtracking search. We have 39 elements. We want the maximum size subset.
    # We can use branch and bound.
    #
    # Actually, let me just try a greedy approach with different orderings and see what we get.
    # Order by: largest first, smallest first, by number of prime factors, etc.
    
    candidates = list(range(2, 41))
    
    def is_valid_subset(subset):
        """Check if all subset products are distinct."""
        if not subset:
            return True
        products = set()
        # Generate all 2^k subset products
        k = len(subset)
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= subset[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    # Greedy: try adding elements in different orders
    # Let's try: sort by value descending
    def greedy(order):
        selected = []
        for x in order:
            candidate = selected + [x]
            if is_valid_subset(candidate):
                selected.append(x)
        return selected
    
    # Try multiple orderings
    orderings = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        sorted(candidates, key=lambda x: -x.bit_count()),  # by popcount descending
        sorted(candidates, key=lambda x: x.bit_count()),   # by popcount ascending
    ]
    
    best = []
    for ordering in orderings:
        result = greedy(ordering)
        if len(result) > len(best):
            best = result
    
    # Also try a backtracking approach for potentially better results
    # Since 39 elements is too many for full backtracking, we'll do a limited search
    # 
    # Actually, let's just use the best greedy result for now.
    # Convert to list of Python ints
    return list(best)
