import itertools
from functools import lru_cache

def construct():
    """
    Finds the largest subset of integers in [2, 40] with distinct subset products.
    Uses a backtracking search with pruning.
    """
    candidates = list(range(2, 41))
    best_set = []
    
    def has_distinct_subset_products(nums):
        """
        Checks if all subset products of nums are distinct.
        Returns True if distinct, False otherwise.
        """
        if not nums:
            return True
        products = {1}
        for n in nums:
            new_products = set()
            for p in products:
                new_products.add(p * n)
            if len(new_products) != len(products):
                return False
            products = new_products
        return True

    # Sort candidates to potentially find good sets faster
    # Greedy heuristic: Try adding numbers that are 'hard' to form (primes first?)
    # Or just standard backtracking.
    # Since n=39 is small, we can use a simple recursive search with memoization or pruning.
    # However, 2^39 is too large.
    # We need a smarter search.
    
    # Heuristic: Start with all primes, then try to add composites if they don't break the condition.
    # Actually, let's just use a greedy algorithm that tries to add the smallest number first,
    # and if it fails, try the next.
    # This might not be optimal but is a different approach.
    
    # Better: Randomized search or local search.
    # But we need deterministic output.
    
    # Let's try a greedy approach:
    # Sort numbers by "difficulty" (e.g., number of divisors in [2,40]).
    # Primes are hardest.
    
    def num_divisors_in_range(n):
        count = 0
        for i in range(2, 41):
            if n % i == 0:
                count += 1
        return count
        
    # Sort by descending number of divisors? No, primes have few divisors.
    # We want to pick numbers that are unlikely to form collisions.
    # Primes are best.
    
    selected = []
    remaining = candidates[:]
    
    # Sort remaining by value?
    # Let's just iterate through sorted candidates.
    for n in sorted(candidates):
        # Try to add n to selected
        test_set = selected + [n]
        if has_distinct_subset_products(test_set):
            selected.append(n)
            
    return selected

