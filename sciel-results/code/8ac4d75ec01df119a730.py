"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    We use a greedy approach with backtracking/local search to find a large subset
    of integers from 2..40 where all subset products are distinct.
    
    Two numbers a and b create a collision if there exist subsets whose products
    are equal. A simpler sufficient condition: if the prime exponent vectors of the
    chosen numbers are linearly independent over the rationals, then all subset
    products are distinct. However, linear independence is sufficient but not
    necessary for distinct subset products. We can do better by directly checking
    for subset product collisions.
    
    For small sets, we can check all subset products directly. For larger sets,
    we use the linear independence check as a fast filter and then verify with
    actual subset product enumeration.
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Prime factorization vectors for numbers 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def factorize(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return vec
    
    # Precompute factorization vectors for all candidates
    candidates = list(range(2, 41))
    factor_vecs = {n: factorize(n) for n in candidates}
    
    def subset_products_distinct(nums):
        """Check if all 2^k subset products of nums are distinct."""
        if len(nums) <= 1:
            return True
        if len(nums) > 20:
            # Too many to enumerate efficiently, use linear independence as proxy
            return _linearly_independent([factor_vecs[n] for n in nums])
        
        products = set()
        products.add(1)  # empty product
        for n in nums:
            new_products = set()
            for p in products:
                new_products.add(p * n)
            if len(products) + len(new_products) > len(products | new_products):
                # There's a collision
                return False
            products = products | new_products
        
        # Check all products are distinct
        return len(products) == (1 << len(nums))
    
    def _linearly_independent(vecs):
        """Check if vectors are linearly independent over Q using fraction elimination."""
        from fractions import Fraction
        basis = []
        for v in vecs:
            vf = [Fraction(x) for x in v]
            for bv in basis:
                if vf[0] != 0 and bv[0] != 0:
                    factor = vf[0] / bv[0]
                    for i in range(len(vf)):
                        vf[i] -= factor * bv[i]
            # Remove leading zeros
            while vf and vf[0] == 0:
                vf.pop(0)
            if not vf:
                return False  # dependent
            basis.append(vf)
        return True
    
    def can_add(current_set, n):
        """Check if adding n to current_set maintains distinct subset products."""
        return subset_products_distinct(list(current_set) + [n])
    
    # Greedy construction: try to add numbers in various orders
    def greedy(order):
        selected = []
        for n in order:
            if can_add(set(selected), n):
                selected.append(n)
        return selected
    
    # Try multiple orderings and keep the best
    best = []
    
    # Order 1: ascending
    best = greedy(candidates)
    
    # Order 2: descending
    if time.time() < deadline:
        best = max(best, greedy(candidates[::-1]), key=len)
    
    # Order 3: primes first, then composites
    primes_first = [p for p in candidates if all(p % q != 0 for q in range(2, int(p**0.5)+1))]
    composites = [n for n in candidates if n not in primes_first]
    order3 = primes_first + composites
    if time.time() < deadline:
        best = max(best, greedy(order3), key=len)
    
    # Order 4: random shuffles
    for _ in range(20):
        if time.time() >= deadline:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy(shuffled)
        if len(result) > len(best):
            best = result
    
    # Local search: try swapping elements to increase size
    if time.time() < deadline:
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best)):
                for n in candidates:
                    if n in best_set:
                        continue
                    # Try replacing best[i] with n
                    new_set = set(best)
                    new_set.remove(best[i])
                    new_set.add(n)
                    if len(new_set) != len(best):
                        continue
                    if can_add(new_set, 0) or True:  # Just check the set
                        # Rebuild list and check
                        new_list = list(new_set)
                        if subset_products_distinct(new_list):
                            best = new_list
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break
    
    # Try to add more elements if possible
    if time.time() < deadline:
        best_set = set(best)
        for n in candidates:
            if n not in best_set and can_add(best_set, n):
                best.append(n)
                best_set.add(n)
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
