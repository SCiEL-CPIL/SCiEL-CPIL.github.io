"""Rewritten program for dense increasing sequences with unique block products."""
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    """
    Return strictly increasing integers in [2,128] such that products of every
    nonempty consecutive block are distinct. Maximize length.
    
    Strategy: Greedy with backtracking-lite using prime exponent vectors.
    We represent each number by its prime exponent vector (for primes up to 128).
    The product of a consecutive block is the sum of exponent vectors of elements in that block.
    We maintain a set of all block-product exponent vectors seen so far.
    
    To maximize length, we try to include as many numbers as possible.
    A key insight: the greedy approach in the seed program is already quite good,
    but we can improve by using a smarter ordering or by trying to swap out
    problematic elements.
    
    However, given the 60s time limit and the need for a deterministic, fast solution,
    let's implement a more sophisticated greedy with lookahead:
    - Precompute prime exponent vectors for all numbers 2..128.
    - Use a greedy approach but when a conflict is detected, try skipping the current
      element and see if we can recover length later.
    
    Actually, let's think about this differently. The seed program's greedy is:
    For each x from 2 to 128, check if adding x creates any duplicate block product.
    If not, add it. This is O(n^2 * p) where p is number of primes, which is fast.
    
    The question is: can we do better than greedy? 
    
    Key observation: The constraint is that ALL consecutive block products must be distinct.
    This includes singletons, so all elements must be distinct (already satisfied by strictly increasing).
    It also means that no two different consecutive blocks can have the same product.
    
    Since the sequence is strictly increasing and all elements are >= 2, the product of any
    block is at least 2^(length). For blocks of the same length, products are likely distinct
    due to the increasing nature. The main collisions would be between blocks of different lengths
    or between blocks that happen to have the same product.
    
    Let me implement a greedy with a twist: instead of just going left to right, I'll use
    the prime factorization to quickly check for collisions, and I'll try a few different
    starting strategies to see if any yield a longer sequence.
    
    Actually, let's just optimize the greedy to be as fast as possible and see if we can
    find any cases where skipping an early element allows more elements later.
    """
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    def factor_vector(n):
        """Return tuple of prime exponents for n."""
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
                if n == 1:
                    break
        return tuple(vec)
    
    # Precompute factor vectors for all numbers 2..128
    factor_vecs = {}
    for n in range(2, 129):
        factor_vecs[n] = factor_vector(n)
    
    def add_to_product_vec(vec, factor):
        """Add factor vector to vec, return new tuple."""
        return tuple(vec[i] + factor[i] for i in range(num_primes))
    
    def try_greedy(skip_set=frozenset()):
        """
        Greedy: iterate 2..128, skip if in skip_set, check for collisions.
        Returns the list of selected values.
        """
        values = []
        products = set()
        
        for x in range(2, 129):
            if x in skip_set:
                continue
            
            # Compute all block products ending at x
            # tails[i] = product of values[len(values)-1-i : len(values)] * x
            # More precisely: if values = [v0, v1, ..., v_{k-1}], then:
            #   new block products ending at x:
            #     x
            #     v_{k-1} * x
            #     v_{k-2} * v_{k-1} * x
            #     ...
            #     v_0 * v_1 * ... * v_{k-1} * x
            # We can compute these incrementally.
            
            k = len(values)
            if k == 0:
                new_products = {factor_vecs[x]}
            else:
                new_products = set()
                prod_vec = factor_vecs[x]
                new_products.add(prod_vec)
                for i in range(k - 1, -1, -1):
                    prod_vec = add_to_product_vec(prod_vec, factor_vecs[values[i]])
                    new_products.add(prod_vec)
            
            # Check for collisions
            if products.intersection(new_products):
                continue
            
            values.append(x)
            products.update(new_products)
        
        return values
    
    # First, run the basic greedy
    best = try_greedy()
    best_len = len(best)
    
    # Try some simple modifications: skip one element and see if we get longer
    # This is a heuristic search
    for skip in range(2, 129):
        if skip in best:
            continue
        # Try skipping 'skip' and see what happens
        candidate = try_greedy(frozenset([skip]))
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
    
    return best
# EVOLVE-BLOCK-END
