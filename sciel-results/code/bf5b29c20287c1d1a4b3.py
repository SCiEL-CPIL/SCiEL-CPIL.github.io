import time
import random
import math
from itertools import combinations

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes k.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Candidate pool: 2..40
    candidates = list(range(2, 41))

    # Helper: check if adding a number to a set keeps all subset products distinct
    def _try_add(s, new_elem):
        """Check if adding new_elem to set s keeps all subset products distinct."""
        # Get existing subset products
        existing_products = set()
        for r in range(len(s) + 1):
            for combo in combinations(s, r):
                prod = 1
                for x in combo:
                    prod *= x
                existing_products.add(prod)
        
        # Check if any existing product * new_elem equals another existing product
        # Or if new_elem equals an existing product
        # More precisely: for each subset T of s, product(T) * new_elem must not equal product(U) for any subset U of s
        # This is equivalent to: for each subset T of s, product(T) * new_elem not in existing_products
        # But we also need product(T) * new_elem to be distinct from each other (which they are since new_elem is fixed and products(T) are distinct)
        
        for prod in list(existing_products):
            if prod * new_elem in existing_products:
                return False
        # Also check if new_elem itself creates a conflict: new_elem * 1 = new_elem, so new_elem must not be in existing_products
        # Wait, the empty product is 1. So product(empty) * new_elem = new_elem. 
        # So we need new_elem not in existing_products.
        # But actually, the condition is: all 2^(k+1) products must be distinct.
        # The new products are {p * new_elem : p in existing_products}.
        # These must be distinct from each other (true, since new_elem != 0 and existing products are distinct)
        # And must be distinct from the old products.
        # So we need: no p in existing_products such that p * new_elem is in existing_products.
        # AND no p in existing_products such that p * new_elem == p' for p' in existing_products.
        # Which is the same condition.
        # But wait, does this cover the case where new_elem * 1 = new_elem, and new_elem is already an existing product?
        # Yes, if new_elem is in existing_products, then for p=1 (empty product), p * new_elem = new_elem which is in existing_products.
        # So the condition covers it.
        
        return True

    # Helper: verify that all subset products are distinct
    def _subset_products_distinct(s):
        if not s:
            return True
        products = set()
        for r in range(len(s) + 1):
            for combo in combinations(s, r):
                prod = 1
                for x in combo:
                    prod *= x
                if prod in products:
                    return False
                products.add(prod)
        return True

    # Helper: local search to try to add more elements
    def _local_search(current_set, deadline):
        """Try to add more elements via local search."""
        current_set = current_set[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each candidate that isn't already in the set
            for c in candidates:
                if c in current_set:
                    continue
                if _try_add(current_set, c):
                    current_set.append(c)
                    improved = True
                    break
        return current_set

    # Helper: swap-based local search to potentially reach a larger set
    def _swap_local_search(current_set, deadline):
        """Try to improve the set by swapping elements."""
        current_set = current_set[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_set)):
                for c in candidates:
                    if c in current_set:
                        continue
                    # Try replacing current_set[i] with c
                    new_set = current_set[:i] + current_set[i+1:] + [c]
                    if _subset_products_distinct(new_set):
                        # Check if this is an improvement (same size, but maybe enables further additions)
                        # For now, just accept if distinct
                        current_set = new_set
                        improved = True
                        break
                if improved:
                    break
        return current_set

    best_set = []
    best_size = 0

    # Strategy: Randomized greedy + local search with multiple restarts
    for restart in range(200):
        if time.time() > deadline:
            break

        random.shuffle(candidates)
        current_set = []

        for c in candidates:
            if time.time() > deadline:
                break
            if _try_add(current_set, c):
                current_set.append(c)

        # Local search to add more
        refined = _local_search(current_set, deadline)
        if time.time() > deadline:
            break

        # Swap local search
        further_refined = _swap_local_search(refined, deadline)
        if time.time() > deadline:
            break

        if len(further_refined) > best_size:
            best_size = len(further_refined)
            best_set = further_refined[:]

    # Final verification
    if best_set and _subset_products_distinct(best_set):
        return [int(x) for x in best_set]

    # Fallback to known valid set of size 16
    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# EVOLVE-BLOCK-END
