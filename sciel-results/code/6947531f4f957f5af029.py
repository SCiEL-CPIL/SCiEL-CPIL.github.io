"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # All candidate numbers in 2..40
    candidates = list(range(2, 41))
    
    # We want to find the largest subset with distinct subset products
    # Use backtracking with heuristic ordering
    
    # Precompute: for a set S, check if all 2^|S| subset products are distinct
    def has_distinct_subset_products(S):
        """Check if all subset products of S are distinct."""
        n = len(S)
        if n > 20:  # Too many subsets to enumerate
            return False
        products = set()
        # Enumerate all subsets via bit masks
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= S[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    # Heuristic: order candidates to prefer those with larger "unique" prime content
    # Try to find a large set via greedy + local search
    
    # First, try a greedy approach: add numbers one by one if they preserve the property
    def greedy_build(ordered_candidates):
        S = []
        for c in ordered_candidates:
            if time.time() > deadline:
                break
            new_S = S + [c]
            if has_distinct_subset_products(new_S):
                S = new_S
        return S
    
    # Try multiple orderings
    best_set = []
    
    # Ordering 1: primes first, then composites
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5)+1):
            if n % d == 0:
                return False
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Try: primes first, then composites sorted by value
    order1 = primes + sorted(composites)
    S1 = greedy_build(order1)
    if len(S1) > len(best_set):
        best_set = S1
    
    # Try: all sorted ascending
    order2 = sorted(candidates)
    S2 = greedy_build(order2)
    if len(S2) > len(best_set):
        best_set = S2
    
    # Try: composites first (some composites like 4,9,25 are "safe")
    order3 = sorted(composites) + primes
    S3 = greedy_build(order3)
    if len(S3) > len(best_set):
        best_set = S3
    
    # Try: descending
    order4 = sorted(candidates, reverse=True)
    S4 = greedy_build(order4)
    if len(S4) > len(best_set):
        best_set = S4
    
    # Local search: try to swap elements to improve
    def local_search(S):
        S = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(S)):
                for j in range(2, 41):
                    if j in S:
                        continue
                    new_S = S[:i] + S[i+1:] + [j]
                    new_S.sort()
                    if has_distinct_subset_products(new_S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
        return S
    
    best_set = local_search(best_set)
    
    # Try to add more elements
    for c in range(2, 41):
        if time.time() > deadline:
            break
        if c not in best_set:
            new_S = best_set + [c]
            if has_distinct_subset_products(new_S):
                best_set = new_S
    
    # Ensure we return a list of Python ints
    return list(best_set)
# EVOLVE-BLOCK-END

