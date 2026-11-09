"""Rewritten program for large integer sets with no two-other-product divisibility."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Candidate set: all integers from 2 to 256
    all_candidates = list(range(2, 257))
    n_total = len(all_candidates)
    
    # Precompute prime factorization for each number
    def prime_factors(n):
        """Return dict of prime -> exponent for n."""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    # Precompute factorizations
    factorizations = {}
    for x in all_candidates:
        factorizations[x] = prime_factors(x)
    
    def divides(a, b):
        """Check if a divides b."""
        return b % a == 0
    
    def can_add(candidate, current_set):
        """Check if adding candidate to current_set maintains validity."""
        current_list = list(current_set)
        n = len(current_list)
        
        # Check 1: candidate ∤ b*c for all distinct b,c in current_set
        for i in range(n):
            for j in range(i + 1, n):
                b = current_list[i]
                c = current_list[j]
                if (b * c) % candidate == 0:
                    return False
        
        # Check 2: for each a in current_set, for each x in current_set where x != a:
        #   a ∤ (candidate * x)
        for i in range(n):
            a = current_list[i]
            for j in range(n):
                if j == i:
                    continue
                x = current_list[j]
                if (candidate * x) % a == 0:
                    return False
        
        return True
    
    def is_valid_set(s):
        """Check if set s satisfies the condition."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Greedy approach with multiple orderings
    best_set = set()
    best_size = 0
    
    # Try different orderings
    orderings = [
        sorted(all_candidates, reverse=True),  # descending
        sorted(all_candidates),  # ascending
        list(all_candidates),  # natural order
    ]
    
    # Add some random orderings
    random.seed(42)
    for _ in range(20):
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        current = set()
        for c in ordering:
            if can_add(c, current):
                current.add(c)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Local search: try to add elements
    not_in_best = [x for x in all_candidates if x not in best_set]
    improved = True
    while improved:
        improved = False
        for c in not_in_best:
            if time.time() > deadline - 5:
                break
            if can_add(c, best_set):
                best_set.add(c)
                not_in_best.remove(c)
                improved = True
                best_size = len(best_set)
        if time.time() > deadline - 5:
            break
    
    # Local search: try removing one element and adding two others
    if time.time() < deadline - 10:
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 5:
                break
            elem_to_remove = best_list[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            candidates_to_add = [x for x in all_candidates if x not in test_set]
            added = []
            for c in candidates_to_add:
                if time.time() > deadline - 5:
                    break
                if len(added) >= 2:
                    break
                if can_add(c, test_set | set(added)):
                    added.append(c)
            if len(added) >= 2:
                new_set = test_set | set(added)
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)
    
    # Final verification
    result = sorted(best_set)
    if not is_valid_set(result):
        # Fallback: try to build a valid set from scratch
        current = set()
        for c in all_candidates:
            if can_add(c, current):
                current.add(c)
        result = sorted(current)
    
    # Ensure we return a list of Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
