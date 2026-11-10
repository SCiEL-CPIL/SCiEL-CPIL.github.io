import time
import itertools
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # safety margin before 60s
    
    # Helper: check if a set of integers has all distinct reciprocal subset sums using exact arithmetic
    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated (all subset sums distinct)."""
        n = len(elements)
        if n == 0:
            return True
        # Use Fraction for exact arithmetic
        reciprocals = [Fraction(1, e) for e in elements]
        seen = {Fraction(0, 1)}
        # Build subset sums iteratively
        sums = [Fraction(0, 1)]
        for r in reciprocals:
            new_sums = []
            for s in sums:
                ns = s + r
                if ns in seen:
                    return False
                new_sums.append(ns)
                seen.add(ns)
            sums.extend(new_sums)
        return True

    # Faster check: maintain the set of sums incrementally using floats
    def check_addable_float(current_sums, seen, element):
        """Check if adding 1/element to the current set maintains dissociation using floats."""
        r = 1.0 / element
        new_sums = []
        for s in current_sums:
            ns = s + r
            # Use a tolerance? No, if we use exact float equality, it might be too strict or too loose?
            # Since min diff is ~1e-9 and precision is ~1e-15, exact float equality is safe for distinctness.
            # However, floating point addition is not associative.
            # s + r might produce slightly different results depending on order.
            # But we are just checking if ns is in seen.
            # If ns is very close to an existing sum, it might be a false positive collision.
            # Given the gap is 1e-9, and float error is ~1e-16, we are safe.
            if ns in seen:
                return None
            new_sums.append(ns)
        return new_sums

    all_elements = list(range(1, 23))
    best_set = []
    best_sums = None
    best_seen = None
    
    # Multiple greedy strategies with different orderings
    # We will generate many random permutations
    orders = [
        all_elements[::-1],  # largest first
        all_elements,        # smallest first
    ]
    
    # Add random permutations
    for _ in range(50):
        random_order = all_elements[:]
        random.shuffle(random_order)
        orders.append(random_order)
        
    # Also add some heuristic orders
    # Pairs that sum to 23?
    # Alternating large/small
    alt = []
    for i in range(11):
        alt.append(22 - i)
        alt.append(i + 1)
    orders.append(alt)

    for order in orders:
        if time.time() > deadline:
            break
        current = []
        current_sums = [0.0]
        current_seen = {0.0}
        
        for e in order:
            if time.time() > deadline:
                break
            new_sums = check_addable_float(current_sums, current_seen, e)
            if new_sums is not None:
                current.append(e)
                current_sums.extend(new_sums)
                current_seen.update(new_sums)
        
        if len(current) > len(best_set):
            best_set = current[:]
            best_sums = current_sums[:]
            best_seen = set(current_seen)
            
            # Early termination if we reach 22 (impossible) or high number
            if len(best_set) >= 20:
                 # Verify with Fraction immediately? 
                 # If it's valid, we can try to find 21, 22
                 # But let's finish the loop to see if we can get 21/22
                 pass

    # Local search: try to add remaining elements
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable_float(best_sums, best_seen, e)
                if new_sums is not None:
                    best_set.append(e)
                    best_sums.extend(new_sums)
                    best_seen.update(new_sums)
                    remaining.remove(e)
                    improved = True
    
    # Local search: try swaps to potentially enable adding more elements
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        # Try removing one element and adding two others
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            # Rebuild without removed
            test_set = best_set[:i] + best_set[i+1:]
            test_sums = [0.0]
            test_seen = {0.0}
            for e in test_set:
                r = 1.0 / e
                new_s = []
                valid = True
                for s in test_sums:
                    ns = s + r
                    if ns in test_seen:
                        valid = False
                        break
                    new_s.append(ns)
                if not valid:
                    break
                test_sums.extend(new_s)
                test_seen.update(new_s)
            
            if len(test_sums) != 2 ** len(test_set):
                continue  # not dissociated (shouldn't happen)
            
            # Try to add two elements from remaining
            added = 0
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable_float(test_sums, test_seen, e)
                if new_sums is not None:
                    test_set.append(e)
                    test_sums.extend(new_sums)
                    test_seen.update(new_sums)
                    added += 1
                    if added >= 2:
                        break
            
            if added >= 2:
                # This swap improves size
                best_set = test_set
                best_sums = test_sums
                best_seen = test_seen
                remaining = [e for e in all_elements if e not in best_set]
                
    # Final verification with exact arithmetic
    if best_set:
        if not is_dissociated(best_set):
            # If the float check gave a false positive, we need to fall back
            # Try to repair or use a known valid set
            # Let's try to remove elements until it's valid?
            # Or just use the seed.
            # Better: try to find a valid subset of the found set?
            # Or just return the seed.
            # Given the gap is large, false positives are unlikely.
            best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    else:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
