import time
from fractions import Fraction
from math import lcm
from functools import reduce
import random

def _verify(subset):
    """Verify all reciprocal subset sums are distinct using integer weights."""
    if not subset:
        return True
    l = reduce(lcm, subset)
    weights = [l // x for x in subset]
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False
            new_sums.add(ns)
        sums = new_sums
    return True

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 55.0

    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    elements = list(range(1, 23))

    def check_collision_fast(current_set, x):
        """Check collision using incremental LCM tracking."""
        if not current_set:
            return False
        all_elems = current_set + [x]
        l = reduce(lcm, all_elems)
        # Only need to check if w_x = l//x creates a collision
        # Build subset sums of existing weights
        sums = {0}
        for y in current_set:
            w = l // y
            new_sums = set()
            for s in sums:
                ns = s + w
                new_sums.add(ns)
            sums = new_sums
        w_x = l // x
        for s in sums:
            if (s + w_x) in sums:
                return True
        return False

    # Superincreasing greedy: pick largest elements first
    # Condition: when adding x (smaller than all placed), need 1/x > sum_{placed} 1/y
    # Equivalent to: x < 1 / sum(1/y for y in placed)
    
    def superincreasing_greedy():
        placed = []
        tail_sum = Fraction(0)
        available = set(elements)
        # Try to add elements from largest to smallest
        for x in sorted(elements, reverse=True):
            if x not in available:
                continue
            r = Fraction(1, x)
            # For superincreasing, we need 1/x > tail_sum when x is the smallest
            # But if we're building from large to small, x is the new smallest
            # Condition: 1/x > tail_sum
            if r > tail_sum:
                placed.append(x)
                tail_sum += r
                available.discard(x)
        return placed

    si = superincreasing_greedy()
    if len(si) > best_size and _verify(si):
        best_set = si
        best_size = len(si)

    # General greedy with multiple orderings
    def greedy(order, target_size):
        current = []
        for x in order:
            if len(current) == target_size:
                break
            if not check_collision_fast(current, x):
                current.append(x)
        return current

    # Try various orderings to find larger sets
    orderings = [
        sorted(elements, reverse=True),  # largest first
        sorted(elements),  # smallest first
        list(reversed(sorted(elements))),  # same as above
    ]
    
    # Add some structured orderings
    # Prime-heavy
    primes = [2,3,5,7,11,13,17,19]
    composites = [x for x in elements if x not in primes]
    orderings.append(primes + composites)
    orderings.append(composites + primes)
    
    # Try to find larger sets
    for k in range(22, best_size, -1):
        if time.time() > deadline:
            break
        found = False
        for order in orderings:
            if time.time() > deadline:
                break
            result = greedy(order, k)
            if len(result) == k and _verify(result):
                best_set = result
                best_size = k
                found = True
                break
        if found:
            break
        
        # Random shuffles
        rng = random.Random(k * 1000 + 7)
        for _ in range(50):
            if time.time() > deadline:
                break
            shuffled = elements[:]
            rng.shuffle(shuffled)
            result = greedy(shuffled, k)
            if len(result) == k and _verify(result):
                best_set = result
                best_size = k
                found = True
                break
        if found:
            break

    # Backtracking for next size
    while best_size < 22 and time.time() < deadline:
        target_k = best_size + 1
        if target_k > 15:  # Backtracking gets expensive
            break
            
        def bt(remaining, current, depth, max_d):
            if time.time() > deadline:
                return None
            if depth == max_d:
                return list(current)
            if len(remaining) < max_d - depth:
                return None
            for i in range(len(remaining)):
                x = remaining[i]
                if not check_collision_fast(current, x):
                    new_rem = remaining[:i] + remaining[i+1:]
                    res = bt(new_rem, current + [x], depth + 1, max_d)
                    if res is not None:
                        return res
            return None

        result = bt(elements, [], 0, target_k)
        if result is not None and _verify(result):
            best_set = result
            best_size = target_k
        else:
            break

    return best_set

def construct():
    return _construct_impl()

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END
