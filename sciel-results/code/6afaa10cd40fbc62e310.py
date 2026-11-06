import time
import random
import math

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _is_dissociated(S):
    """Check if all subset sums of {1/s for s in S} are distinct."""
    if len(S) <= 1:
        return True
    # Compute LCM of elements in S
    L = 1
    for s in S:
        L = _lcm(L, s)
        if L > 10**18:  # Cap to avoid huge numbers; use modular check instead
            # Fall back to modular check with a large prime
            p = 10**18 + 9  # A large prime
            seen = {0}
            sums = [0]
            for s in S:
                w = (L % p) * pow(s, -1, p) % p  # This is wrong if L is capped
                # Actually, let's just use Fraction for correctness
                pass
            # Use Fraction-based check for correctness
            from fractions import Fraction
            seen = {Fraction(0)}
            sums = [Fraction(0)]
            for s in S:
                w = Fraction(1, s)
                new_sums = [x + w for x in sums]
                for ns in new_sums:
                    if ns in seen:
                        return False
                seen.update(new_sums)
                sums.extend(new_sums)
            return True
    
    # Use exact integer arithmetic with LCM
    weights = [L // s for s in S]
    seen = {0}
    sums = [0]
    for w in weights:
        new_sums = [x + w for x in sums]
        for ns in new_sums:
            if ns in seen:
                return False
        seen.update(new_sums)
        sums.extend(new_sums)
    return True

def construct():
    start = time.time()
    deadline = start + 55.0
    
    best = []
    best_size = 0
    
    def try_greedy(order):
        current = set()
        for e in order:
            if e in current:
                continue
            new = current | {e}
            if _is_dissociated(new):
                current = new
        return current
    
    # Try various orderings
    all_elements = list(range(1, 23))
    
    # Deterministic orders
    orders = [
        list(range(22, 0, -1)),  # Descending (larger first)
        list(range(1, 23)),       # Ascending
    ]
    
    # Random orders
    rng = random.Random(42)
    for _ in range(200):
        order = all_elements[:]
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        result = try_greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best = list(result)
    
    # Local search: try to add any remaining element
    if best:
        current = set(best)
        for e in range(1, 23):
            if e not in current:
                new = current | {e}
                if _is_dissociated(new):
                    current = new
                    best_size = len(current)
                    best = list(current)
    
    # Try swaps: remove one, add one, to unlock further additions
    if best and time.time() < deadline - 5.0:
        current = set(best)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for remove in list(current):
                for add in range(1, 23):
                    if add in current:
                        continue
                    new = (current - {remove}) | {add}
                    if _is_dissociated(new):
                        # Check if this allows further additions
                        extended = set(new)
                        for e in range(1, 23):
                            if e not in extended:
                                ext2 = extended | {e}
                                if _is_dissociated(ext2):
                                    extended = ext2
                        if len(extended) > len(current):
                            current = extended
                            best_size = len(current)
                            best = list(current)
                            improved = True
                            break
                if improved:
                    break
    
    if not best:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best
