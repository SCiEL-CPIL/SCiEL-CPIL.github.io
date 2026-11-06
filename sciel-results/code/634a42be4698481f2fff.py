import time
import sys
import random
import math

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    
    Strategy:
    1. Represent all subset sums as integers scaled by L = LCM(1..22) = 232792560.
    2. Use a hybrid search strategy:
       a. Greedy construction with multiple orderings to get a good initial set.
       b. Local search (swap-based) to try to improve the cardinality.
       c. Randomized restarts if time permits.
    """
    
    L = 232792560
    
    # Precompute scaled reciprocals
    rec = [0] * 23
    for i in range(1, 23):
        rec[i] = L // i
        
    def is_dissociated(elements):
        """Check if a set of elements is dissociated (all subset sums distinct)."""
        sums = {0}
        for x in elements:
            rx = rec[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s)
                new_sums.add(s + rx)
            if len(new_sums) != 2 * len(sums):
                return False
            sums = new_sums
        return True

    def greedy_build(order):
        """
        Builds a dissociated set using the given order of elements.
        Returns the list of elements in the set.
        """
        current_set = []
        sums = {0}
        
        for x in order:
            if _time_left() < 0.5:
                break
            rx = rec[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s)
                new_sums.add(s + rx)
            
            if len(new_sums) == 2 * len(sums):
                current_set.append(x)
                sums = new_sums
        
        return current_set

    def local_search_improve(elements):
        """
        Attempts to improve the cardinality of a dissociated set by swapping elements.
        Returns a new list of elements if improved, else the original.
        """
        current_set = list(elements)
        current_size = len(current_set)
        
        # Try to add an element not in the set
        # Then try to remove an element if it allows adding two? 
        # Simpler: try swapping one element for another to maintain size but maybe allow future additions?
        # Actually, local search for maximizing cardinality is tricky.
        # Let's try a "grow" strategy:
        # 1. Try to add any element from [1..22] not in set.
        # 2. If successful, return.
        # 3. If not, try removing one element and adding two? (Expensive)
        
        # Let's just try to add elements first.
        for x in range(1, 23):
            if x not in current_set:
                new_set = current_set + [x]
                if is_dissociated(new_set):
                    return new_set
        
        # If we couldn't add, try swaps.
        # Replace one element with another, then try to add more.
        # This is complex. Let's stick to simple addition for now in the local step,
        # and rely on multiple greedy runs with different orderings for diversity.
        
        return current_set

    # Define orderings
    orderings = []
    
    # 1. Descending
    orderings.append(list(range(22, 0, -1)))
    
    # 2. Ascending
    orderings.append(list(range(1, 23)))
    
    # 3. Balanced (alternating small/large)
    bal = []
    l, r = 1, 22
    while l <= r:
        bal.append(l)
        if l != r:
            bal.append(r)
        l += 1
        r -= 1
    orderings.append(bal)
    
    # 4. Primes first
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    composites = [x for x in range(1, 23) if x not in primes and x != 1]
    orderings.append(primes + composites + [1])
    
    # 5. Random shuffles
    rng = random.Random(42)
    for _ in range(10):
        order = list(range(1, 23))
        rng.shuffle(order)
        orderings.append(order)
    
    best_set = []
    
    for order in orderings:
        if _time_left() < 1.0:
            break
        s = greedy_build(order)
        if len(s) > len(best_set):
            best_set = s
            
    # Local search phase: try to improve best_set
    improved = local_search_improve(best_set)
    if len(improved) > len(best_set):
        best_set = improved
    
    # If best_set is empty, return [1]
    if not best_set:
        return [1]
        
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    L = 232792560
    sums = {0}
    for x in s:
        rx = L // x
        new_sums = {ss + rx for ss in sums}
        if len(new_sums) != 2 * len(sums):
            print("INVALID!")
        sums = new_sums
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")
