"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _is_bad_triple(a, b, c):
    """Check if three distinct elements form an equal-lcm triple."""
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _can_add(x, selected):
    """Check if x can be added to selected without creating a bad triple."""
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if _is_bad_triple(x, selected[i], selected[j]):
                return False
    return True

def _greedy_construct(ordering, deadline):
    """Greedy construction following given ordering."""
    selected = []
    for x in ordering:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to add elements or perform 1-swaps to improve the set."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding elements
        for x in range(1, 129):
            if x not in selected and _can_add(x, selected):
                selected.append(x)
                improved = True
                break
        # Try 1-swaps: remove one element, try to add two
        if not improved:
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected
                removed = selected[i]
                temp = selected[:i] + selected[i+1:]
                # Try to add any element
                added_any = False
                for x in range(1, 129):
                    if x != removed and x not in temp and _can_add(x, temp):
                        # Check if we can add a second element too
                        temp2 = temp + [x]
                        for y in range(1, 129):
                            if y != removed and y != x and y not in temp2 and _can_add(y, temp2):
                                selected = temp2 + [y]
                                improved = True
                                added_any = True
                                break
                        if added_any:
                            break
                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Multiple strategies with different orderings
    best_set = []
    
    # Strategy 1: Natural order
    ordering1 = list(range(1, 129))
    s1 = _greedy_construct(ordering1, deadline)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Reverse order
    ordering2 = list(range(128, 0, -1))
    s2 = _greedy_construct(ordering2, deadline)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: Random orderings with local search
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        ordering = list(range(1, 129))
        random.shuffle(ordering)
        s = _greedy_construct(ordering, deadline)
        if time.time() > deadline:
            break
        s = _local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 4: Prefer numbers with fewer conflicts
    # Numbers that are prime or have few divisors may be safer
    def conflict_score(x):
        """Estimate how many conflicts x might cause."""
        # Count how many pairs (a,b) in [1,128] would form bad triple with x
        count = 0
        for a in range(1, 129):
            if a == x:
                continue
            for b in range(a+1, 129):
                if b == x:
                    continue
                if _is_bad_triple(x, a, b):
                    count += 1
        return count
    
    scored = [(conflict_score(x), x) for x in range(1, 129)]
    scored.sort(key=lambda t: (t[0], t[1]))
    ordering4 = [x for _, x in scored]
    s4 = _greedy_construct(ordering4, deadline)
    if len(s4) > len(best_set):
        best_set = s4
    if time.time() < deadline:
        s4 = _local_search(s4, deadline)
        if len(s4) > len(best_set):
            best_set = s4
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
