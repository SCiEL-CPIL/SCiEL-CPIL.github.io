# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_valid(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        # Check triples that include a? No - a is the divisor, not in the triple
        # Check: for each triple of 3 distinct OTHER elements, a does not divide their sum
        others = [selected[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_with_order(order, deadline):
    """Greedy: try to add elements in given order, check validity incrementally."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        # Check if adding x keeps the set valid
        candidate = selected + [x]
        # Fast check: only need to verify new constraints involving x
        # 1) x must not divide sum of any triple from selected
        if len(selected) >= 3:
            valid = True
            for t in combinations(selected, 3):
                if sum(t) % x == 0:
                    valid = False
                    break
            if not valid:
                continue
        # 2) For each existing a in selected, x+a+b+c must not be divisible by a
        #    where b,c are from selected, b!=c, b!=a, c!=a
        for a in selected:
            others = [v for v in selected if v != a]
            # Need to check triples (x, b, c) where b,c in others, b!=c
            for b, c in combinations(others, 2):
                if (x + b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    return selected

def construct():
    start = time.time()
    deadline = start + 85.0  # 85s to leave margin before 90s kill
    
    # Pre-compute all numbers 1..192
    all_nums = list(range(1, 193))
    
    best_set = []
    
    # Strategy 1: Largest first (original)
    order1 = sorted(all_nums, reverse=True)
    result1 = _greedy_with_order(order1, deadline)
    if len(result1) > len(best_set):
        best_set = result1
    
    # Strategy 2: Smallest first
    if time.time() < deadline - 5:
        order2 = sorted(all_nums)
        result2 = _greedy_with_order(order2, deadline)
        if len(result2) > len(best_set):
            best_set = result2
    
    # Strategy 3: Try some random shuffles
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 2:
        attempt += 1
        order3 = all_nums[:]
        rng.shuffle(order3)
        result3 = _greedy_with_order(order3, deadline)
        if len(result3) > len(best_set):
            best_set = result3
        # If we've found 48, we're done
        if len(best_set) >= 48:
            break
    
    return best_set
# EVOLVE-BLOCK-END
