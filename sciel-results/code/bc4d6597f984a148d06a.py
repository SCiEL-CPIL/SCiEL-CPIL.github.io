"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_pair_sums(selected, x):
    """Check if x divides the sum of any 3 distinct other selected elements."""
    if len(selected) < 3:
        return True
    # Build frequency map of pair sums from selected
    pair_sums = {}
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i + 1, n):
            b = selected[j]
            s = a + b
            pair_sums[s] = pair_sums.get(s, 0) + 1
    # For each element c in selected, check if there exists pair sum p
    # such that p + c is a positive multiple of x
    # p + c = m*x for some integer m >= 1
    for c in selected:
        # p must be in pair_sums and p != (sum of pairs involving c is not directly excluded,
        # but we need 3 DISTINCT elements, so pair (a,b) must not include c)
        # Since pair_sums counts all pairs, and c is one element,
        # we need pair (a,b) where a,b != c and a != b
        # The pair_sums includes pairs that might involve c.
        # Let's be careful: we need 3 distinct elements a,b,c where none equals x (x not in selected yet)
        # and a,b,c are all in selected.
        # So for a fixed c in selected, we need a,b in selected, a!=b, a!=c, b!=c,
        # such that a+b+c is divisible by x.
        # a+b = m*x - c for some m.
        # We need a+b to be a pair sum where neither a nor b is c.
        # Let's compute pair sums excluding c.
        pass
    # Alternative: iterate over all triples? Too slow for large selected.
    # Better: for each c, compute needed pair sum range.
    # Max sum of 3 elements <= 3*192 = 576. Min sum >= 1+2+3=6.
    # So m*x is in [6, 576], meaning m is in [ceil(6/x), floor(576/x)].
    for c in selected:
        # Find all m such that m*x - c could be a valid pair sum
        # Pair sum min: 1+2=3 (but elements are distinct and positive, min pair sum with distinct is 1+2=3)
        # Actually min pair sum from selected depends on selected values.
        # Let's just iterate m from 1 to floor(576/x)
        max_m = 576 // x
        for m in range(1, max_m + 1):
            target = m * x - c
            if target <= 0:
                continue
            # Check if there exists a pair (a,b) in selected with a+b=target, a!=b, a!=c, b!=c
            # We need to check pair_sums[target] but exclude pairs involving c.
            # Count of pairs summing to target that don't involve c:
            # total pairs with sum target: pair_sums.get(target, 0)
            # pairs involving c with sum target: for each a in selected, a != c, if c+a=target and a in selected and a!=c, count 1
            # Actually: pairs involving c: (c, a) where a in selected, a != c, and c+a = target => a = target - c
            # So if target - c is in selected and target - c != c, then there's 1 pair involving c.
            # But we need to be careful: pair (a, c) where a = target - c.
            # Number of such pairs: 1 if (target - c) is in selected_set and (target - c) != c, else 0.
            # But also need a != c, which is target - c != c => target != 2c.
            total_pairs = pair_sums.get(target, 0)
            # Subtract pairs involving c
            a_val = target - c
            if a_val in selected_set and a_val != c:
                total_pairs -= 1
            # Also, the pair (a_val, c) is the same as (c, a_val), counted once in pair_sums.
            # If a_val == c, then pair would be (c,c) which isn't a valid pair of distinct elements,
            # so it wouldn't be in pair_sums anyway.
            if total_pairs > 0:
                return False
    return True

def _greedy_order(order, deadline):
    """Greedy construction using a given order of candidates."""
    selected = []
    selected_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        # Check if adding x is valid
        if _check_pair_sums(selected, x):
            selected.append(x)
            selected_set.add(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Strategy: Try multiple orders and keep the best result
    best_selected = []
    
    # Order 1: Descending (original approach)
    order1 = list(range(192, 0, -1))
    result1 = _greedy_order(order1, deadline)
    best_selected = result1
    
    if time.time() > deadline:
        return best_selected
    
    # Order 2: Ascending
    order2 = list(range(1, 193))
    result2 = _greedy_order(order2, deadline)
    if len(result2) > len(best_selected):
        best_selected = result2
    
    if time.time() > deadline:
        return best_selected
    
    # Order 3: Random shuffles
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy_order(order, deadline)
        if len(result) > len(best_selected):
            best_selected = result
    
    return best_selected
# EVOLVE-BLOCK-END
