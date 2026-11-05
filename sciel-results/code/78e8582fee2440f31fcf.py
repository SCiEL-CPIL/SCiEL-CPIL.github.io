"""
Strategy: Greedy construction of a large family of 4-subsets of [1,16] with no
three-petal sunflowers (no three distinct sets A,B,C with A∩B = A∩C = B∩C).

Uses bitmask representation for efficiency. Multiple greedy passes with different
orderings, followed by local improvement (swap) passes to refine the solution.

Key optimization: For a new set X, a conflict with existing family F exists iff
there are two sets A, B in F such that A∩B = A∩X = B∩X. Let S = A∩X. Then we
need B to contain S and have B∩X = S. Since |X|=4, S is a subset of X.

We can precompute for each subset S of X (16 subsets), which existing family
members have intersection exactly S with X. If any two such members also have
intersection S with each other, we have a conflict.

Actually, the check (a&b) == (a&x) == (b&x) is fast with bitmasks. The main
bottleneck is the O(m^2) loop. With m up to ~250, that's 31K pairs per candidate,
times 1820 candidates = ~57M checks. Each check is 3 ANDs and 2 comparisons.
Should run in ~30-50s in Python. We use multiple orderings and a local polish.
"""

import time
import random
from itertools import combinations

def _check_conflict(family, x):
    """Check if adding x to family creates a sunflower triple.
    A sunflower triple A,B,C has A∩B = A∩C = B∩C.
    For new set x, check all pairs (A,B) in family:
    if (A&B) == (A&x) == (B&x), then conflict.
    """
    m = len(family)
    for i in range(m):
        a = family[i]
        ax = a & x
        for j in range(i + 1, m):
            b = family[j]
            if (a & b) == ax and (b & x) == ax:
                return True
    return False

def _greedy_pass(all_sets, order, deadline):
    """Run one greedy pass with given order."""
    family = []
    for idx in order:
        if time.time() > deadline:
            break
        x = all_sets[idx]
        if not _check_conflict(family, x):
            family.append(x)
    return family

def _local_improve(family, all_sets, n_total, deadline, max_passes=3):
    """Try to improve family by removing a set and adding a better one.
    Returns improved family."""
    improved = list(family)
    improved_set = set(improved)
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved_any = False
        for i in range(len(improved)):
            if time.time() > deadline:
                break
            removed = improved[i]
            temp_family = improved[:i] + improved[i+1:]
            
            # Try to find a replacement from unselected sets
            best_candidate = None
            for j in range(n_total):
                if time.time() > deadline:
                    break
                candidate = all_sets[j]
                if candidate in improved_set:
                    continue
                if not _check_conflict(temp_family, candidate):
                    # This candidate works; pick the first one (or could score)
                    best_candidate = candidate
                    break
            
            if best_candidate is not None:
                improved[i] = best_candidate
                improved_set.discard(removed)
                improved_set.add(best_candidate)
                improved_any = True
                # Restart scan after a change
                break
        
        if not improved_any:
            break
    
    return improved

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)  # 1820
    
    best_family = []
    
    # Multiple greedy passes with different orderings
    orders = []
    
    # Order 1: natural
    orders.append(list(range(n_total)))
    
    # Order 2: reverse
    orders.append(list(range(n_total - 1, -1, -1)))
    
    # Orders 3-6: different random shuffles
    for seed in [42, 123, 777, 2024]:
        rng = random.Random(seed)
        shuffled = list(range(n_total))
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline - 10.0:
            break
        family = _greedy_pass(all_sets, order, deadline - 10.0)
        if len(family) > len(best_family):
            best_family = family
    
    # Local improvement on the best family
    if time.time() < deadline - 5.0 and len(best_family) > 0:
        best_family = _local_improve(best_family, all_sets, n_total, deadline - 5.0)
    
    # One more greedy pass starting from best_family (try to add more)
    if time.time() < deadline - 3.0 and len(best_family) > 0:
        existing_set = set(best_family)
        extended = list(best_family)
        for idx in range(n_total):
            if time.time() > deadline - 3.0:
                break
            x = all_sets[idx]
            if x in existing_set:
                continue
            if not _check_conflict(extended, x):
                extended.append(x)
                existing_set.add(x)
        if len(extended) > len(best_family):
            best_family = extended
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
