import time
import random
from fractions import Fraction
from functools import reduce
from math import gcd

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // gcd(a, b)

LCM_ALL = 1
for i in range(1, 25):
    LCM_ALL = _lcm(LCM_ALL, i)

# Cache for subset sum checks to speed up local search
# Key: (target_int, tuple(sorted(term_ints)))
# Value: bool
_subset_sum_cache = {}

def _can_form_subset_sum(target_int, terms_int):
    """
    Check if any subset of terms_int sums to target_int.
    terms_int is a list of positive integers.
    Uses meet-in-the-middle.
    """
    if not terms_int:
        return target_int == 0
    
    # Quick check
    if sum(terms_int) < target_int:
        return False
    if target_int == 0:
        return False

    # Normalize terms_int for caching
    key = (target_int, tuple(sorted(terms_int)))
    if key in _subset_sum_cache:
        return _subset_sum_cache[key]

    n = len(terms_int)
    if n == 0:
        res = False
    elif n == 1:
        res = (terms_int[0] == target_int)
    else:
        half = n // 2
        left = terms_int[:half]
        right = terms_int[half:]
        
        # Generate all subset sums for left
        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        # Check if any right sum complements a left sum to target
        res = False
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if (target_int - s) in left_sums:
                res = True
                break

    _subset_sum_cache[key] = res
    return res

def _is_subsum(target_frac, terms_list):
    """
    Check if target_frac can be formed by a subset sum of terms_list.
    """
    target_int = int(target_frac * LCM_ALL)
    terms_int = [int(t * LCM_ALL) for t in terms_list]
    return _can_form_subset_sum(target_int, terms_int)

def _is_valid_set(S):
    """
    Check if set S is reciprocal-subsum-free.
    """
    S_list = list(S)
    for a in S_list:
        others = [x for x in S_list if x != a]
        if others:
            if _is_subsum(Fraction(1, a), others):
                return False
    return True

def _greedy_construct(order):
    """
    Construct a valid set by greedily adding elements in the given order.
    """
    S = []
    for a in order:
        trial = S + [a]
        if _is_valid_set(trial):
            S = trial
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Phase 1: Greedy Construction
    # Descending order tends to work well
    best_S = _greedy_construct(list(range(24, 0, -1)))
    
    # Try a few other greedy orders to see if we get a larger base
    for _ in range(5):
        if time.time() > deadline - 10: break
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        S = _greedy_construct(random_order)
        if len(S) > len(best_S):
            best_S = S
            
    # Phase 2: Local Search (Add and Swap)
    S = list(best_S)
    no_improvement_count = 0
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Try to add any element
        in_set = set(S)
        out_set = [x for x in range(1, 25) if x not in in_set]
        
        added = False
        for x in out_set:
            if time.time() > deadline - 5.0: break
            trial = S + [x]
            if _is_valid_set(trial):
                S = trial
                improved = True
                added = True
                break
        
        if added:
            no_improvement_count = 0
            continue
        
        # If no add possible, try swaps to enable future adds
        # We look for a swap (remove y, add x) that results in a valid set
        # This doesn't increase size immediately, but might change the structure
        # to allow an add in the next iteration.
        
        # To avoid getting stuck in loops, we can limit the number of swaps
        # or use a random selection.
        
        swap_performed = False
        # Randomize the order of out_set and S to explore different swaps
        random.shuffle(out_set)
        random.shuffle(S)
        
        for x in out_set:
            if time.time() > deadline - 5.0: break
            for y in S:
                if time.time() > deadline - 5.0: break
                # Trial swap
                trial = [z for z in S if z != y] + [x]
                if _is_valid_set(trial):
                    # Check if this swap might lead to an improvement
                    # Heuristic: Does the new set have "room" for an add?
                    # We just accept the swap if it's valid and we haven't done too many
                    # without an add.
                    S = trial
                    swap_performed = True
                    improved = True
                    break
            if swap_performed:
                break
        
        if not improved:
            no_improvement_count += 1
            # If stuck for a while, try a random restart or perturbation
            if no_improvement_count > 50:
                # Random restart with a small valid set to break out
                # Or just break and try a new greedy order
                no_improvement_count = 0
                # Try a new greedy order
                random_order = list(range(1, 25))
                random.shuffle(random_order)
                new_S = _greedy_construct(random_order)
                if len(new_S) > len(S):
                    S = new_S
                else:
                    # Perturb current S: remove a random element and try to add
                    if S:
                        random_y = random.choice(S)
                        S = [z for z in S if z != random_y]
                        # Try to add something
                        in_set = set(S)
                        out_set = [x for x in range(1, 25) if x not in in_set]
                        random.shuffle(out_set)
                        for x in out_set:
                            trial = S + [x]
                            if _is_valid_set(trial):
                                S = trial
                                break
            else:
                # Small random perturbation
                if S and random.random() < 0.5:
                    random_y = random.choice(S)
                    S = [z for z in S if z != random_y]
                    
    # Final check and return
    if _is_valid_set(S):
        return S
    else:
        # Fallback to best known valid set
        return best_S

# Ensure the function name matches the requirement
# The prompt says: "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.
# I have defined `construct()`.

# One minor detail: The prompt says "Return a nonempty subset of [1,24]".
# My construct returns a list. This is a subset.

# Let's double check the LCM and integer conversion.
# Fraction(1, a) * LCM_ALL is an integer.
# Sum of Fractions * LCM_ALL is an integer.
# The meet-in-the-middle works on these integers.
# This is exact.

# The cache `_subset_sum_cache` might grow large.
# Key size: target_int (up to LCM_ALL), terms_int tuple (up to 24 elements).
# For 90 seconds, this should be manageable.
# To be safe, we can clear the cache if it gets too big, or just let it be.
# Given the limited domain, the number of unique states is limited.

# Final check on the "Swap" logic.
# If we swap y for x, size remains same.
# We need to ensure we don't loop forever between two states.
# The `no_improvement_count` and perturbation help with this.
# Also, `random.shuffle` helps explore the space.

# One optimization: In `_is_valid_set`, we can stop early.
# This is already done.

# Let's refine the "Swap" acceptance.
# Accepting any valid swap might lead to a local minimum.
# But combined with the "Add" step, it should eventually find a larger set.
# If we are stuck at size K, and no add is possible, and no swap leads to a state where an add is possible, we are stuck.
# The perturbation helps escape.

# I will add a check to ensure the returned set is valid.
# If not, I return the best valid set found so far.
# `best_S` is updated in Phase 1.
# In Phase 2, I should update `best_S` whenever `len(S) > len(best_S)`.

# Let's update the code to track `best_S` properly.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_S = []
    
    # Phase 1: Greedy Construction
    # Try descending first
    S = _greedy_construct(list(range(24, 0, -1)))
    if len(S) > len(best_S):
        best_S = S
        
    # Try a few random greedy orders
    for _ in range(10):
        if time.time() > deadline - 20: break
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        S = _greedy_construct(random_order)
        if len(S) > len(best_S):
            best_S = S

    # Phase 2: Local Search
    S = list(best_S)
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Try to add
        in_set = set(S)
        out_set = [x for x in range(1, 25) if x not in in_set]
        
        added = False
        for x in out_set:
            if time.time() > deadline - 5.0: break
            trial = S + [x]
            if _is_valid_set(trial):
                S = trial
                improved = True
                added = True
                if len(S) > len(best_S):
                    best_S = S
                break
        
        if added:
            continue
        
        # Try swaps
        swap_performed = False
        random.shuffle(out_set)
        random.shuffle(S)
        
        for x in out_set:
            if time.time() > deadline - 5.0: break
            for y in S:
                if time.time() > deadline - 5.0: break
                trial = [z for z in S if z != y] + [x]
                if _is_valid_set(trial):
                    S = trial
                    swap_performed = True
                    improved = True
                    break
            if swap_performed:
                break
        
        if not improved:
            # Perturbation
            if S:
                random_y = random.choice(S)
                S = [z for z in S if z != random_y]
                # Try to add
                in_set = set(S)
                out_set = [x for x in range(1, 25) if x not in in_set]
                random.shuffle(out_set)
                for x in out_set:
                    trial = S + [x]
                    if _is_valid_set(trial):
                        S = trial
                        if len(S) > len(best_S):
                            best_S = S
                        break
            else:
                S = [1] # Ensure nonempty

    if _is_valid_set(S) and len(S) > len(best_S):
        best_S = S
        
    return best_S

# The code looks solid.
# I will output the complete program.

