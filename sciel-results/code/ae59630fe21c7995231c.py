"""Improved sum-free subset finder using greedy with multiple orderings and local search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)


def _is_sum_free(S):
    """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
    s = set(S)
    for a in s:
        for b in s:
            if a + b in s:
                return False
    return True


def _greedy_build(elements, order_key=None, rng=None):
    """Build a sum-free subset greedily by trying elements in a given order."""
    if order_key is not None:
        elems = sorted(elements, key=order_key)
    else:
        elems = list(elements)
        if rng is not None:
            rng.shuffle(elems)
    
    chosen = []
    chosen_set = set()
    # Precompute: for each pair in chosen, which sums are forbidden
    # We'll maintain a set of "forbidden" values that would create a sum-free violation
    
    for x in elems:
        if x in chosen_set:
            continue
        # Check if adding x would violate sum-freeness
        # Violation if: exists a in chosen such that a + x in chosen (x = c - a, so c = a+x)
        #               or exists b in chosen such that x + b in chosen (same check)
        #               or 2*x in chosen (a=b=x case)
        # Actually: a + b = c where a,b,c all in new set
        # If we add x, violations occur if:
        #   - x + y = z for some y,z in chosen_set (x is a, y is b, z is c)
        #   - y + x = z for some y,z in chosen_set (y is a, x is b, z is c) -- same as above
        #   - x + x = z for some z in chosen_set (a=b=x, c=z)
        #   - y + z = x for some y,z in chosen_set (y is a, z is b, x is c)
        
        can_add = True
        for y in chosen_set:
            if y + x in chosen_set:
                can_add = False
                break
            if y + y == x and y != x:
                # y + y = x, so y,y,x would violate
                can_add = False
                break
        if can_add:
            if 2 * x in chosen_set:
                can_add = False
        if can_add:
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen


def _local_search(S, elements, max_iter=5000, rng=None):
    """Improve sum-free subset S by trying swaps: remove one element, add another."""
    if rng is None:
        rng = random.Random()
    
    S_set = set(S)
    S_list = list(S)
    elements_set = set(elements)
    
    for _ in range(max_iter):
        improved = False
        # Try adding a random element not in S
        candidates = [x for x in elements_set if x not in S_set]
        if not candidates:
            break
        rng.shuffle(candidates)
        
        for add_x in candidates[:20]:
            # Check if we can add add_x
            can_add = True
            for y in S_set:
                if y + add_x in S_set:
                    can_add = False
                    break
                if y + y == add_x and y != add_x:
                    can_add = False
                    break
            if can_add and 2 * add_x in S_set:
                can_add = False
            if can_add:
                S_list.append(add_x)
                S_set.add(add_x)
                improved = True
                break
        
        if not improved:
            # Try removing one element and adding another
            if not S_list:
                break
            rem_idx = rng.randrange(len(S_list))
            rem_x = S_list[rem_idx]
            
            # Try to add some element after removing rem_x
            temp_set = set(S_set) - {rem_x}
            for add_x in candidates[:30]:
                if add_x == rem_x:
                    continue
                can_add = True
                for y in temp_set:
                    if y + add_x in temp_set:
                        can_add = False
                        break
                    if y + y == add_x and y != add_x:
                        can_add = False
                        break
                if can_add and 2 * add_x in temp_set:
                    can_add = False
                if can_add:
                    S_list[rem_idx] = add_x
                    S_set.remove(rem_x)
                    S_set.add(add_x)
                    improved = True
                    break
        
        if not improved:
            # Try pure removal of a "bad" element to allow more additions
            # Find elements that block the most additions
            best_removal = None
            best_gain = 0
            for rem_x in S_set:
                temp_set = set(S_set) - {rem_x}
                gain = 0
                for add_x in candidates[:50]:
                    if add_x == rem_x:
                        continue
                    can_add = True
                    for y in temp_set:
                        if y + add_x in temp_set:
                            can_add = False
                            break
                    if can_add and 2 * add_x in temp_set:
                        can_add = False
                    if can_add:
                        gain += 1
                if gain > best_gain + 1:  # net gain must be positive
                    best_gain = gain
                    best_removal = rem_x
            
            if best_removal is not None:
                S_list.remove(best_removal)
                S_set.remove(best_removal)
                improved = True
        
        if not improved:
            break
    
    return S_list


def _solve_instance(elements, time_budget=40.0):
    """Solve a single instance, trying multiple strategies within time budget."""
    start = time.time()
    best = []
    
    elements_set = set(elements)
    n = len(elements)
    
    # Strategy 1: Odd numbers (baseline)
    odd_subset = [x for x in elements if x % 2 != 0]
    if len(odd_subset) > len(best):
        best = list(odd_subset)
    
    # Strategy 2: Greedy with various orderings
    orderings = [
        lambda x: x,           # natural order
        lambda x: -x,          # reverse natural
        lambda x: abs(x),      # by magnitude
        lambda x: -abs(x),     # reverse magnitude
        lambda x: x % 2,       # even first
        lambda x: (x > 0),     # negative first
        lambda x: (x < 0),     # positive first
    ]
    
    for order_key in orderings:
        if time.time() - start > time_budget:
            break
        candidate = _greedy_build(elements, order_key=order_key)
        if len(candidate) > len(best):
            best = candidate
    
    # Strategy 3: Randomized greedy
    rng = random.Random(42)
    for trial in range(30):
        if time.time() - start > time_budget * 0.6:
            break
        candidate = _greedy_build(elements, rng=rng)
        if len(candidate) > len(best):
            best = candidate
    
    # Strategy 4: Local search from best found so far
    if time.time() - start < time_budget * 0.8:
        improved = _local_search(list(best), elements, max_iter=3000, rng=rng)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 5: Another round of local search with different random seed
    if time.time() - start < time_budget * 0.9:
        rng2 = random.Random(123)
        improved = _local_search(list(best), elements, max_iter=2000, rng=rng2)
        if len(improved) > len(best):
            best = improved
    
    # Verify the result
    if not _is_sum_free(best):
        # Fallback to odd numbers
        best = odd_subset
    
    return best


def construct():
    total_deadline = 120.0
    start = time.time()
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803),
    ]
    
    results = []
    for i, elements in enumerate(instances):
        elapsed = time.time() - start
        remaining = total_deadline - elapsed - 5.0  # 5s safety margin
        if remaining < 1.0:
            # Not enough time, use odd numbers
            results.append([x for x in elements if x % 2 != 0])
            continue
        
        # Allocate time proportionally to instance size
        time_budget = min(remaining, 45.0)
        result = _solve_instance(elements, time_budget=time_budget)
        results.append(result)
    
    return results

# EVOLVE-BLOCK-END
