import time
import random
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _can_form_sum(target, items):
    if not items:
        return target == 0
    if target < 0:
        return False
    total_sum = sum(items)
    if total_sum < target:
        return False
    
    n = len(items)
    items_sorted = sorted(items, reverse=True)
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items_sorted[i]
        
    found = [False]
    
    def dfs(i, current_sum, is_non_empty):
        if found[0]:
            return
        if i == n:
            if current_sum == target and is_non_empty:
                found[0] = True
            return
        if current_sum + suffix_sums[i] < target:
            return
        if current_sum > target:
            return

        dfs(i + 1, current_sum, is_non_empty)
        if found[0]:
            return
            
        new_sum = current_sum + items_sorted[i]
        if new_sum <= target:
            dfs(i + 1, new_sum, True)
            
    dfs(0, Fraction(0), False)
    return found[0]

def _is_subsum_free(selected, candidate):
    if selected:
        items = [Fraction(1, x) for x in selected]
        target = Fraction(1, candidate)
        if _can_form_sum(target, items):
            return False
    
    for x in selected:
        others = [y for y in selected if y != x]
        others.append(candidate)
        items = [Fraction(1, y) for y in others]
        target = Fraction(1, x)
        if _can_form_sum(target, items):
            return False
            
    return True

def _greedy_construct(order):
    trial_set = []
    for candidate in order:
        if _time_remaining() < 2.0:
            break
        if _is_subsum_free(trial_set, candidate):
            trial_set.append(candidate)
    return trial_set

def construct():
    all_elements = list(range(1, 25))
    best_set = []
    
    # Initial good set
    initial = [7] + list(range(9, 25))
    if _is_subsum_free(initial[:-1], initial[-1]): # Just a check, actually initial is known good
         best_set = initial
    else:
        # Fallback to greedy
        s = _greedy_construct(sorted(all_elements, reverse=True))
        best_set = s

    # Random restarts
    random.seed(42)
    for _ in range(500):
        if _time_remaining() < 10.0:
            break
        candidates = list(all_elements)
        random.shuffle(candidates)
        trial_set = _greedy_construct(candidates)
        if len(trial_set) > len(best_set):
            best_set = trial_set
            
    # Local search: try to add elements to best_set
    # If we can't add, try swapping?
    # For now, just return best found.
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
