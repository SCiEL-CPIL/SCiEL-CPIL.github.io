import time
from fractions import Fraction
import random

_START_TIME = time.time()
DEADLINE = 85.0  # 5 seconds safety margin

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _can_form_target(target, items):
    """
    Check if target can be formed by a subset of items (Fractions).
    Uses a pruned DFS. Returns True if found, False otherwise.
    """
    if not items:
        return target == Fraction(0)
    
    # Filter items: only those < target can be in a sum equal to target
    # (since all are positive). If any item == target, return True.
    filtered = []
    for v in items:
        if v == target:
            return True
        if v < target:
            filtered.append(v)
    
    if not filtered:
        return False
        
    # Sort descending to prune faster
    filtered.sort(key=lambda x: x, reverse=True)
    n = len(filtered)
    
    # Suffix sums
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + filtered[i]
        
    if suffix_sums[0] < target:
        return False

    # Memoization: (idx, current_sum) -> bool
    # Since sums are Fractions, we can use a set of tuples
    memo = set()
    
    def dfs(idx, current_sum):
        if current_sum == target:
            return True
        if idx >= n:
            return False
        if current_sum > target:
            return False
        if current_sum + suffix_sums[idx] < target:
            return False
            
        key = (idx, current_sum)
        if key in memo:
            return False
            
        # Option 1: Include
        if current_sum + filtered[idx] <= target:
            if dfs(idx + 1, current_sum + filtered[idx]):
                return True
                
        # Option 2: Exclude
        if dfs(idx + 1, current_sum):
            return True
            
        memo.add(key)
        return False
        
    return dfs(0, Fraction(0))

def _check_violation(S, a):
    """
    Check if 1/a can be formed by a subset of {1/x for x in S, x != a}.
    Returns True if violation found, else False.
    """
    target = Fraction(1, a)
    items = []
    for x in S:
        if x != a:
            items.append(Fraction(1, x))
            
    return _can_form_target(target, items)

def is_valid(S):
    for a in S:
        if _check_violation(S, a):
            return False
    return True

def get_violation(S):
    """Returns (a, T) if violation exists, else None. T is the subset (list of values) causing it."""
    for a in S:
        target = Fraction(1, a)
        items = [Fraction(1, x) for x in S if x != a]
        if _can_form_target(target, items):
            # We don't strictly need T for the greedy removal logic in the current program,
            # but the original program used it. 
            # However, for 'min' strategy, we just pick min(a) + min(T).
            # Let's just return a dummy T or find one?
            # Actually, the original code returned T.
            # To keep the interface similar or just return True/False is enough for the new logic?
            # The current program's `greedy_solve` uses `a` and `T`.
            # `y = min(candidates)` where `candidates = [a] + T`.
            # If we don't have T, we can't do that exact heuristic.
            # But we can just remove `a`? Or remove the element corresponding to the largest reciprocal?
            # Let's stick to removing `a` or the element that "caused" the most issues.
            # For simplicity and speed, let's just return `a` and let the caller decide.
            # Or better, let's find the minimal subset? That's expensive.
            # Let's just return (a, None) and modify greedy_solve to handle it.
            return (a, None)
    return None

def greedy_remove_strategy(S_init, strategy='remove_target'):
    """
    Greedy removal from S_init.
    strategy: 
      - 'remove_target': remove the 'a' in the violation.
      - 'remove_largest': remove the largest element in S.
      - 'random': remove a random element.
    """
    S = list(S_init)
    while _time_left() > 1.0 and not is_valid(S):
        viol = get_violation(S)
        if viol is None:
            break
        a, T = viol
        if strategy == 'remove_target':
            y = a
        elif strategy == 'remove_largest':
            y = max(S)
        elif strategy == 'random':
            y = random.choice(S)
        else:
            y = a
        S.remove(y)
    return S

def greedy_add_strategy(order, best_size):
    """
    Grow a set by adding elements in `order`, keeping validity.
    """
    chosen = []
    recips = []
    for v in order:
        if v in chosen:
            continue
        # If we already have a set larger than best_size, we only care if we can beat it
        # But we must maintain validity.
        # Try adding v
        trial = chosen + [v]
        # Optimization: if len(trial) <= best_size and we are just trying to find a valid set of size > best_size,
        # we still need to build a valid set. 
        # Actually, the goal is to maximize size. 
        # If we can't beat best_size, we might as well skip? No, we need a valid set to compare.
        # But if we are building from scratch, we just want the largest valid set we can form in this order.
        
        # Check if 1/v is a subset sum of current recips
        if _can_form_target(Fraction(1, v), recips):
            continue
        
        # Check if adding v breaks validity of existing elements
        valid = True
        for u in chosen:
            # 1/u must not be a subset sum of (recips + [1/v])
            # This is equivalent to: 1/u is not a subset sum of recips (already true) 
            # AND 1/u is not a subset sum of (recips + [1/v]) that includes 1/v.
            # i.e., 1/u - 1/v is a subset sum of recips.
            # Note: 1/u > 1/v if u < v. If u > v, 1/u < 1/v, so 1/u - 1/v is negative, impossible.
            if u < v:
                target_diff = Fraction(1, u) - Fraction(1, v)
                if target_diff > 0 and _can_form_target(target_diff, recips):
                    valid = False
                    break
        
        if valid:
            chosen = trial
            recips.append(Fraction(1, v))
            
    return chosen

def construct():
    best_set = []
    best_size = 0
    
    # 1. Structural anchors
    # Seed: [13..24]
    S0 = list(range(13, 25))
    if is_valid(S0):
        best_set = S0
        best_size = len(S0)
        
    # Top halves
    for start in range(2, 25):
        if _time_left() < 2.0:
            break
        S = list(range(start, 25))
        if len(S) <= best_size:
            continue
        if is_valid(S):
            best_set = S
            best_size = len(S)
            
    # 2. Greedy Removal from Full Set
    # Try different removal strategies
    full_set = list(range(1, 25))
    
    # Strategy: Remove target (the element a in 1/a = sum...)
    if _time_left() > 5.0:
        S = greedy_remove_strategy(full_set, 'remove_target')
        if len(S) > best_size and is_valid(S):
            best_set = S
            best_size = len(S)
            
    # Strategy: Remove largest
    if _time_left() > 5.0:
        S = greedy_remove_strategy(full_set, 'remove_largest')
        if len(S) > best_size and is_valid(S):
            best_set = S
            best_size = len(S)
            
    # 3. Greedy Add
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        list(range(1, 25, 2)) + list(range(2, 25, 2)),
        list(range(24, 0, -2)) + list(range(23, 0, -2))
    ]
    
    for order in orders:
        if _time_left() < 2.0:
            break
        S = greedy_add_strategy(order, best_size)
        if len(S) > best_size and is_valid(S):
            best_set = S
            best_size = len(S)
            
    # 4. Randomized Greedy Add
    rng = random.Random(42)
    for _ in range(50):
        if _time_left() < 2.0:
            break
        order = list(range(1, 25))
        rng.shuffle(order)
        S = greedy_add_strategy(order, best_size)
        if len(S) > best_size and is_valid(S):
            best_set = S
            best_size = len(S)
            
    # 5. Local Search / Swap
    # Try to improve best_set by swapping elements or adding/removing
    # If best_set is valid, try to add elements not in it
    if best_set and _time_left() > 5.0:
        current = list(best_set)
        improved = True
        while improved and _time_left() > 1.0:
            improved = False
            for v in range(1, 25):
                if v in current:
                    continue
                trial = current + [v]
                if is_valid(trial):
                    current = trial
                    improved = True
                    break # Restart loop with new set
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
        # Try swapping: remove one, add one
        if _time_left() > 5.0:
            for i in range(len(current)):
                if _time_left() < 1.0:
                    break
                for v in range(1, 25):
                    if v in current:
                        continue
                    trial = current[:i] + current[i+1:] + [v]
                    trial = list(set(trial)) # Ensure no duplicates, though set logic above handles it
                    if len(trial) != len(current):
                        continue
                    if is_valid(trial):
                        # Check if it's actually better? Size is same. 
                        # Maybe it allows further additions? 
                        # For now, let's just stick to size maximization.
                        # If sizes are equal, we don't necessarily swap unless it helps future steps.
                        # But local search might get stuck. 
                        # Let's prioritize size.
                        pass

    if not best_set:
        best_set = list(range(13, 25))
        
    return best_set

if __name__ == "__main__":
    print(construct())
