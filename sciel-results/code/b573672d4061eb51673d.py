import time
import random
from itertools import combinations

L = 5354228880  # lcm(1..24)
N = 24

# Precompute weights
W = [0] + [L // i for i in range(1, N + 1)]

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle for speed.
    """
    if target == 0:
        return True
    if not weights:
        return False
    
    # Filter weights > target (positive weights only)
    weights = [w for w in weights if w <= target]
    if not weights:
        return False
    if sum(weights) < target:
        return False
        
    n = len(weights)
    if n <= 20:
        # Simple DP or bitmask for small n
        # Actually for n<=20, 2^20 is ~1M, might be ok but let's stick to MITM for consistency
        pass
        
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = {0}
    for w in left:
        new_sums = set()
        for s in left_sums:
            if s + w <= target:
                new_sums.add(s + w)
        left_sums.update(new_sums)
        
    right_sums = {0}
    for w in right:
        new_sums = set()
        for s in right_sums:
            if s + w <= target:
                new_sums.add(s + w)
        right_sums.update(new_sums)
        
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def is_valid(S):
    """
    Check if S is a valid set.
    S is a list of ints.
    """
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = W[x]
        weights = [W[y] for y in S_set if y != x]
        if subset_sum_exists(weights, target):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin
    
    best_set = None
    best_size = 0
    
    # 1. Try to find a large set using a smart greedy + local search approach
    # The key insight is that larger numbers have smaller reciprocals (larger weights in LCM terms? No, smaller weights).
    # Wait, 1/x is smaller for larger x.
    # L/x is smaller for larger x.
    # So large x contribute small weights.
    # The condition is: 1/x != sum(1/y_j)
    # L/x != sum(L/y_j)
    # Since L/x is small for large x, it's hard to sum up to it using other small weights?
    # Actually, if we pick all large numbers, the weights are small.
    # Let's check the seed: range(13, 25) -> 13..24. Size 12.
    # Can we add smaller numbers?
    # Smaller numbers have larger weights.
    # If we add 12, weight L/12.
    # Does L/12 = sum of some L/y for y in 13..24?
    # L/12 = 446185740
    # L/13 = 411863760
    # L/14 = 382444920
    # ...
    # It's possible.
    
    # Strategy:
    # 1. Start with a large set of "safe" elements (e.g., large numbers).
    # 2. Try to insert smaller elements one by one.
    # 3. Use local search to remove conflicting elements and try to add more.
    
    # Let's use a randomized local search (Simulated Annealing-like or Hill Climbing)
    # State: a set S.
    # Objective: maximize |S| subject to validity.
    # Move: add a random element, or remove a random element, or swap.
    # Acceptance: if valid, accept. If invalid, reject (for hill climbing).
    # To escape local optima, we can allow temporary invalid states? No, validity is hard to check incrementally.
    # Better: Start with a valid set. Try to add elements. If adding x fails, try removing a conflicting element y and adding x.
    
    # Let's try a "Greedy Insertion with Conflict Resolution"
    
    def try_insert(S, x):
        """Try to insert x into S. Returns (new_S, success)."""
        if x in S:
            return S, True
        new_S = list(S) + [x]
        if is_valid(new_S):
            return new_S, True
        return S, False
    
    # Initial sets to try
    initial_sets = [
        list(range(13, 25)),  # 12 elements
        list(range(10, 25)),  # 15 elements?
        list(range(1, 25)),   # All 24 (likely invalid)
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13],
    ]
    
    # Validate initial sets and pick the largest valid one
    for S_init in initial_sets:
        if is_valid(S_init):
            if len(S_init) > best_size:
                best_size = len(S_init)
                best_set = S_init
    
    if best_set is None:
        # Fallback: single element
        best_set = [24]
        best_size = 1
    
    # Local Search: Hill Climbing with Random Restarts
    # We will try to improve the current best set.
    
    def local_search(S, max_steps=200):
        S = list(S)
        S_set = set(S)
        current_size = len(S)
        steps = 0
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            improved = False
            
            # 1. Try adding a missing element
            missing = [i for i in range(1, N + 1) if i not in S_set]
            random.shuffle(missing)
            for m in missing:
                new_S, ok = try_insert(S, m)
                if ok:
                    S = new_S
                    S_set = set(S)
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swapping: remove one, add one
            # Try to remove an element that is "blocking" many additions.
            # Heuristic: Remove the smallest element? Or random?
            # Let's try random removal and then try to add a missing one.
            for _ in range(10): # Try 10 random swaps
                if time.time() > deadline:
                    break
                if not S:
                    break
                remove_idx = random.randint(0, len(S) - 1)
                removed = S.pop(remove_idx)
                S_set.remove(removed)
                
                # Now try to add a missing element
                missing = [i for i in range(1, N + 1) if i not in S_set]
                random.shuffle(missing)
                added = False
                for m in missing:
                    new_S, ok = try_insert(S, m)
                    if ok:
                        S = new_S
                        S_set = set(S)
                        added = True
                        break
                
                if not added:
                    # Put it back
                    S.append(removed)
                    S_set.add(removed)
                else:
                    improved = True
                    break
        
        return S
    
    # Run local search from the best initial set
    if time.time() < deadline:
        improved_set = local_search(best_set, max_steps=500)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Random Restarts
    # Generate random valid sets and run local search on them.
    # How to generate a random valid set?
    # Start with empty, add random elements if valid.
    
    def random_greedy():
        S = []
        S_set = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for c in candidates:
            new_S, ok = try_insert(S, c)
            if ok:
                S = new_S
                S_set.add(c)
        return S
    
    attempts = 0
    while time.time() < deadline and attempts < 20:
        attempts += 1
        S_rand = random_greedy()
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_set = S_rand
            
        if time.time() < deadline:
            improved_set = local_search(S_rand, max_steps=100)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        
        # Also try local search on the current best
        if time.time() < deadline:
            improved_best = local_search(best_set, max_steps=100)
            if len(improved_best) > best_size:
                best_size = len(improved_best)
                best_set = improved_best
    
    return best_set

# To ensure reproducibility in some contexts, but random is allowed
random.seed(42)
