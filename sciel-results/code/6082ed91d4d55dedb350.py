import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill

# Precompute triples for faster sum checking if needed, 
# but direct calculation is fine for small sets.
# We will use a direct check function optimized with local variables.

def _can_add(S_list, x):
    """
    Check if x can be added to the set S_list (which is a list of elements in S).
    S_list does not contain x.
    """
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S_list
    # Iterate over all combinations of 3 from S_list
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            ab_sum = a + b
            for k in range(j + 1, n):
                c = S_list[k]
                if (ab_sum + c) % x == 0:
                    return False
    
    # Check 2: For each a in S_list, a does not divide sum of x + 2 others from S_list
    # Iterate over each a in S_list
    for i, a in enumerate(S_list):
        # Iterate over pairs (b, c) in S_list distinct from a
        # We can iterate over all pairs and skip if b or c is a
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(j + 1, n):
                if k == i:
                    continue
                c = S_list[k]
                if (x + b + c) % a == 0:
                    return False
    
    return True

def _greedy_initialization(order):
    """
    Build a valid set using greedy addition in the given order.
    """
    S = []
    S_set = set()
    for x in order:
        if x in S_set:
            continue
        if _can_add(S, x):
            S.append(x)
            S_set.add(x)
            if len(S) >= _MAX:
                break
    return S, S_set

def _local_search(S, S_set, time_budget):
    """
    Local search to improve the set S.
    Tries to swap elements to increase size or maintain size with better properties.
    """
    start_time = time.time()
    best_S = S[:]
    best_set = set(S_set)
    best_len = len(S)
    
    # Precompute all values
    all_vals = list(range(1, _N + 1))
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # 1. Try to add an element
        if len(S) < _MAX:
            candidates = [v for v in all_vals if v not in S_set]
            random.shuffle(candidates)
            # Try a few candidates
            for x in candidates[:50]:
                if _can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_len:
                        best_len = len(S)
                        best_S = S[:]
                        best_set = set(S_set)
                    break
        
        # 2. Try swaps
        if len(S) >= 1:
            # Try to remove one element and add another
            # Prioritize removing elements that are "problematic" (small)
            # and adding elements that are "safe" (large)
            
            S_list = S[:]
            random.shuffle(S_list)
            
            for a in S_list[:20]: # Try removing up to 20 candidates
                if a not in S_set:
                    continue
                # Temporarily remove a
                S.remove(a)
                S_set.remove(a)
                
                # Try to add a new element
                candidates = [v for v in all_vals if v not in S_set]
                # Sort candidates by value descending (larger is generally safer)
                candidates.sort(reverse=True)
                
                for x in candidates[:50]:
                    if _can_add(S, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        if len(S) > best_len:
                            best_len = len(S)
                            best_S = S[:]
                            best_set = set(S_set)
                        break
                
                # If we didn't improve or add, revert
                if not improved or x not in S_set:
                    # Revert removal of a
                    if a not in S_set:
                        S.append(a)
                        S_set.add(a)
                    else:
                        # This case shouldn't happen if we didn't add x
                        pass
                else:
                    # We added x, so a is permanently removed
                    pass
                
                if improved:
                    break
        
        if not improved:
            # Perturbation: remove 2 random elements to escape local optimum
            if len(S) > 2:
                to_remove = random.sample(S, 2)
                for a in to_remove:
                    S.remove(a)
                    S_set.remove(a)

    return best_S, best_set

def construct():
    """
    Main entry point.
    """
    random.seed(42)
    
    best_S = []
    best_set = set()
    best_len = 0
    
    # Phase 1: Greedy with different orders
    orders = [
        range(_N, 0, -1),  # Largest first
        range(1, _N + 1),  # Smallest first
        sorted(range(1, _N + 1), key=lambda x: random.random()), # Random
    ]
    
    # Add some structured orders
    # Large even, large odd, etc.
    orders.append(sorted([x for x in range(1, _N+1) if x % 2 == 0], reverse=True) + 
                  sorted([x for x in range(1, _N+1) if x % 2 == 1], reverse=True))
    
    for order in orders:
        S, S_set = _greedy_initialization(order)
        if len(S) > best_len:
            best_len = len(S)
            best_S = S
            best_set = S_set
            
    # Phase 2: Local Search from the best initial set
    if time.time() < _DEADLINE - 10:
        budget = min(40.0, _DEADLINE - time.time() - 5)
        S, S_set = _local_search(best_S, best_set, budget)
        if len(S) > best_len:
            best_len = len(S)
            best_S = S
            best_set = S_set

    # Phase 3: Multi-start Local Search
    # Run a few more local searches from slightly perturbed states
    if time.time() < _DEADLINE - 5:
        for _ in range(5):
            if time.time() > _DEADLINE - 5:
                break
            
            # Perturb the best set
            S = best_S[:]
            S_set = set(best_S)
            
            # Remove 1-2 random elements
            if len(S) > 2:
                k = random.randint(1, 2)
                to_remove = random.sample(S, k)
                for a in to_remove:
                    S.remove(a)
                    S_set.remove(a)
            
            # Run a short local search
            budget = min(5.0, _DEADLINE - time.time() - 2)
            S_new, S_set_new = _local_search(S, S_set, budget)
            
            if len(S_new) > best_len:
                best_len = len(S_new)
                best_S = S_new
                best_set = S_set_new

    return best_S
