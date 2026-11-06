import time
import random

N = 512

def _can_add(in_set, x):
    """
    Check if adding x to the set represented by in_set (list of bools)
    creates a 4-term arithmetic progression.
    in_set[i] is True if i is in the set.
    """
    # x is the 4th term: a, a+d, a+2d, x => a=x-3d
    # d can be 1 to (x-1)//3
    for d in range(1, (x - 1) // 3 + 1):
        if in_set[x - d] and in_set[x - 2 * d] and in_set[x - 3 * d]:
            return False
            
    # x is the 3rd term: a, a+d, x, x+d => a=x-2d
    # d can be 1 to (x-1)//2, and x+d <= N
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if in_set[x - d] and in_set[x - 2 * d] and in_set[x + d]:
            return False
            
    # x is the 2nd term: a, x, x+d, x+2d => a=x-d
    # d can be 1 to min(x-1, (N-x)//2)
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
            return False
            
    # x is the 1st term: x, x+d, x+2d, x+3d
    # d can be 1 to (N-x)//3
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
            return False
            
    return True

def construct():
    deadline = time.time() + 50.0
    best_size = 0
    best_set = []
    
    # Run multiple restarts to find a better local optimum
    while time.time() < deadline:
        # 1. Greedy Initialization
        # We can try different orderings. Standard 1..N is a good baseline.
        # Randomized greedy can find different local optima.
        order = list(range(1, N + 1))
        # Mix in some randomness: 50% chance to shuffle, 50% sorted
        if random.random() < 0.5:
            random.shuffle(order)
        
        in_set = [False] * (N + 1)
        current_size = 0
        
        for x in order:
            if _can_add(in_set, x):
                in_set[x] = True
                current_size += 1
        
        # 2. Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if _can_add(in_set, x):
                    in_set[x] = True
                    current_size += 1
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # If no simple add, try swaps
            # Pick a random element not in set
            if not candidates:
                break
                
            x = random.choice(candidates)
            
            # Find conflicts for x
            # We need to find which elements block x.
            # Instead of full conflict list, we can just try swapping with random elements
            # that might be part of a conflict.
            # But a more directed approach:
            # Check if x can be added if we remove some y.
            
            # Let's try a random swap: remove y, add x.
            # y should be chosen such that it might resolve a conflict.
            # Conflicts involve x.
            # Let's just pick a random y in the set and try to swap.
            # This is less efficient but simpler.
            
            elements_in_set = [i for i in range(1, N + 1) if in_set[i]]
            if not elements_in_set:
                break
            
            # Try a few random swaps
            swaps_tried = 0
            max_swaps = 10
            while swaps_tried < max_swaps and time.time() < deadline:
                swaps_tried += 1
                y = random.choice(elements_in_set)
                
                # Tentatively remove y
                in_set[y] = False
                current_size -= 1
                
                # Try to add x
                if _can_add(in_set, x):
                    in_set[x] = True
                    current_size += 1
                    improved = True
                    break
                else:
                    # Revert
                    in_set[y] = True
                    current_size += 1
            
            if not improved and current_size > best_size:
                best_size = current_size
                best_set = [i for i in range(1, N + 1) if in_set[i]]
        
        if current_size > best_size:
            best_size = current_size
            best_set = [i for i in range(1, N + 1) if in_set[i]]
            
    return sorted(best_set)
