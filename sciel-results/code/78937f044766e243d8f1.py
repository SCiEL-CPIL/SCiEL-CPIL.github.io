import time
import random

def construct():
    N = 512
    # Precompute all 4-APs? 
    # There are roughly N^2 / 2 APs. 512^2 / 2 ~ 130k.
    # Storing them might be memory heavy but manageable.
    # However, checking validity of a specific x against a set S is faster on the fly.
    
    def is_valid_addition(S, x):
        """
        Check if adding x to S creates a 4-AP.
        x can be term 1, 2, 3, or 4.
        """
        # x is term 1: x+d, x+2d, x+3d in S
        # Max d such that x+3d <= N
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return False
        
        # x is term 2: x-d, x+d, x+2d in S
        # Max d such that x-d >= 1 and x+2d <= N
        # x-d >= 1 => d <= x-1
        # x+2d <= N => d <= (N-x)//2
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return False
        
        # x is term 3: x-2d, x-d, x+d in S
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= N => d <= N-x
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return False
        
        # x is term 4: x-3d, x-2d, x-d in S
        # x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return False
        
        return True

    def greedy_construct(order=None):
        if order is None:
            order = list(range(1, N + 1))
        S = set()
        for x in order:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def local_optimize(S, randomize=True):
        """
        Try to add elements to S until no more can be added.
        If randomize, try in random order.
        """
        candidates = [x for x in range(1, N + 1) if x not in S]
        if randomize:
            random.shuffle(candidates)
        
        # We need to re-check because adding one element might invalidate others?
        # No, adding an element only *creates* APs, it doesn't destroy existing ones.
        # So if x is valid now, it remains valid unless we add elements that form an AP with x.
        # Wait, if we add y, and y forms an AP with x, then x is still in S, but the set is invalid.
        # The check `is_valid_addition(S, x)` checks if *x* completes an AP with *existing* elements.
        # If we add x, the set becomes S U {x}.
        # Later, if we add y, we check if y completes an AP with S U {x}.
        # This is correct.
        
        added_any = True
        # Simple approach: iterate through candidates, try to add.
        # If we add some, the remaining candidates might become invalid?
        # No, adding elements to S can only make it *harder* for other candidates to be added (more potential conflicts).
        # So if x was valid, and we add y, x is still in S. The set is still valid.
        # But if we tried to add x and failed, and then we add y, x might still fail.
        # Actually, if x failed because it formed an AP with a, b, c in S, adding y doesn't remove a, b, c. So x will still fail.
        # So we can just iterate once through the list of candidates not in S.
        
        for x in candidates:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    # 1. Initial Greedy
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # 2. Iterated Local Search
    start_time = time.time()
    timeout = 50  # Safety margin for 60s limit
    
    # We will try to improve by perturbing and re-optimizing
    while time.time() - start_time < timeout:
        # Create a perturbed version of the best set
        current_set = set(best_set)
        
        # Randomly remove a few elements
        # The number of removals is a hyperparameter. 
        # Removing too many loses too much. Too few doesn't escape local optima.
        # Let's try removing 3 to 10 elements.
        num_removals = random.randint(3, 10)
        if len(current_set) <= num_removals:
            continue
            
        elements_to_remove = random.sample(list(current_set), num_removals)
        for e in elements_to_remove:
            current_set.remove(e)
            
        # Re-optimize: Add back valid elements
        # Use random order to explore different local optima
        current_set = local_optimize(current_set, randomize=True)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # Optional: If we haven't improved in a while, we could restart from scratch with different random seed?
        # But perturbation of the best found so far is usually good.
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
