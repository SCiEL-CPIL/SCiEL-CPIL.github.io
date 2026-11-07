import time
import math

def construct():
    n = 256
    
    # 1. Precomputation
    is_prime = [True] * (n + 1)
    if n >= 0: is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n+1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * (n + 1)
    masks[1] = 0
    for i in range(2, n + 1):
        mask = 0
        for p in primes:
            if i % p == 0:
                mask |= (1 << prime_index[p])
        masks[i] = mask
        
    numbers = list(range(n, 1, -1))
    
    # Greedy Initialization
    greedy_set = []
    greedy_mask = 0
    greedy_sum = 0
    for v in numbers:
        if (masks[v] & greedy_mask) == 0:
            greedy_set.append(v)
            greedy_mask |= masks[v]
            greedy_sum += v
            
    best_sum = greedy_sum
    best_set = greedy_set[:]
    
    # Add 1
    best_sum += 1
    best_set.append(1)
    
    start_time = time.time()
    deadline = 85.0
    
    current_set = greedy_set[:] # Start DFS from greedy state? 
    # No, standard B&B explores the space. 
    # Starting from greedy state as a "lower bound" is good.
    # But we need to explore to see if we can do better.
    # We can start DFS from scratch, but use best_sum from greedy for pruning.
    
    # To make it faster, we can try to "repair" the greedy solution?
    # No, B&B is exhaustive.
    
    # We will run B&B from the start (idx=0) to find the global optimum.
    # The greedy solution provides a strong initial lower bound.
    
    def dfs(idx, current_sum, used_mask, current_set):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() - start_time > deadline:
            raise TimeoutError()
            
        # Upper bound: current_sum + sum of remaining feasible numbers
        bound_sum = current_sum
        for j in range(idx, len(numbers)):
            if (masks[numbers[j]] & used_mask) == 0:
                bound_sum += numbers[j]
                
        if bound_sum <= best_sum:
            return
            
        if idx == len(numbers):
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        val = numbers[idx]
        mask = masks[val]
        
        # Option 1: Include val
        if (used_mask & mask) == 0:
            current_set.append(val)
            dfs(idx + 1, current_sum + val, used_mask | mask, current_set)
            current_set.pop()
            
        # Option 2: Exclude val
        dfs(idx + 1, current_sum, used_mask, current_set)
        
    try:
        # Start with empty set for B&B, but best_sum is already set by greedy
        dfs(0, 0, 0, [])
    except TimeoutError:
        pass
        
    # Ensure 1 is in the final best_set
    # The DFS only considers 2..256.
    # We added 1 to best_sum and best_set in greedy init.
    # If DFS finds a better set of 2..256, it updates best_set (without 1).
    # We need to add 1 back to the final result.
    if 1 not in best_set:
        best_set.append(1)
    # best_sum already includes 1?
    # In greedy init: best_sum += 1.
    # In DFS: current_sum is sum of 2..256 subset.
    # When DFS updates best_sum, it does `best_sum = current_sum`. This LOSES the +1.
    # So we must be careful.
    
    # Let's fix the logic:
    # best_sum_global = max_sum_of_2_to_256 + 1
    # In DFS, we track sum of 2..256.
    # Compare DFS current_sum + 1 with best_sum_global?
    # Or just keep best_sum as sum of 2..256, and add 1 at the end.
    
    # Let's restart the logic slightly to be cleaner.
    
    # Re-implementation of the core logic to ensure correctness
    
    # Reset
    best_sum_256 = greedy_sum
    best_set_256 = greedy_set[:]
    
    def dfs2(idx, current_sum, used_mask, current_set):
        nonlocal best_sum_256, best_set_256
        
        if time.time() - start_time > deadline:
            raise TimeoutError()
            
        bound_sum = current_sum
        for j in range(idx, len(numbers)):
            if (masks[numbers[j]] & used_mask) == 0:
                bound_sum += numbers[j]
                
        if bound_sum <= best_sum_256:
            return
            
        if idx == len(numbers):
            if current_sum > best_sum_256:
                best_sum_256 = current_sum
                best_set_256 = current_set[:]
            return
            
        val = numbers[idx]
        mask = masks[val]
        
        if (used_mask & mask) == 0:
            current_set.append(val)
            dfs2(idx + 1, current_sum + val, used_mask | mask, current_set)
            current_set.pop()
            
        dfs2(idx + 1, current_sum, used_mask, current_set)
        
    try:
        dfs2(0, 0, 0, [])
    except TimeoutError:
        pass
        
    best_set_256.append(1)
    return best_set_256
