import time
import random

def count_violations(S):
    """
    Counts how many elements in S are the arithmetic mean of a subset of at least 2 other elements in S.
    """
    if len(S) < 3:
        return 0
        
    violations = 0
    S_list = list(S)
    
    for i, a in enumerate(S_list):
        others = [x for j, x in enumerate(S_list) if j != i]
        
        # DP for subset sums of 'others' with exactly k elements
        # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
        # Max sum is sum(others) <= 512 * 50 = 25600
        # We only need to check up to max_sum
        max_sum = sum(others)
        
        # Initialize dp
        # dp[0] = 1 (sum 0 with 0 elements)
        dp = [0] * (len(others) + 1)
        dp[0] = 1
        
        for x in others:
            # Update dp in reverse order of k to avoid using x multiple times
            # k ranges from current_max_k down to 0
            # We can just iterate from len(others)-1 down to 0, but we need to know which k are active.
            # Simpler: iterate k from len(others)-1 down to 0.
            # Actually, we only need to go up to the number of elements processed so far.
            # But since we are iterating over elements one by one, we can just go from max_k down to 0.
            # To be safe and simple, we iterate k from len(others)-1 down to 0.
            # However, dp[k] is 0 for k > number of elements processed so far.
            
            # Optimization: track max_k_so_far
            # But for small N (50), just doing full range is fine.
            for k in range(len(others) - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
                    
        # Check for violations
        # For a to be a mean of a subset T of size k >= 2 from others:
        # sum(T) = k * a
        # So we check if bit (k * a) is set in dp[k]
        
        is_violation = False
        # k can be from 2 to len(others)
        for k in range(2, len(others) + 1):
            target = k * a
            if target > max_sum:
                break # Sums only increase, so no need to check larger k if target > max_sum? 
                      # Wait, max_sum is fixed for 'others'. If k*a > max_sum, it's impossible.
            if (dp[k] >> target) & 1:
                is_violation = True
                break
                
        if is_violation:
            violations += 1
            
    return violations

def construct():
    deadline = time.time() + 55.0 # Safety margin
    
    # Initial set: Try to build a large set greedily
    # Start with empty, add random numbers? Or start with a known good structure?
    # Let's start with a random set of size 40
    # Or better: start with a greedy set that tries to add numbers 1..512
    # Greedy might get stuck small. Random might be better for SA start.
    
    best_set = set()
    best_cost = float('inf')
    
    # Helper to get cost
    def get_cost(S):
        v = count_violations(S)
        return 10000 * v - len(S)
    
    # Initial solution: Random set of size 45
    # We want to explore high cardinality.
    initial_size = 45
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    current_set = set(candidates[:initial_size])
    
    current_cost = get_cost(current_set)
    if current_cost < best_cost:
        best_cost = current_cost
        best_set = set(current_set)
        
    T = 100.0 # Initial temperature
    cooling = 0.999
    
    all_numbers = set(range(1, 513))
    
    while time.time() < deadline:
        if T < 0.01:
            # Reheat? Or just keep going with low T (hill climbing)
            T = 100.0 # Reheat to escape local minima
            
        # Generate neighbor
        # Move types:
        # 1. Swap: remove x, add y
        # 2. Add: add y (if size < max)
        # 3. Remove: remove x (if size > min)
        
        move_type = random.random()
        new_set = set(current_set)
        
        if move_type < 0.7: # Swap
            if not current_set:
                continue
            x = random.choice(list(current_set))
            # Pick y not in current_set
            # To be efficient, pick from all_numbers - current_set
            # This set is large (512 - 45 = ~467). 
            # Random choice from a set of size 467 is fast.
            available = list(all_numbers - current_set)
            if not available:
                continue
            y = random.choice(available)
            new_set.remove(x)
            new_set.add(y)
            
        elif move_type < 0.85: # Add
            # Only if we haven't hit 512 (impossible) or if we want to grow
            available = list(all_numbers - current_set)
            if not available:
                continue
            y = random.choice(available)
            new_set.add(y)
            
        else: # Remove
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set.remove(x)
            
        new_cost = get_cost(new_set)
        delta_cost = new_cost - current_cost
        
        # Acceptance criteria
        if delta_cost <= 0 or random.random() < pow(2.718281828, -delta_cost / T):
            current_set = new_set
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = set(current_set)
                
        T *= cooling
        
    return list(best_set)
