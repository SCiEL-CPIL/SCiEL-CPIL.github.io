import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 512
    MAX_K = 48
    
    def is_valid_fast(S):
        """
        Checks if S is a valid set.
        S is a list of distinct integers.
        Returns True if no element is the mean of a subset of >=2 other elements.
        """
        m = len(S)
        if m < 3:
            return True
        
        # Sort for deterministic behavior and potential pruning, though not strictly necessary for bitset
        S_sorted = sorted(S)
        
        # For each element a, check if it is a mean
        for i, a in enumerate(S_sorted):
            others = S_sorted[:i] + S_sorted[i+1:]
            n_others = len(others)
            
            # Max sum of k elements is k * max(others) <= k * 512
            # We only care about sums equal to k * a
            # k ranges from 2 to n_others
            
            # Bitset DP
            # dp[k] is an integer where bit s is 1 if sum s is achievable with k elements
            # Max sum we care about is n_others * 512. But we can limit to max(k*a) for k in range.
            # Actually, k*a increases with k.
            # Let's just run DP for all k up to n_others.
            
            # Optimization: If a is very small, k*a is small. If a is large, k*a is large.
            # We can early terminate if we find a conflict.
            
            # Initialize dp
            # dp[0] = 1 (sum 0 with 0 elements)
            # We only need to track sums up to max_target = n_others * a? No, sum of others can be larger.
            # But we only check specific targets k*a.
            # Let's use a list of integers for dp.
            
            max_possible_sum = n_others * 512
            # We don't need to track sums > max_target_for_any_k
            # max_target = n_others * a
            # But a is fixed.
            
            # To save memory/time, we can cap the bitset length.
            # But Python integers are arbitrary precision, so it's fine.
            
            dp = [0] * (n_others + 1)
            dp[0] = 1
            
            for v in others:
                # Iterate backwards to avoid using v multiple times in the same step
                # But we are iterating over elements, so each element is used at most once.
                # Standard 0/1 knapsack style update.
                for k in range(n_others, 0, -1):
                    if dp[k-1]:
                        dp[k] |= dp[k-1] << v
            
            # Check for conflicts
            for k in range(2, n_others + 1):
                target = k * a
                if target <= max_possible_sum:
                    if (dp[k] >> target) & 1:
                        return False
                else:
                    # If target is larger than max possible sum, it's not achievable
                    # But if target is small, we already checked.
                    pass
        
        return True

    # Heuristic: Greedy construction
    def greedy_construct():
        selected = []
        # Randomize order to find different solutions
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(selected) >= MAX_K:
                break
            candidate_set = selected + [x]
            if is_valid_fast(candidate_set):
                selected.append(x)
        
        return selected

    best_set = []
    best_size = 0
    
    # Try multiple random restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        current_set = greedy_construct()
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if best_size >= MAX_K:
            return best_set
        
        # Local Search to improve current_set
        # If greedy didn't reach MAX_K, try to swap elements to increase size
        # Or just try to add more if possible (greedy might have been blocked by order)
        
        # Let's do a simple hill climbing on the set
        # Represent set as a boolean array or just a set object
        current = set(current_set)
        
        for _ in range(1000): # Max local search steps
            if time.time() > deadline - 2.0:
                break
                
            if len(current) >= MAX_K:
                break
                
            # Try to add a random element
            available = [x for x in range(1, N + 1) if x not in current]
            if not available:
                break
                
            random.shuffle(available)
            improved = False
            for x in available[:10]: # Try a few random additions
                new_set = current | {x}
                if is_valid_fast(list(new_set)):
                    current = new_set
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                    break # Restart search loop if improved
            
            if not improved:
                # Try a swap
                if current:
                    a = random.choice(list(current))
                    available_swaps = [x for x in range(1, N + 1) if x not in current]
                    if available_swaps:
                        b = random.choice(available_swaps)
                        new_set = (current - {a}) | {b}
                        if is_valid_fast(list(new_set)):
                            current = new_set
                            if len(current) > best_size:
                                best_size = len(current)
                                best_set = list(current)
    
    return best_set

# EVOLVE-BLOCK-END
