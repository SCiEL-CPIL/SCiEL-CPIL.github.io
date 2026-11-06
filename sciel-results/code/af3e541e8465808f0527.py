import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_n = 512
    max_size = 48
    
    def is_valid(S):
        """
        Check if S is a valid set.
        No element y in S is the mean of any subset T of S\{y} with |T| >= 2.
        Condition: sum(T) != |T| * y for all T subset S\{y}, |T| >= 2.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return True
        
        # For each y, check subsets of S\{y}
        # We can precompute DP for S\{y} for each y?
        # Or just loop.
        
        for i, y in enumerate(S_list):
            # Build DP for S without y
            # S_without_y = S_list[:i] + S_list[i+1:]
            # Size is n-1
            # Max sum is sum(S_without_y)
            # We only care about sums up to (n-1) * y?
            # Actually, sum(T) = k * y. Max k is n-1.
            # So we only need to track sums up to (n-1) * y.
            # But y varies.
            # Let's just use a large enough bitset.
            # Max possible sum is sum(S) < 512 * 48 = 24576.
            
            # Optimization: 
            # We need to check if there exists k >= 2 such that
            # bit (k*y) is set in dp[k] for S\{y}.
            
            # Initialize dp for S\{y}
            # dp[k] is bitset of sums achievable with k elements from S\{y}
            # Max k is n-1.
            dp = [0] * n # indices 0..n-1
            dp[0] = 1
            
            # Iterate through elements in S\{y}
            # To avoid modifying S_list, we just skip index i
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                # Update dp in reverse
                # Current max size is j (if i > j) or j-1 (if i < j)?
                # It's easier to just track current_max_k
                # Or simply iterate k from current_count down to 0.
                # Let's maintain a list of elements
                pass
            
            # Simpler: Just build the list
            elements = [x for j, x in enumerate(S_list) if j != i]
            m = len(elements)
            
            # dp[k] for k=0..m
            dp = [0] * (m + 1)
            dp[0] = 1
            current_max_k = 0
            
            for x in elements:
                # Update dp
                # k goes from current_max_k down to 0
                for k in range(current_max_k, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
                current_max_k += 1
                
            # Check for conflicts
            # For each k >= 2, check if bit (k*y) is set in dp[k]
            for k in range(2, m + 1):
                target_sum = k * y
                # If target_sum is beyond the max possible sum, skip
                # Max sum is sum(elements)
                # We can just check the bit
                if (dp[k] >> target_sum) & 1:
                    return False
            
        return True

    def greedy_build(order):
        S = []
        # We need to check validity incrementally?
        # The incremental check in the original program was faster than full re-validation.
        # But for simplicity and robustness in the new structure, 
        # we can use incremental check if we maintain the DP.
        # Let's use the incremental DP approach from the seed, but with a given order.
        
        dp = [0] * (max_size + 1)
        dp[0] = 1
        current_set = []
        
        for x in order:
            if time.time() > deadline - 5.0:
                break
            if len(current_set) >= max_size:
                break
                
            safe = True
            # Check if adding x violates condition for any y in current_set
            # y = mean(T U {x}) => sum(T) = y*(|T|+1) - x
            # T is subset of current_set, size k >= 1.
            for y in current_set:
                for k in range(1, len(current_set) + 1):
                    target_sum = y * (k + 1) - x
                    if target_sum < 0:
                        continue
                    if (dp[k] >> target_sum) & 1:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe:
                current_set.append(x)
                # Update DP
                for k in range(len(current_set) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
                    
        return current_set

    # 1. Generate initial candidates
    candidates = []
    
    # Descending
    candidates.append(greedy_build(range(max_n, 0, -1)))
    
    # Ascending
    candidates.append(greedy_build(range(1, max_n + 1)))
    
    # Random orders
    random.seed(42)
    for _ in range(10):
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        candidates.append(greedy_build(order))
        
    # Find the best initial set
    best_set = max(candidates, key=len)
    
    # 2. Local Search / Improvement
    # Try to add elements or swap to increase size
    # We will use a simple hill climbing with random swaps to escape local minima
    
    current_set = list(best_set)
    current_size = len(current_set)
    
    # Pre-compute a list of all numbers
    all_numbers = set(range(1, max_n + 1))
    
    # We will try to improve for a certain time
    while time.time() < deadline - 10.0:
        if current_size >= max_size:
            break
            
        # Try to add any valid element
        added = False
        remaining = list(all_numbers - set(current_set))
        random.shuffle(remaining)
        
        for x in remaining:
            new_set = current_set + [x]
            if is_valid(new_set):
                current_set = new_set
                current_size += 1
                added = True
                break
                
        if added:
            continue
            
        # If no addition possible, try swaps
        # Pick a random y in current_set and random x not in current_set
        # Check if swap is valid
        # To speed up, we can just try a few random swaps
        swapped = False
        for _ in range(50): # Try 50 random swaps
            if not current_set:
                break
            y = random.choice(current_set)
            x = random.choice(remaining)
            
            new_set = [v for v in current_set if v != y]
            new_set.append(x)
            
            if is_valid(new_set):
                current_set = new_set
                swapped = True
                break
                
        if not swapped and not added:
            # Stuck. Try restarting local search with a different random seed?
            # Or just break and try another greedy restart?
            # For now, break to save time for other restarts if we were doing multiple.
            # But we are in a single loop.
            # Let's try a few more random swaps with a different 'x' selection?
            # Actually, if we are stuck, we might be in a local minimum.
            # We can try to remove a random element and see if we can add something else?
            # That's effectively a swap.
            # If 50 random swaps fail, it's likely a hard local minimum.
            # We can break out and try a new greedy restart if we had budget.
            # But we are in the "improve" phase.
            # Let's just break.
            break

    return current_set
