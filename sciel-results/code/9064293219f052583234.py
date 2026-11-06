import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    best_set = []
    best_size = 0
    
    # Helper: Check if a set is valid
    def is_valid(S):
        if not S:
            return True
        S_list = list(S)
        n = len(S_list)
        for a in S_list:
            others = [x for x in S_list if x != a]
            if not others:
                continue
            
            # DP for subset sums of 'others'
            # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
            # Max sum is approx 48 * 512 = 24576
            max_sum = sum(others)
            dp = [0] * (n)  # dp[0] to dp[n-1]
            dp[0] = 1  # sum 0 with 0 elements
            
            for y in others:
                # Update in reverse order of counts to avoid using y multiple times
                # Actually, we iterate counts from current_max_count down to 0
                # But we need to know current max count.
                # Simpler: iterate counts from n-1 down to 1
                for k in range(n - 1, 0, -1):
                    # Shift dp[k-1] by y and OR into dp[k]
                    dp[k] |= (dp[k-1] << y)
            
            # Check if a is the mean of any subset of others
            # Mean condition: sum = k * a for some k >= 2
            for k in range(2, n):
                target_sum = k * a
                if target_sum > max_sum:
                    break
                # Check if bit target_sum is set in dp[k]
                if (dp[k] >> target_sum) & 1:
                    return False
        return True

    # Multi-start greedy
    for trial in range(200):
        if time.time() > deadline:
            break
            
        # Randomize order
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        for x in candidates:
            if time.time() > deadline:
                break
            # Try adding x
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
            # If we reached 48, we can stop early? 
            # The task says "Maximize cardinality. At most 48".
            # So 48 is the max possible.
            if best_size == 48:
                break

    # Local search to try to improve if stuck below 48
    # We can try to swap elements
    if best_size < 48:
        current_S = list(best_set)
        current_set = set(current_S)
        
        # Try to add more elements by removing some
        # This is complex, so we just rely on the multi-start greedy.
        # The greedy with random shuffle is quite robust.
        
    return best_set
