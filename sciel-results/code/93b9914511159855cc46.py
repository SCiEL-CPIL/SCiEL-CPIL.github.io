import random
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    best_set = []
    
    # Pre-compute candidates
    all_candidates = list(range(1, 513))
    
    def check_and_add(x, B, current_sum, max_k):
        """
        Checks if x is valid given state B and current_sum.
        If valid, returns updated B and new_sum.
        If invalid, returns None.
        """
        # Check if x is mean of any subset of size k >= 2
        # Sum of subset must be k*x
        # Max possible sum is current_sum
        for k in range(2, max_k + 1):
            target = k * x
            if target > current_sum:
                break
            if (B[k] >> target) & 1:
                return None
        
        # If valid, update B
        new_B = B[:]
        # Update from high k to low k to use old B[k-1]
        for k in range(max_k, 0, -1):
            new_B[k] = B[k] | (B[k-1] << x)
        
        return new_B, current_sum + x

    # We will use a greedy approach with random restarts.
    # To speed up, we can limit the number of candidates we consider in each run?
    # No, we should consider all 512 to maximize the set.
    
    # Optimization: Sort candidates in a specific way for some restarts?
    # Random is good for diversity.
    
    max_subset_size = 49 # We only care about subset sizes up to 48
    
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        candidates = all_candidates[:]
        random.shuffle(candidates)
        
        current_set = []
        B = [0] * max_subset_size
        B[0] = 1
        current_sum = 0
        max_k = 1 # Current max subset size in B that is non-zero? 
                  # Actually B[k] is 0 for k > len(current_set).
                  # We can just iterate up to len(current_set) + 1?
                  # But we need to check up to 48.
                  # If len(current_set) is small, k*x will exceed current_sum quickly.
        
        for x in candidates:
            # We only need to check k up to len(current_set)
            # Because B[k] is 0 for k > len(current_set)
            # So max_k_check = len(current_set)
            if len(current_set) < 2:
                # No subset of size >= 2 exists yet, so x is always valid
                current_set.append(x)
                # Update B
                new_B = B[:]
                for k in range(1, max_subset_size):
                    new_B[k] = B[k] | (B[k-1] << x)
                B = new_B
                current_sum += x
            else:
                res = check_and_add(x, B, current_sum, len(current_set))
                if res is not None:
                    B, current_sum = res
                    current_set.append(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]

    return best_set
