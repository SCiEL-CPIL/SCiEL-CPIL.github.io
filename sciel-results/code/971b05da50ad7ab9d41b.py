import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    selected = []
    
    # We want to maximize size.
    # Greedy from largest to smallest usually works well for "large sums" separation.
    # Or smallest to largest?
    # If we pick small numbers, sums are small.
    # If we pick large numbers, sums are large.
    # The condition is min_sum_k > max_sum_{k-1}.
    # This is easier to satisfy if numbers are large and close together?
    # Let's stick to descending order as in the original, but with the fast check.
    
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
            
        # Tentative set
        new_selected = selected + [x]
        new_selected.sort()
        
        valid = True
        n = len(new_selected)
        
        # Precompute prefix sums (sum of k smallest)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + new_selected[i]
            
        # Precompute suffix sums (sum of k largest)
        # suffix[k] = sum of largest k elements
        # largest are at the end of new_selected
        suffix = [0] * (n + 1)
        for i in range(n):
            # i=0 corresponds to 0 largest? No.
            # Let's just compute sum of largest k directly or via reverse prefix.
            pass
            
        # Easier:
        # max_sum_{k-1} is sum of largest k-1 elements.
        # min_sum_k is sum of smallest k elements.
        
        for k in range(1, n + 1):
            min_sum_k = prefix[k]
            
            # Sum of largest k-1 elements
            if k - 1 == 0:
                max_sum_prev = 0
            else:
                # Sum of elements from index n-(k-1) to n-1
                # prefix[n] - prefix[n-(k-1)]
                max_sum_prev = prefix[n] - prefix[n - (k - 1)]
                
            if min_sum_k <= max_sum_prev:
                valid = False
                break
                
        if valid:
            selected = new_selected

    return selected
