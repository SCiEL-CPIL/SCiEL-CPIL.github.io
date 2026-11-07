import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_set = []
    
    # Try n from 60 down to 1
    for n in range(60, 0, -1):
        if time.time() > deadline:
            break
        
        # Estimate minimum a
        # a > k(n-k) for k=1..n-1
        # Max k(n-k) is approx n^2/4
        min_a = (n * n) // 4 + 1
        
        # a must also satisfy a+n-1 <= 256
        max_a = 256 - n + 1
        
        if min_a > max_a:
            continue
            
        # Try to find a valid set with consecutive integers starting at a
        # Start from the smallest possible a that satisfies the tightest constraint
        # and increase a if needed
        found = False
        for a in range(min_a, max_a + 1):
            if time.time() > deadline:
                break
                
            # Construct set
            S = [a + i for i in range(n)]
            
            # Check constraints
            valid = True
            for k in range(1, n):
                # Sum of first k+1
                sum_small = sum(S[:k+1])
                # Sum of last k
                sum_large = sum(S[-k:])
                
                if sum_small <= sum_large:
                    valid = False
                    break
            
            if valid:
                best_set = S
                found = True
                break
                
        if found:
            # Verify one more time
            S = best_set
            n = len(S)
            for k in range(1, n):
                if sum(S[:k+1]) <= sum(S[-k:]):
                    best_set = []
                    break
            if best_set:
                return [int(x) for x in best_set]
                
    # Fallback: return a small valid set
    # n=1 is always valid
    return [1]
