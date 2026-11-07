import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Search for maximum k such that there exists an AP [a, a+1, ..., a+k-1]
    # within 1..256 where sum ranges for different cardinalities are disjoint.
    # Condition: for all i in 1..k-1:
    #   i*(a+k-1) - i*(i-1)//2 < (i+1)*a + i*(i+1)//2
    
    best_k = 0
    best_a = 1
    
    # Try all possible k from 60 down to 1, and for each try all valid a
    for k in range(60, 0, -1):
        # a must satisfy: 1 <= a and a+k-1 <= 256, so a <= 257-k
        a_max = 257 - k
        if a_max < 1:
            continue
        
        found = False
        for a in range(1, a_max + 1):
            valid = True
            for i in range(1, k):
                max_i_sum = i * (a + k - 1) - i * (i - 1) // 2
                min_ip1_sum = (i + 1) * a + i * (i + 1) // 2
                if max_i_sum >= min_ip1_sum:
                    valid = False
                    break
            if valid:
                best_k = k
                best_a = a
                found = True
                break
        
        if found:
            break
    
    if best_k == 0:
        # Fallback to a known valid set
        best_k = 31
        best_a = 226
    
    result = list(range(best_a, best_a + best_k))
    
    # Verify
    assert len(set(result)) == len(result)
    assert all(1 <= x <= 256 for x in result)
    
    return result
# EVOLVE-BLOCK-END
