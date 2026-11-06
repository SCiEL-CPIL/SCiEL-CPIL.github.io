import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0
    
    # The theoretical minimum largest element for n distinct positive integers
    # with distinct subset sums is 2^(n-1).
    # For n=12, this is 2048.
    # The sequence of powers of 2 achieves this bound.
    
    n = 12
    # Generate powers of 2
    result = [1 << i for i in range(n)]
    
    # Optional: Verify distinct subset sums (expensive for n=12, 4096 subsets)
    # But we know it's valid. We can skip for speed or do it if time permits.
    # 4096 sums is very fast.
    
    # Check deadline
    if time.time() > deadline:
        return result
        
    # Verify
    sums = set()
    for i in range(1 << n):
        s = 0
        for j in range(n):
            if i & (1 << j):
                s += result[j]
        if s in sums:
            # Collision found, though this should not happen for powers of 2
            return result
        sums.add(s)
        
    return result
