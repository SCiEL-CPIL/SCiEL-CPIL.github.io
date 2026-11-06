import time

def construct():
    p = 1009
    n = 1008

    # Deadline guard as per directive
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    # Check time after precomputation (should be negligible)
    if time.time() > deadline:
        return [1, 1] # Fallback, though this will never happen

    # A[x] = min(x, inv[x]), B[x] = max(x, inv[x])
    # We need M(s) = min { B[x] | A[x] >= s }
    # Initialize min_B_for_A with infinity
    min_B_for_A = [float('inf')] * (n + 2)
    
    for x in range(1, n + 1):
        a = min(x, inv[x])
        b = max(x, inv[x])
        if b < min_B_for_A[a]:
            min_B_for_A[a] = b

    # Compute suffix minimum of min_B_for_A
    # M[s] = min(min_B_for_A[s], M[s+1])
    M = [float('inf')] * (n + 2)
    current_min = float('inf')
    for s in range(n, 0, -1):
        if min_B_for_A[s] < current_min:
            current_min = min_B_for_A[s]
        M[s] = current_min

    best_start = 1
    best_length = 0

    for s in range(1, n + 1):
        # Check time
        if time.time() > deadline:
            break
        
        # Max valid end e is M[s] - 1
        max_e = M[s] - 1
        if max_e > n:
            max_e = n
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    # Final verification step (lightweight, only checks the found interval)
    s = best_start
    e = s + best_length - 1
    valid = True
    for x in range(s, e + 1):
        if (x * x) % p == 1:
            valid = False
            break
        if valid:
            for y in range(x + 1, e + 1):
                if (x * y) % p == 1:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known safe small interval if logic failed (should not happen)
        return [2, 1]

    return [best_start, best_length]
