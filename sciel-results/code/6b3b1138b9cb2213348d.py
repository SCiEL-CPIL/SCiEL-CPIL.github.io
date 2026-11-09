"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s guard within 60s limit

    n = 5040
    target = 4979
    
    # 1. Get divisors
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort()
    
    # 2. Split into two halves
    mid = len(divs) // 2
    A = divs[:mid]
    B = divs[mid:]
    
    # 3. DP for half A
    # dp[s] = (count, mask)
    # We only need sums up to 'target'
    MAX_S = target
    dp_A = [None] * (MAX_S + 1)
    dp_A[0] = (0, 0)
    
    for i, d in enumerate(A):
        if time.time() > deadline:
            break
        # Iterate backwards to avoid using d multiple times
        # We only care about sums <= MAX_S
        for s in range(MAX_S, d - 1, -1):
            if dp_A[s - d] is not None:
                c, m = dp_A[s - d]
                nc = c + 1
                nm = m | (1 << i)
                if dp_A[s] is None or nc < dp_A[s][0]:
                    dp_A[s] = (nc, nm)
                    
    # 4. DP for half B
    dp_B = [None] * (MAX_S + 1)
    dp_B[0] = (0, 0)
    
    for i, d in enumerate(B):
        if time.time() > deadline:
            break
        for s in range(MAX_S, d - 1, -1):
            if dp_B[s - d] is not None:
                c, m = dp_B[s - d]
                nc = c + 1
                nm = m | (1 << i)
                if dp_B[s] is None or nc < dp_B[s][0]:
                    dp_B[s] = (nc, nm)
                    
    # 5. Combine - optimize by limiting range
    best_count = float('inf')
    best_split_sum = -1
    
    # Calculate valid range for s
    sum_A = sum(A)
    sum_B = sum(B)
    s_min = max(0, target - sum_B)
    s_max = min(target, sum_A)
    
    for s in range(s_min, s_max + 1):
        if time.time() > deadline:
            break
        if dp_A[s] is not None and dp_B[target - s] is not None:
            cA = dp_A[s][0]
            cB = dp_B[target - s][0]
            total_c = cA + cB
            if total_c < best_count:
                best_count = total_c
                best_split_sum = s
                
    if best_split_sum == -1:
        # Fallback to greedy if DP fails or times out (unlikely)
        remainder = target
        chosen = []
        for d in sorted(divs, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
        
    # 6. Reconstruct
    mask_A = dp_A[best_split_sum][1]
    mask_B = dp_B[target - best_split_sum][1]
    
    result = []
    for i in range(len(A)):
        if mask_A & (1 << i):
            result.append(A[i])
    for i in range(len(B)):
        if mask_B & (1 << i):
            result.append(B[i])
            
    # Final verification
    assert sum(result) == target
    assert len(result) == len(set(result))
    assert all(n % d == 0 for d in result)
    
    return result
# EVOLVE-BLOCK-END
