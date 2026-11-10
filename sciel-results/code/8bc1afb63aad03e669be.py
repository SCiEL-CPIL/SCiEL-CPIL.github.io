import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds limit, leaving 10s margin
    
    N = 1000000
    # We need lpf for indices up to N+1 because we check n+1
    # Max n is 1000000, so max n+1 is 1000001
    max_val = N + 1
    lpf = [1] * (max_val + 1)
    
    # Sieve for Largest Prime Factor
    # Complexity O(N log log N)
    for i in range(2, max_val + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    # Check time
    if time.time() > deadline:
        # Fallback to seed if sieve took too long (unlikely)
        return 10000

    # Find n in [10000, N] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_score = max(lpf[10000], lpf[10001])
    
    # We can break early if we find a very small number, 
    # but we want the global min. 
    # Theoretical min is 2 (if n and n+1 are powers of 2, impossible for n>1) 
    # or 3 (n=2, n+1=3 -> max(2,3)=3. But n>=10000).
    # Let's just scan the whole range.
    
    for n in range(10000, N):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n+1])
        if val < best_score:
            best_score = val
            best_n = n
            
    return best_n
