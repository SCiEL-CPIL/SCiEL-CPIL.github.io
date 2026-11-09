import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        # Add d to all multiples of d
        sigma[d::d] += d
        # Check time periodically
        if d % 1_000_000 == 0:
            if time.time() > deadline:
                # If we run out of time during sieve, return a known valid pair
                # 11 and 6: sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = |1.833 - 1.5| = 0.333
                return [11, 6]
    
    # Search for best pair
    best_err = float('inf')
    best_pair = [11, 6] # Fallback
    
    # We want to minimize |a/b - 1.5| = |2a - 3b| / (2b)
    # Let k = 2a - 3b. We want small |k| and large b.
    # a = (3b + k) / 2
    # For a to be integer, 3b + k must be even.
    # If k is even, b can be any integer.
    # If k is odd, b must be odd.
    
    # Max b such that a <= N
    # (3b + k) / 2 <= N => 3b + k <= 2N => b <= (2N - k) / 3
    # For k=0, b_max = 2N/3 = 6666666
    # For k=1, b_max = (2N-1)/3 = 6666666
    # For k=-1, b_max = (2N+1)/3 = 6666667
    
    max_k = 20
    
    for k in range(1, max_k + 1):
        # Check k and -k
        for sign in [1, -1]:
            kk = sign * k
            # Calculate max b
            # 3b + kk <= 2N => b <= (2N - kk) / 3
            b_max = int((2 * N - kk) / 3)
            if b_max > N:
                b_max = N
            if b_max < 1:
                continue
            
            # Determine start step and parity
            # 3b + kk is even.
            # If kk is even, 3b must be even => b even.
            # If kk is odd, 3b must be odd => b odd.
            
            if kk % 2 == 0:
                # b must be even
                if b_max % 2 != 0:
                    b_max -= 1
                step = -2
                b_start = b_max
            else:
                # b must be odd
                if b_max % 2 == 0:
                    b_max -= 1
                step = -2
                b_start = b_max
            
            # Iterate b downwards
            for b in range(b_start, 0, step):
                a = (3 * b + kk) // 2
                if a < 1 or a > N:
                    continue
                
                # Check sigma
                if sigma[a] == sigma[b]:
                    # Calculate error
                    # err = abs(a/b - 1.5) = abs(2a - 3b) / (2b) = abs(kk) / (2b)
                    err = abs(kk) / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = [int(a), int(b)]
                        # If we found a pair with k=1, it's likely the best possible
                        # since error scales with 1/b and k=1 is minimal non-zero k.
                        # We can break early if k=1 and b is large enough?
                        # Actually, we just keep the best found.
                        # But if k=1, we found the max b for k=1.
                        # Any other k will have larger numerator.
                        # To be safe, we continue checking other k's but break inner loop.
                        break
            
            # Time check
            if time.time() > deadline:
                return best_pair
    
    return best_pair
