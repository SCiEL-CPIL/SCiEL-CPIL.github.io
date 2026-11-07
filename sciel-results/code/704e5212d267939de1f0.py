import numpy as np

def construct():
    N = 10000000
    
    # Sieve for sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    best_diff = float('inf')
    best_a, best_b = 1, 1
    
    # We want to minimize |a/b - 1.5| = |2a - 3b| / (2b)
    # Let k = 2a - 3b. We want small |k| and large b.
    # k=0 implies a/b = 1.5 exactly, which is invalid (error 0).
    # So we iterate |k| starting from 1.
    
    # Upper bound for |k|:
    # If |k| is large, the minimal possible error |k|/(2*b_max) might exceed current best_diff.
    # b_max <= N.
    # So if |k| / (2*N) >= best_diff, we can stop.
    # |k| >= 2 * N * best_diff.
    
    k_abs = 1
    while True:
        # Check if minimal possible error for this k_abs is already worse than best
        # Max b is roughly N (actually (2N-k)/3 ~ 2N/3)
        # Min error approx |k_abs| / (2 * (2N/3)) = 3|k_abs| / 4N
        # If this is >= best_diff, no solution with this k_abs can improve.
        if 3.0 * k_abs / (4.0 * N) >= best_diff:
            break
        
        for k in [k_abs, -k_abs]:
            # 2a - 3b = k
            # a = (3b + k) / 2
            # a must be integer.
            # 3b + k must be even.
            # If k is even, 3b must be even -> b even.
            # If k is odd, 3b must be odd -> b odd.
            
            # Range for b:
            # 1 <= b <= N
            # 1 <= a <= N  =>  1 <= (3b+k)/2 <= N
            # 2 <= 3b+k <= 2N
            # 2-k <= 3b <= 2N-k
            # b >= ceil((2-k)/3)
            # b <= floor((2N-k)/3)
            
            b_min = max(1, (2 - k + 2) // 3) # ceil((2-k)/3)
            # Note: integer division for ceil(a/b) is (a+b-1)//b for positive.
            # Here 2-k can be negative.
            # Let's just use math.ceil or careful integer logic.
            # Python // is floor division.
            # ceil(x) = -floor(-x)
            # b_min = max(1, -((-2+k) // 3))
            
            # Let's re-calculate carefully.
            # 3b >= 2-k
            # b >= (2-k)/3
            b_min = max(1, int(np.ceil((2.0 - k) / 3.0)))
            
            # 3b <= 2N-k
            # b <= (2N-k)/3
            b_max = min(N, int((2 * N - k) / 3))
            
            if b_min > b_max:
                continue
                
            # Parity constraint
            # k even -> b even
            # k odd -> b odd
            if k % 2 == 0:
                if b_min % 2 != 0:
                    b_min += 1
                if b_max % 2 != 0:
                    b_max -= 1
            else:
                if b_min % 2 == 0:
                    b_min += 1
                if b_max % 2 == 0:
                    b_max -= 1
            
            if b_min > b_max:
                continue
                
            # We want to check b in descending order to find the largest b first.
            # Stop if error for current b is >= best_diff.
            # Error = |k| / (2b).
            # If |k| / (2b) >= best_diff, then b <= |k| / (2 * best_diff).
            # So we only need to check b > |k| / (2 * best_diff).
            
            b_stop = int(abs(k) / (2.0 * best_diff)) if best_diff > 0 else 0
            # We need b > b_stop.
            # So we iterate from b_max down to max(b_min, b_stop + 1).
            
            b_start = max(b_min, b_stop + 1)
            
            if b_start > b_max:
                continue
                
            # Iterate b downwards
            # We can't use a simple range with step -1 in numpy easily for scalar loop,
            # but a python loop is fine if the number of iterations is small.
            # With pruning, it should be very small.
            
            for b in range(b_max, b_start - 1, -1):
                # Check parity again? b_min/b_max adjusted, step is 2?
                # No, the range b_min..b_max contains numbers of correct parity.
                # But range(b_max, b_start-1, -1) steps by 1.
                # We need to step by 2 to maintain parity?
                # Or just check parity inside.
                # Better: construct the range with step 2.
                pass # Placeholder for logic below

            # Let's refine the loop to respect parity step.
            # If we adjusted b_min and b_max to have correct parity,
            # and we step by 2, we are good.
            
            # Determine step
            step = -2
            
            # Ensure b_max has correct parity (it does, from adjustment)
            # Ensure b_start has correct parity?
            # b_start = max(b_min, b_stop + 1).
            # b_min has correct parity.
            # b_stop + 1 might not.
            # We need the largest b <= b_max with correct parity AND b >= b_min AND b > b_stop.
            
            # Let's just find the starting b properly.
            b_curr = b_max
            # b_max has correct parity.
            
            # Check if b_curr is within valid range (>= b_min and > b_stop)
            if b_curr < b_min or b_curr <= b_stop:
                continue
                
            while b_curr >= b_min and b_curr > b_stop:
                a = (3 * b_curr + k) // 2
                # a is integer by construction of parity and k
                if 1 <= a <= N:
                    if sigma[a] == sigma[b_curr]:
                        diff = abs(k) / (2.0 * b_curr)
                        if diff < best_diff:
                            best_diff = diff
                            best_a, best_b = a, b_curr
                        # We found the largest b for this k.
                        # Any smaller b for this k will have larger diff.
                        # So we can break the while loop for this k.
                        break
                b_curr += step
                
        k_abs += 1
        
    return [int(best_a), int(best_b)]
