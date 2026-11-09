import math

def construct():
    N = 123457
    
    # Sieve for Largest Prime Factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_p = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N // 2 + 1):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            if best_max_p == 2: # Theoretical minimum for even N, but N is odd so min is at least 3?
                # If N is odd, one is even (P>=2), one is odd (P>=3). Max >= 3.
                # If N is even, both can be odd? No, sum of two odds is even.
                # If N is odd, one even one odd.
                # Min possible max prime factor:
                # If N is odd, can we have P(a)=2, P(b)=3?
                # a=2^k, b=3^m. 2^k + 3^m = N.
                # If such a pair exists, max is 3.
                # We can break early if we find 3.
                break
                
    return [best_a, best_b]
