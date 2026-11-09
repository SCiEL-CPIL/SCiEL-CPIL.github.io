import time
import math

def construct():
    t_start = time.time()
    deadline = t_start + 55.0
    
    MAX_N = 10_000_000
    
    # 1. Linear Sieve for Smallest Prime Factor (SPF)
    spf = [0] * (MAX_N + 1)
    primes = []
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            spf[i * p] = p
            
    if time.time() > deadline - 5:
        return [6, 11]

    # 2. Helper to calculate sigma(n) using SPF
    def calc_sigma(n):
        if n == 1:
            return 1
        res = 1
        while n > 1:
            p = spf[n]
            s = 1
            pk = 1
            while n % p == 0:
                n //= p
                pk *= p
                s += pk
            res *= s
        return res

    best_err = float('inf')
    best_pair = (6, 11)
    
    # 3. Iterate b downwards to find large b with small error
    # We can stop if the theoretical minimum error for current b 
    # is greater than or equal to the best error found.
    # Min error for odd b is 1/(2b). For even b is 1/b.
    # The absolute minimum for any b' <= b is 1/(2b) (if b is odd) or 1/b (if even).
    # Actually, for any b', min err is >= 1/(2b').
    # So if 1/(2*b) >= best_err, we can stop.
    
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline - 3:
            break
            
        # Termination condition
        # The smallest possible error for this b (and any smaller b) 
        # is bounded by 1/(2*b). 
        # If our current best is smaller than the max possible error for 
        # any future b (which is roughly 1/(2*b_current)), we can stop.
        # Actually, future b are smaller, so 1/(2*b_future) is larger.
        # So we CANNOT stop based on future b having larger min error.
        # We want to find the GLOBAL minimum.
        # The global minimum is likely at the largest b that has a match.
        # So we should just iterate all the way down?
        # No, if we find a match at b=10^7 with err=5e-8, 
        # any b < 10^7 has min err > 5e-8.
        # So if we find a match, we can stop immediately?
        # Wait. If we find a match at b=10^7, is it the best?
        # Yes, because for any b' < 10^7, min_err(b') > min_err(10^7).
        # And the actual error for b=10^7 is at least min_err(10^7).
        # So the first valid pair found when iterating downwards from MAX_N 
        # is guaranteed to be the global minimum?
        
        # Let's verify.
        # err(b) >= 1/(2b).
        # If we find a valid pair at b, err(b) is some value >= 1/(2b).
        # For any b' < b, err(b') >= 1/(2b') > 1/(2b).
        # But err(b) could be much larger than 1/(2b) if the "closest" a doesn't match sigma.
        # Example: b=10^7. Closest a gives diff 1. err = 5e-8.
        # If sigma(a) != sigma(b), we check next a? 
        # My logic only checks the 2 closest a's.
        # If they don't match, there is NO pair for this b with error close to min.
        # The next closest a has diff 3 (for odd b) or 4 (for even b)?
        # For odd b: 2a = 3b +/- 1 (diff 1). Next is 3b +/- 3 (diff 3).
        # So if closest doesn't work, the error jumps to 3/(2b).
        # Still, 3/(2b) > 1/(2b).
        # And for any b' < b, min possible error is 1/(2b') > 1/(2b).
        # It is possible that 1/(2b') < 3/(2b)?
        # 1/b' < 3/b => b > 3b'.
        # So if b is much larger than b', the "second best" for b could be better than "best" for b'.
        # So we cannot just take the first match.
        
        # However, we can prune.
        # If we have a best_err, and for current b, the minimum possible error 
        # (which is 1/(2b)) is >= best_err, then no future b (smaller) 
        # can have a min error < best_err?
        # No, future b are smaller, so 1/(2b_future) is LARGER.
        # So if 1/(2b) >= best_err, then for all b' <= b, 1/(2b') >= 1/(2b) >= best_err.
        # So no b' <= b can have an error < best_err.
        # So we CAN stop if 1/(2*b) >= best_err.
        
        if 1.0 / (2.0 * b) >= best_err:
            break
            
        # Check candidates for current b
        sb = calc_sigma(b)
        
        if b % 2 == 0:
            # b even, 3b even. a = 1.5b +/- 1
            a1 = (3 * b) // 2 - 1
            a2 = (3 * b) // 2 + 1
            candidates = [a1, a2]
        else:
            # b odd, 3b odd. a = (3b +/- 1) / 2
            a1 = (3 * b - 1) // 2
            a2 = (3 * b + 1) // 2
            candidates = [a1, a2]
            
        for a in candidates:
            if a < 1 or a > MAX_N:
                continue
            if a == b:
                continue
            if calc_sigma(a) == sb:
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue
                err = diff / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
    return [int(best_pair[0]), int(best_pair[1])]
