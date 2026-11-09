import time

def construct():
    N = 123457
    
    # Precompute largest prime factor for all numbers up to N
    # lp[i] stores the largest prime factor of i
    lp = [0] * (N + 1)
    lp[1] = 1
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            # i is prime
            lp[i] = i
            for j in range(2 * i, N + 1, i):
                if lp[j] == 0:
                    lp[j] = i
                # Ensure we store the LARGEST prime factor
                # Since we iterate i from 2 upwards, if we just assign lp[j]=i when 0,
                # we get the smallest. We need the largest.
                # A better way:
                # lp[j] = max(lp[j], i) is not correct if lp[j] is 0.
                # Actually, if we iterate primes in increasing order, the last prime divisor is the largest.
                # So we can just do:
                # if lp[j] == 0: lp[j] = i
                # else: lp[j] = i (since i is larger than any previous prime factor)
                # Wait, the loop above `if lp[j] == 0: lp[j] = i` only sets it once.
                # We need to update it.
                pass

    # Let's redo the LPF calculation correctly and efficiently.
    # Standard sieve for largest prime factor.
    lp = [0] * (N + 1)
    lp[1] = 1
    for i in range(2, N + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lp[j] = i  # Since i increases, the last assignment is the largest prime factor
    
    best_a = 1
    best_b = N - 1
    best_score = max(lp[best_a], lp[best_b])
    
    # Iterate a from 1 to N//2
    # We check all pairs (a, N-a)
    for a in range(1, N // 2 + 1):
        b = N - a
        pa = lp[a]
        pb = lp[b]
        current_score = pa if pa > pb else pb
        
        if current_score < best_score:
            best_score = current_score
            best_a = a
            best_b = b
            
            # Early termination if we hit the theoretical minimum?
            # The minimum possible prime factor is 2.
            # But one number is odd, so its LPF is at least 3 (unless it's 1, LPF=1).
            # If a=1, b=N. P(1)=1, P(N)=N (if N is prime) or smaller.
            # 123457 is not divisible by 2, 3, 5.
            # 123457 / 7 = 17636.7
            # 123457 / 11 = 11223.3
            # 123457 / 13 = 9496.6
            # 123457 / 17 = 7262.1
            # 123457 / 19 = 6497.7
            # 123457 / 23 = 5367.6
            # 123457 / 29 = 4257.1
            # 123457 / 31 = 3982.4
            # 123457 / 37 = 3336.6
            # 123457 / 41 = 3011.1
            # 123457 / 43 = 2871.0 -> 43 * 2871 = 123453. No.
            # 123457 / 47 = 2626.7
            # 123457 / 53 = 2329.3
            # 123457 / 59 = 2092.4
            # 123457 / 61 = 2023.8
            # 123457 / 67 = 1842.6
            # 123457 / 71 = 1738.8
            # 123457 / 73 = 1691.1
            # 123457 / 79 = 1562.7
            # 123457 / 83 = 1487.4
            # 123457 / 89 = 1387.1
            # 123457 / 97 = 1272.7
            
            # If we find score 2, it's impossible for both to be even.
            # If we find score 3, one must be 3-smooth (powers of 2 and 3) and the other 3-smooth.
            # Since sum is odd, one is even (2-smooth), one is odd (3-smooth odd part).
            # If best_score becomes 3, we can't do better except 2 (impossible) or 1 (impossible for sum > 2).
            # Actually, P(1)=1. If a=1, P(a)=1. P(b)=P(123456).
            # 123456 = 2^5 * 3 * 1291? 
            # 123456 / 32 = 3858. 3858 / 2 = 1929. 1929 / 3 = 643.
            # 643 is prime? sqrt(643) approx 25.
            # 643 not div by 2,3,5,7 (643=7*91+6), 11 (643=11*58+5), 13 (643=13*49+6), 17 (643=17*37+14), 19 (643=19*33+16), 23 (643=23*27+22).
            # So 643 is prime. P(123456) = 643.
            # So a=1 gives score 643.
            
            # If we find score 5, it's very good.
            # If best_score == 2, break (impossible).
            # If best_score == 3, break? 
            # Can we have max(P(a), P(b)) = 2? No, sum is odd.
            # Can we have max(P(a), P(b)) = 3?
            # a, b must be 3-smooth.
            # a=2^x 3^y, b=2^u 3^v.
            # a+b = 123457.
            # If such a pair exists, score is 3.
            # 3 is the smallest prime.
            # So if we find 3, it's optimal.
            if best_score == 3:
                break
            # Wait, is 3 the minimum?
            # P(1)=1.
            # If a=1, P(a)=1. P(b)=P(N-1).
            # If P(N-1) is small, score is small.
            # But we are minimizing max(P(a), P(b)).
            # If we find a pair with score 3, we cannot find score 2.
            # So 3 is the global minimum possible for N > 2.
            
    return [best_a, best_b]
