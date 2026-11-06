import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1

    # Sieve to compute the largest prime factor for each number up to MAX_VAL
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (MAX_VAL + 1)
    
    # Standard sieve modification: iterate through numbers, if lpf[i] is 0, it's prime.
    # Then mark all multiples of i with i as their largest prime factor?
    # No, if we iterate i from 2 upwards, and i is prime, we set lpf[j] = i for all multiples j.
    # Since we go in increasing order of i, the last prime that divides j will be the largest one.
    # Example: j=12.
    # i=2 (prime): lpf[2]=2, lpf[4]=2, lpf[6]=2, lpf[8]=2, lpf[10]=2, lpf[12]=2
    # i=3 (prime): lpf[3]=3, lpf[6]=3, lpf[9]=3, lpf[12]=3
    # i=5 (prime): lpf[5]=5, lpf[10]=5
    # i=7 (prime): lpf[7]=7
    # ...
    # For j=12, lpf[12] becomes 3.
    # Wait. 12 = 2^2 * 3. Largest prime factor is 3.
    # The algorithm sets lpf[12] to 2 first, then to 3.
    # So lpf[12] ends up as 3. Correct.
    # For j=6, lpf[6] becomes 2, then 3. Ends up 3. Correct.
    # For j=4, lpf[4] becomes 2. No other primes divide 4. Ends up 2. Correct.
    
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i

    best_n = N_MIN
    best_val = float('inf')
    
    # Scan for the n that minimizes max(lpf[n], lpf[n+1])
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
            
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n

    return best_n
