import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for the 90s limit
    
    max_val = 1000001
    # We need lpf for indices 0..max_val (we access lpf up to 1000001)
    # lpf[i] will store the largest prime factor of i
    # For i=0,1 lpf will be 0 (undefined but we don't care)
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # For each prime p, assign p to all multiples of p
    # We need to identify primes. If lpf[p] == 0, p is prime (for p>=2)
    
    # Optimization: Only iterate up to max_val. 
    # But we can stop early if time is running out? 
    # If we stop early, some lpf values might be incorrect (missing larger prime factors).
    # However, if we process primes in increasing order, the LAST assignment is the largest prime factor.
    # So if we stop early, the lpf values are the largest prime factor LESS THAN OR EQUAL TO the last processed prime.
    # This underestimates the true LPF for numbers with large prime factors.
    # This might lead to a suboptimal (lower) score than the true minimum, which is dangerous?
    # Wait, the task asks to MINIMIZE the largest prime factor.
    # If we underestimate the LPF, we might pick an n that looks good but isn't.
    # The evaluator computes the TRUE LPF. If we return an n with a large true LPF but our estimate was small, we get a bad score.
    # So we MUST compute the full LPF for all numbers in range to be safe.
    # Or, we need to ensure that if we break early, we handle it safely.
    
    # How long does the full sieve take?
    # N = 1,000,001
    # Primes up to N: ~78,498
    # Sum of N/p for primes p ~ N * log(log(N))
    # Operations: ~ 1,000,000 * 2-3 = few million vectorized ops.
    # Should be well under 1 second in NumPy.
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            lpf[p::p] = p
            
            # Time check
            if p % 50000 == 0:
                if time.time() > deadline:
                    # If we timeout, we cannot safely return a result based on incomplete LPF.
                    # We must return a known good result or a safe default.
                    # 10000 is a safe default.
                    return 10000

    if time.time() > deadline:
        return 10000

    # We want to minimize max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # n ranges from 10000 to 1000000
    # lpf[n] corresponds to indices 10000 to 1000000
    # lpf[n+1] corresponds to indices 10001 to 1000001
    
    # Vectorized calculation of scores
    # lpf[10000:1000001] has length 990001 (indices 10000..1000000)
    # lpf[10001:1000002] has length 990001 (indices 10001..1000001)
    
    # Note: The array size is max_val + 1 = 1000002.
    # Index 1000001 is valid.
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    scores = np.maximum(lpf_n, lpf_np1)
    
    if len(scores) == 0:
        return 10000
        
    min_score_idx = np.argmin(scores)
    best_n = 10000 + min_score_idx
    
    return int(best_n)
