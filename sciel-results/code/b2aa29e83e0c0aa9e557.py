import time
from collections import defaultdict
from array import array

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115  # Leave 5s for final processing
    
    # 1. Estimate performance
    N_test = 100_000
    sigma_test = array('Q', [0]) * (N_test + 1)
    t_start = time.time()
    for i in range(1, N_test + 1):
        for j in range(i, N_test + 1, i):
            sigma_test[j] += i
    t_test = time.time() - t_start
    
    # Estimate time for N
    # T(N) ~ N log N
    # log(N_test) ~ 11.5
    # We want T(N) < 90s
    # T(N) = t_test * (N log N) / (N_test log N_test)
    # N log N < 90 / t_test * N_test log N_test
    
    max_ops_factor = (90 / t_test) if t_test > 0 else 1000
    target_ops = max_ops_factor * N_test * 11.5 # approx log N_test
    
    # Solve N log N = target_ops
    # Try a few values
    N_est = 10_000_000
    while N_est * (N_est.bit_length()) > target_ops and N_est > 1_000_000:
        N_est //= 2
    while N_est * (N_est.bit_length()) < target_ops * 0.8 and N_est < 50_000_000:
        N_est *= 2
        
    N = min(N_est, 20_000_000) # Cap at 20M to be safe
    
    # 2. Run Sieve
    sigma = array('Q', [0]) * (N + 1)
    t_start = time.time()
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
        # Check time
        if i % 10000 == 0:
            if time.time() > hard_deadline - 20:
                # If we are running out of time, we might not finish.
                # But we are in the middle of the sieve.
                # We cannot return partial sigma.
                # So we should have estimated better.
                # If this happens, we have a problem.
                pass

    # 3. Find Best M
    solutions = defaultdict(list)
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            solutions[M].append(k)
            
    best_M = 0
    best_k = []
    for M, ks in solutions.items():
        if len(ks) > len(best_k):
            best_k = ks
            best_M = M
            
    return {'M': best_M, 'k': best_k}
