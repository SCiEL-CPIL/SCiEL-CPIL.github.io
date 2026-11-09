import time

def construct():
    start_time = time.time()
    deadline = 120  # Safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Candidate N values
    N_candidates = [10**6, 2 * 10**6, 4 * 10**6, 8 * 10**6]
    
    for N in N_candidates:
        # Estimate time: if previous took T, next takes ~4T
        # Just check if we have enough time
        elapsed = time.time() - start_time
        # Heuristic: if elapsed > 30, maybe skip large N?
        # Let's just try and break if deadline is close
        if elapsed > deadline - 20:
            break
            
        # Compute sigma
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
        
        # Check time again
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time to process and find best, but we have sigma
            # We can still process, but maybe skip if it's too close
            pass

        # Find M with most k
        M_to_k = {}
        for k in range(1, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                if M not in M_to_k:
                    M_to_k[M] = []
                M_to_k[M].append(k)
        
        # Find best
        current_best_M = None
        current_best_k_list = None
        max_count = 0
        for M, k_list in M_to_k.items():
            if len(k_list) > max_count:
                max_count = len(k_list)
                current_best_M = M
                current_best_k_list = k_list
        
        if current_best_M is not None:
            if len(current_best_k_list) > len(best_k):
                best_M = current_best_M
                best_k = current_best_k_list
                
    return {'M': best_M, 'k': best_k}
