import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s safety margin
    
    # We will search for k up to N, increasing N as time permits.
    # Start with a small N to get a quick result, then expand.
    # The best result so far
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # We will use a dictionary to store M -> list of k's
    # But storing lists for all M is memory heavy.
    # Instead, store M -> count, and M -> list only for counts > 1?
    # Or just store M -> list, and prune if needed?
    # For N=10^7, there are 10^7 entries. A dict with 10^7 entries is ~500MB?
    # Might be too much.
    # Let's store M -> count, and separately keep track of M's with count > 1.
    # Actually, we need the list of k's for the best M.
    # We can store M -> list of k's only for M's that appear > 1 time.
    # For M's that appear 1 time, we just count them.
    
    M_count = {}
    M_ks = {}  # Only for M with count > 1
    
    def update_best(M, k):
        nonlocal best_M, best_ks, best_count
        if M not in M_count:
            M_count[M] = 1
        else:
            M_count[M] += 1
            if M_count[M] > 1:
                if M not in M_ks:
                    M_ks[M] = []
                M_ks[M].append(k)
                if M_count[M] > best_count:
                    best_count = M_count[M]
                    best_M = M
                    best_ks = M_ks[M].copy()
            elif M_count[M] == 1:
                # First occurrence, do nothing special yet
                pass
    
    # We need to initialize M_count and M_ks with the seed values?
    # No, we will recompute everything.
    # But if we start from scratch, we might miss the seed if N is too small.
    # The seed k's are around 400. So N=1000 is enough to find them.
    
    # Let's start with N=1000 to quickly find the seed.
    N = 1000
    while time.time() < deadline:
        # Compute sigma for k=1..N
        # Use a simple sieve for sigma
        # sigma[k] = sum of divisors of k
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
        
        # Now compute M_k and update counts
        # We need to clear M_count and M_ks?
        # No, we are increasing N, so we are adding more k's.
        # But the previous k's are already in M_count.
        # So we only need to process k from previous_N + 1 to N.
        # However, the sigma array is recomputed for all k up to N.
        # This is inefficient.
        
        # Better approach:
        # Maintain a list of k's processed so far.
        # But sigma calculation is the bottleneck.
        # Let's just recompute sigma for all k up to N each time?
        # For N=10^7, the sieve is O(N log N) ~ 10^8 ops.
        # If we do this for N=1000, 10^4, 10^5, ..., it's fine.
        
        # Let's optimize the sieve using NumPy if possible?
        # Pure Python sieve for N=10^7 might take ~10s.
        # Let's try to do it in chunks or just accept the time.
        
        # To avoid recomputing sigma for small k's, we can store them.
        # But memory is a concern.
        # Let's just recompute. It's simpler.
        
        # Actually, we can just run the sieve once for a large N, and then process k's in increasing order.
        # But we don't know how large N can be.
        # Let's try N=10^6 first.
        
        if N == 1000:
            # Process k=1..1000
            for k in range(1, N + 1):
                M = k * sigma[k]
                if M <= 10**18:
                    if M not in M_count:
                        M_count[M] = 1
                    else:
                        M_count[M] += 1
                        if M_count[M] > 1:
                            if M not in M_ks:
                                M_ks[M] = []
                            M_ks[M].append(k)
                            if M_count[M] > best_count:
                                best_count = M_count[M]
                                best_M = M
                                best_ks = M_ks[M].copy()
            N = 10**5
            continue
        
        # For larger N, we need to be careful with time.
        # Let's jump to N=10**6
        if N == 10**5:
            # We need to recompute sigma for 1..10**6
            # This will take some time.
            pass
        
        # Let's structure the loop differently.
        # We will try increasing N values: 1000, 10^5, 10^6, 10^7, ...
        # And break if time runs out.
        
        break # Exit this while loop structure, let's rewrite

    # Rewrite with a more controlled approach
    N_limits = [1000, 10**5, 10**6, 10**7]
    
    # We need to reset M_count and M_ks for each N?
    # No, if we recompute sigma for 1..N, we are adding k's from 1..N.
    # If we do this for N=1000, then N=10^5, we are duplicating work for 1..1000.
    # This is inefficient but correct if we handle counts properly.
    # Actually, if we recompute, we will add the same k's again, doubling the counts.
    # So we MUST either:
    # 1. Clear M_count and M_ks for each N (and lose previous results).
    # 2. Only process new k's (from prev_N+1 to N).
    
    # Option 2 is better.
    # But sigma calculation is for all 1..N.
    # We can compute sigma for 1..N, and then only process k > prev_N.
    
    prev_N = 0
    M_count = {}
    M_ks = {}
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for N in N_limits:
        if time.time() > deadline:
            break
        
        # Compute sigma up to N
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
        
        # Process k from prev_N+1 to N
        for k in range(prev_N + 1, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                if M not in M_count:
                    M_count[M] = 1
                else:
                    M_count[M] += 1
                    if M_count[M] > 1:
                        if M not in M_ks:
                            M_ks[M] = []
                        M_ks[M].append(k)
                        if M_count[M] > best_count:
                            best_count = M_count[M]
                            best_M = M
                            best_ks = M_ks[M].copy()
        
        prev_N = N
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    return {'M': best_M, 'k': best_ks}
