"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # 1. Generate primes <= 1000
    sieve = np.ones(1001, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array([i for i, v in enumerate(sieve) if v], dtype=np.int64)
    n_primes = len(primes)

    # 2. Evaluation function: returns L such that all 1..L have count >= 2
    #    We evaluate on a window of size W. If L < W, we can extend.
    W = 2000  # Initial window. We can increase if we hit the end.
    
    def get_L(residues):
        # residues: array of int64
        # We want to check x in 1..W
        # Count for each x is sum over p of (x == r_p mod p)
        # Vectorized approach:
        # For each prime p_i, the numbers in 1..W that are r_i mod p_i are:
        # start = r_i (if r_i >= 1 else r_i + p_i) ... wait, r_i can be 0.
        # If r_i == 0, the first positive number is p_i.
        # Actually, x % p == r.
        # If r == 0, x is p, 2p, ...
        # If r > 0, x is r, r+p, ...
        
        counts = np.zeros(W + 1, dtype=np.int32)
        
        # Optimization: loop over primes, update counts using numpy slicing
        # This is O(n_primes * W/p) which is roughly W * sum(1/p) ~ W * log(log(1000)) ~ 2000 * 2 = 4000 ops. Very fast.
        
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            
            # Find first x in [1, W] such that x % p == r
            if r == 0:
                first = p
            else:
                first = r
                if first == 0: # r is in [0, p-1], so if r>0, first is r. But x must be >= 1.
                    first = p # This case doesn't happen if r>0. If r=0 handled above.
            
            if first <= W:
                # Update counts[first::p] += 1
                counts[first::p] += 1
        
        # Find largest L such that counts[1:L+1] >= 2
        # We need the first index where count < 2.
        # counts[0] is unused.
        valid = counts[1:] >= 2
        if not any(valid):
            return 0
        
        # Find the first False in valid
        # np.argmax on a boolean array returns the first True.
        # We want the first False.
        # If all are True, L = W.
        if all(valid):
            return W
        
        first_false = np.argmax(~valid)
        return first_false

    # 3. Optimization Loop
    # Strategy: Greedy coordinate ascent with restarts.
    # Start with a reasonable initialization.
    # The seed used r_p = (i+2)//2 % p.
    # Let's try a few initializations:
    # 1. Seed initialization
    # 2. Random initialization
    # 3. "Zero" initialization (r_p = 0)
    
    best_L = 0
    best_residues = None
    
    def optimize(residues, time_limit):
        nonlocal best_L, best_residues
        current_L = get_L(residues)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to improve each residue
            # To speed up, we can shuffle the order of primes or just iterate.
            # Iterating 168 times is fast.
            for i in range(n_primes):
                if time.time() > time_limit:
                    return
                
                old_r = residues[i]
                p = primes[i]
                
                # Try all possible residues? p can be up to 997.
                # Trying all p residues for 168 primes is 168 * 1000 = 168,000 evaluations.
                # Each evaluation is O(W * sum(1/p)) ~ 4000 ops.
                # Total ops per full pass: 168,000 * 4000 = 672,000,000.
                # This is too slow for Python loops.
                
                # We need a faster way to evaluate the effect of changing r_i.
                # When we change r_i from r_old to r_new:
                # The counts for numbers x where x % p == r_old decrease by 1.
                # The counts for numbers x where x % p == r_new increase by 1.
                
                # We can maintain the `counts` array incrementally.
                
                # Let's refactor to maintain counts.
                pass
                
        # This incremental approach is better. Let's rewrite optimize to use incremental counts.
        
    # Let's implement the incremental version directly.
    
    def optimize_incremental(residues_init, time_limit):
        nonlocal best_L, best_residues
        
        residues = residues_init.copy()
        counts = np.zeros(W + 1, dtype=np.int32)
        
        # Initialize counts
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first <= W:
                counts[first::p] += 1
                
        current_L = get_L_from_counts(counts)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Iterate through primes
            # Randomize order to avoid local minima traps?
            order = list(range(n_primes))
            random.shuffle(order)
            
            for i in order:
                if time.time() > time_limit:
                    return
                
                old_r = residues[i]
                p = primes[i]
                
                # Identify indices affected by old_r
                if old_r == 0:
                    old_first = p
                else:
                    old_first = old_r
                
                old_indices = np.arange(old_first, W + 1, p)
                
                # Try to find a better new_r
                # We want to maximize L.
                # Changing r affects counts at old_indices (decrease) and new_indices (increase).
                
                # A simple heuristic: try a few random new residues, or scan?
                # Scanning all p residues is expensive if we recompute L from scratch.
                # But we can update counts, compute L, and rollback.
                # Computing L from counts is O(W).
                # So trying one new_r is O(W + p).
                # Trying all p is O(p*W + p^2).
                # For p=1000, W=2000, that's 2,000,000 ops.
                # 168 primes * 2,000,000 = 336,000,000 ops per pass.
                # This might be borderline in 130s.
                
                # Let's try a subset of candidates first, or just random sampling.
                # Or, better: Only try to change if the current residue is "bad".
                # What is bad? If the prime is not contributing to the "bottleneck".
                
                # Let's try a limited scan: try 10 random candidates + old_r + 0.
                candidates = [old_r]
                if old_r != 0:
                    candidates.append(0)
                
                # Add some randoms
                for _ in range(5):
                    candidates.append(random.randint(0, p - 1))
                
                best_new_r = old_r
                best_new_L = current_L
                best_new_counts = None
                
                # Apply old_r removal for testing? No, we apply new_r on top of current counts?
                # Current counts include old_r.
                # To test new_r:
                # 1. Subtract old_r contribution.
                # 2. Add new_r contribution.
                # 3. Compute L.
                # 4. If better, keep. Else rollback.
                
                # Optimization: Precompute old_indices
                # We need to be careful with counts modification.
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                        
                    if new_r == 0:
                        new_first = p
                    else:
                        new_first = new_r
                        
                    new_indices = np.arange(new_first, W + 1, p)
                    
                    # Modify counts
                    counts[old_indices] -= 1
                    counts[new_indices] += 1
                    
                    new_L = get_L_from_counts(counts)
                    
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_new_r = new_r
                        # Store a copy? Or just remember the indices?
                        # Storing a copy of counts (size 2000) is cheap.
                        best_new_counts = counts.copy()
                        
                    # Rollback
                    counts[old_indices] += 1
                    counts[new_indices] -= 1
                    
                if best_new_r != old_r:
                    # Apply the best change
                    residues[i] = best_new_r
                    counts = best_new_counts
                    current_L = best_new_L
                    improved = True
                    
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                        
                    # If we reached W, we might want to expand W?
                    # For now, assume W=2000 is enough to find L>=1000.
                    # If L hits W, we can break early or increase W.
                    if current_L >= W:
                        # Expand W?
                        # Let's just stop this optimization pass, we'll handle expansion separately.
                        break

    def get_L_from_counts(counts):
        # counts is array of size W+1
        # Find largest L such that counts[1:L+1] >= 2
        valid = counts[1:] >= 2
        if not any(valid):
            return 0
        if all(valid):
            return W
        first_false = np.argmax(~valid)
        return first_false

    # Initializations
    # 1. Seed
    r_seed = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    # 2. Zeros
    r_zero = np.zeros(n_primes, dtype=np.int64)
    
    # 3. Randoms
    r_rand = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    inits = [r_seed, r_zero, r_rand]
    
    # Run optimization on each
    # Allocate time budget
    # Total time 130s.
    # Let's give each init 40s.
    
    for init in inits:
        if time.time() > deadline:
            break
        time_limit = min(time.time() + 40.0, deadline)
        optimize_incremental(init, time_limit)
        
    # If we have time left, do more random restarts
    while time.time() < deadline - 10.0:
        r_new = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
        time_limit = min(time.time() + 10.0, deadline)
        optimize_incremental(r_new, time_limit)
        
    # Ensure we return a valid list
    if best_residues is None:
        return [int(r) for r in r_seed]
        
    return [int(r) for r in best_residues]

# EVOLVE-BLOCK-END
