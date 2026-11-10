import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Generate primes up to 1000
    def get_primes(n):
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = False
        return np.array([i for i, is_prime in enumerate(sieve) if is_prime], dtype=np.int64)
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # We want to maximize L such that every x in 1..L is covered by at least 2 congruences.
    # Strategy: Greedy construction + Local Search with efficient scoring.
    # Key insight: To cover x, we need at least 2 primes p_i such that x % p_i == r_i.
    # This is equivalent to: for each x, count(x) >= 2.
    
    # We will work with a window of x values, say 1..MAX_X.
    # MAX_X should be large enough to capture the maximum possible L.
    # Given the problem context, L=1000 is a target. Let's use 2000 to be safe.
    MAX_X = 2000
    x_vals = np.arange(1, MAX_X + 1, dtype=np.int64)
    
    # Initialize residues.
    # A good initial guess: r_p = 0 for all p?
    # Or r_p = x % p for some x?
    # Let's try r_p = 0 first, then improve.
    # Actually, a smarter initialization: 
    # For each x in 1..MAX_X, we want at least 2 primes to cover it.
    # If we set r_p = 0, then p covers 0, p, 2p, ...
    # This leaves gaps.
    
    # Let's use a randomized initialization or a simple heuristic.
    # Heuristic: r_p = 1 for all p? Then 1 is covered by all primes. 2 is covered by odd primes.
    # Let's just start with 0 and let local search do the work.
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Precompute masks for each prime and each possible residue?
    # masks[i, r] is a boolean array of length MAX_X, True where x % p_i == r.
    # Memory: 168 * 1000 * 2000 booleans = 336 million booleans = 336 MB.
    # This is too much.
    
    # Instead, compute mask on the fly or store only for current residue?
    # We need to evaluate score for many candidate residues.
    # Score evaluation:
    # counts = sum(masks[i, residues[i]] for i in range(n_primes))
    # L = first x where counts[x] < 2.
    
    # To speed up local search, we can precompute masks for each prime i for all residues r?
    # For a fixed prime p, the masks for r=0..p-1 are just shifts of each other.
    # mask_r = (x_vals - r) % p == 0.
    # We can compute these on the fly.
    
    # Let's try a different approach:
    # 1. Initialize residues randomly or heuristically.
    # 2. Compute initial counts.
    # 3. Local search:
    #    For each prime i, try all residues r in 0..p_i-1.
    #    For each r, compute new_counts = counts - mask_old + mask_new.
    #    Find new L.
    #    If new L > current L, accept.
    
    # To make this fast, we need efficient mask generation and L finding.
    # L finding: np.argmax(counts < 2) gives the first index where count < 2.
    # If no such index, L = MAX_X.
    
    # Let's optimize the inner loop.
    # For a given prime i, we have p_i possible residues.
    # We can precompute the masks for all residues of prime i?
    # masks_i = np.zeros((p_i, MAX_X), dtype=bool)
    # for r in range(p_i):
    #     masks_i[r] = (x_vals - r) % p_i == 0
    # This takes O(p_i * MAX_X) time and memory.
    # Sum of p_i is ~76k. 76k * 2000 = 152 million booleans = 152 MB.
    # This is borderline but might be okay if we do it for one prime at a time.
    
    # Let's try this.
    
    def compute_L(counts):
        fail = counts < 2
        if not np.any(fail):
            return MAX_X
        first_fail_idx = np.argmax(fail)
        return x_vals[first_fail_idx] - 1
    
    # Initialize counts
    counts = np.zeros(MAX_X, dtype=np.int32)
    for i in range(n_primes):
        p = primes[i]
        r = residues[i]
        counts += ((x_vals - r) % p == 0).astype(np.int32)
        
    current_L = compute_L(counts)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle primes to avoid bias?
        order = np.random.permutation(n_primes)
        for i in order:
            p = primes[i]
            r_old = residues[i]
            
            # Precompute masks for all residues of prime i
            # mask_r = (x_vals - r) % p == 0
            # We can compute this as:
            # x_vals_mod = x_vals % p
            # mask_r = (x_vals_mod == r)
            x_vals_mod = x_vals % p
            
            # Remove old contribution
            old_mask = (x_vals_mod == r_old).astype(np.int32)
            counts -= old_mask
            
            best_r = r_old
            best_L = current_L
            
            # Try all other residues
            # We can vectorize the L computation?
            # For each r, new_counts = counts + (x_vals_mod == r).
            # L_r = compute_L(new_counts).
            # We want max L_r.
            
            # Looping over r is O(p).
            # compute_L is O(MAX_X).
            # Total O(p * MAX_X) per prime.
            # Sum over primes: O(sum(p) * MAX_X) = 76k * 2000 = 152 million ops.
            # This is feasible in NumPy if vectorized, but compute_L is not easily vectorized.
            
            # Let's try to vectorize compute_L.
            # L is the first index where counts < 2.
            # We can create a 2D array of new_counts for all r.
            # new_counts shape: (p, MAX_X)
            # new_counts[r, :] = counts + (x_vals_mod == r)
            
            # masks_all = (x_vals_mod[None, :] == np.arange(p)[:, None])
            # new_counts_all = counts[None, :] + masks_all.astype(np.int32)
            
            # Memory: p * MAX_X integers. Max p=997. 997 * 2000 * 4 bytes = 8 MB.
            # This is fine.
            
            try:
                masks_all = (x_vals_mod[None, :] == np.arange(p)[:, None])
                new_counts_all = counts[None, :] + masks_all.astype(np.int32)
                
                # Compute L for each row
                # fail_all = new_counts_all < 2
                # L_r = x_vals[first_fail_idx] - 1
                # If no fail, L_r = MAX_X
                
                fail_all = new_counts_all < 2
                # For each row, find first True index.
                # If all False, index is MAX_X (or -1, we handle it).
                
                # np.argmax returns first occurrence of max value.
                # We want first occurrence of True in fail_all.
                # If no True, argmax returns 0, which is wrong.
                
                # Let's use a trick:
                # If any(fail_all[r]), then L_r = x_vals[argmax(fail_all[r])] - 1
                # else L_r = MAX_X
                
                # We can do this in a loop over p, which is small (max 997).
                # Or vectorize with np.where?
                
                # Let's just loop over r for simplicity, it's only 1000 iterations.
                for r in range(p):
                    if r == r_old:
                        L_r = current_L
                    else:
                        # We already have new_counts_all[r]
                        # But we can also just use the precomputed fail_all
                        if np.any(fail_all[r]):
                            idx = np.argmax(fail_all[r])
                            L_r = x_vals[idx] - 1
                        else:
                            L_r = MAX_X
                    
                    if L_r > best_L:
                        best_L = L_r
                        best_r = r
                        
            except Exception:
                # Fallback to slow loop if memory error or other issue
                for r in range(p):
                    if r == r_old:
                        continue
                    new_mask = (x_vals_mod == r).astype(np.int32)
                    new_counts = counts + new_mask
                    L_r = compute_L(new_counts)
                    if L_r > best_L:
                        best_L = L_r
                        best_r = r
            
            # Update
            residues[i] = best_r
            new_mask = (x_vals_mod == best_r).astype(np.int32)
            counts += new_mask
            
            if best_L > current_L:
                current_L = best_L
                improved = True
    
    return [int(r) for r in residues]
