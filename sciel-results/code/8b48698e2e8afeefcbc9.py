import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Evaluation function: computes the largest L such that every integer 1..L 
    # is covered by at least 2 congruences.
    MAX_L = 5000
    xs = np.arange(1, MAX_L + 1, dtype=np.int64)
    
    def evaluate(residues):
        # residues: array of shape (n_primes,)
        # Compute coverage for all x in 1..MAX_L
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_L
        return int(np.argmax(mask))

    # Initial solution: Use the seed strategy
    seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    residues = seed_residues.copy()
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # Multi-Residue Batch Coordinate Descent
    # Strategy: 
    # 1. Find the first failure point L.
    # 2. Look at integers in [L, L+window] that are failing (coverage < 2).
    # 3. Identify primes that cover these failing integers.
    # 4. Select a batch of k primes from this set (prioritizing smaller primes for more frequent hits).
    # 5. Perform local search on this batch: try changing residues of these k primes 
    #    to maximize the score. We use a greedy approach: for each prime in the batch,
    #    try all possible residues (if p is small) or a subset, and pick the best one,
    #    repeating for a few passes.
    
    # To make this faster, we limit the batch size and the search depth.
    # We will alternate between:
    # - Single-residue SA (for fine tuning)
    # - Batch optimization (for breaking local optima)
    
    # Let's implement a helper to optimize a specific set of indices
    def optimize_batch(indices, residues_copy, max_iters=10):
        # indices: list of prime indices to optimize
        # residues_copy: current residues
        # We will do a few passes of greedy coordinate descent on these indices
        
        current_res = residues_copy.copy()
        current_val = evaluate(current_res)
        
        for _ in range(max_iters):
            improved = False
            # Shuffle indices to avoid bias
            random.shuffle(indices)
            for idx in indices:
                p = primes[idx]
                old_r = current_res[idx]
                
                # If p is small, try all residues. If large, try a random subset or just random.
                # For p <= 50, try all. Else, try 10 random candidates + old.
                if p <= 50:
                    candidates = range(p)
                else:
                    # Include old and some randoms
                    candidates = [old_r]
                    candidates += [random.randint(0, p-1) for _ in range(10)]
                
                best_r = old_r
                best_local_val = current_val
                
                for r in candidates:
                    if r == old_r:
                        continue
                    current_res[idx] = r
                    val = evaluate(current_res)
                    if val > best_local_val:
                        best_local_val = val
                        best_r = r
                
                if best_r != old_r:
                    current_res[idx] = best_r
                    current_val = best_local_val
                    improved = True
            
            if not improved:
                break
                
        return current_res, current_val

    # Main loop
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    
    # We will use a hybrid approach:
    # - 80% of the time: Single residue SA (fast, fine-grained)
    # - 20% of the time: Batch optimization (slower, coarse-grained, breaks plateaus)
    
    while time.time() < deadline:
        if time.time() > deadline - 5.0:
            break
            
        # Determine mode
        if random.random() < 0.2:
            # Batch Mode
            # Identify bottleneck
            # To save time, we don't re-evaluate fully if we just did it, but let's assume we need to.
            # Actually, we can just pick random primes for batch, or try to pick "useful" ones.
            # Let's pick 5 random primes, but weighted towards smaller primes (they affect more x in 1..L).
            # Smaller primes have higher frequency.
            # Weights: 1/p
            weights = 1.0 / primes_arr
            weights /= weights.sum()
            batch_indices = np.random.choice(n_primes, size=5, replace=False, p=weights)
            batch_indices = batch_indices.tolist()
            
            # Optimize this batch
            new_res, new_val = optimize_batch(batch_indices, residues, max_iters=5)
            
            # Accept if better
            if new_val > current_score:
                residues = new_res
                current_score = new_val
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # With some probability, accept worse to escape local optima (SA-like)
                delta = new_val - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if random.random() < prob:
                        residues = new_res
                        current_score = new_val
        else:
            # Single Residue Mode (SA)
            idx = random.randint(0, n_primes - 1)
            old_r = residues[idx]
            new_r = random.randint(0, primes[idx] - 1)
            if new_r == old_r:
                continue
                
            residues[idx] = new_r
            new_score = evaluate(residues)
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues.copy()
            else:
                if delta == 0:
                    accept_prob = 1.0
                else:
                    try:
                        accept_prob = np.exp(delta / T)
                    except OverflowError:
                        accept_prob = 0.0
                        
                if random.random() < accept_prob:
                    current_score = new_score
                else:
                    residues[idx] = old_r
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_residues.tolist()
