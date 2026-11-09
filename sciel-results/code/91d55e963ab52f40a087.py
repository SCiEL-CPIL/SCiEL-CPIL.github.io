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
    # We check up to a max limit, say 5000, which is well beyond 1000.
    MAX_L = 5000
    xs = np.arange(1, MAX_L + 1, dtype=np.int64)
    
    def evaluate(residues):
        # residues: array of shape (n_primes,)
        # Compute coverage for all x in 1..MAX_L
        # For each prime p_i and residue r_i, x satisfies x % p_i == r_i
        # We want count[x] >= 2 for all x in 1..L
        
        # Efficient way:
        # For each prime i, the positions covered are r_i, r_i + p_i, r_i + 2p_i, ...
        # We can use a boolean array or integer array.
        # Since MAX_L is small (5000), we can just loop over primes and update coverage.
        # But looping in Python is slow. Let's use NumPy vectorization.
        
        # Actually, for each prime p_i, the mask of covered positions in 1..MAX_L is:
        # (xs - r_i) % p_i == 0
        # But doing this for all 168 primes and 5000 xs creates a 168x5000 boolean array,
        # which is ~840k elements, very fast.
        
        # Let's do it in one go:
        # residues[:, None] - xs[None, :] -> shape (168, 5000)
        # Then mod by primes[:, None]
        # Then check == 0
        
        # Memory: 168 * 5000 * 8 bytes = 6.7 MB. Fine.
        
        # However, doing this from scratch every evaluation might be slightly slow.
        # Let's try it first.
        
        # diff = xs[None, :] - residues[:, None]
        # mods = np.remainder(diff, primes_arr[:, None])
        # hits = (mods == 0)
        # coverage = np.sum(hits, axis=0)
        
        # Find the largest L such that coverage[0:L] >= 2
        # Note: xs starts at 1, so coverage[0] is for x=1.
        
        # We need the first index where coverage < 2.
        # If no such index, L = MAX_L.
        
        # To speed up evaluation, we can stop early if we find a failure.
        # But vectorized sum is fast.
        
        # Let's optimize the evaluation:
        # Instead of full sum, we can check if min(coverage[:L]) >= 2.
        # But we don't know L. We want the *first* failure.
        
        # coverage < 2 gives a boolean array.
        # Find first True.
        
        # Let's implement a fast version.
        
        # Optimization: Pre-compute nothing, just do the math.
        # The bottleneck is the modulo operation on a 168x5000 array.
        
        # Alternative: Loop over primes, but use slicing?
        # For prime p, residue r:
        # indices = np.arange(r, MAX_L+1, p)
        # But r can be 0, and we start from 1.
        # If r == 0, indices are p, 2p, ...
        # If r > 0, indices are r, r+p, ... if r >= 1.
        # This is tricky with vectorization across primes.
        
        # Let's stick to the broadcast approach, it's likely fast enough in NumPy.
        
        # diff = xs[None, :] - residues[:, None]
        # This creates a 168x5000 array.
        # np.remainder(diff, primes_arr[:, None])
        # This is also 168x5000.
        
        # Let's try to make it faster by checking only up to current best L + some margin?
        # No, we need to know if we can extend L.
        
        # Let's just do the full check.
        
        # One more optimization: 
        # We only care about the prefix.
        # If coverage[0] < 2, L=0.
        # If coverage[1] < 2, L=1.
        
        # Let's write the evaluation.
        
        # To save time, we can compute coverage incrementally? No, residues change.
        
        # Let's use the broadcast method.
        
        # Note: xs is 1..MAX_L. residues are 0..p-1.
        # x % p == r  <=>  (x - r) % p == 0.
        
        # Let's try.
        
        # If this is too slow, we can reduce MAX_L or use a different method.
        # 168 * 5000 = 840,000 operations. NumPy can do ~100M ops/sec. So ~10ms per eval.
        # 115s / 10ms = 11,500 evals. That's plenty for SA.
        
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        # Find first index where coverage < 2
        # We want the largest L such that coverage[0:L] >= 2.
        # So L is the index of the first failure, or MAX_L if no failure.
        
        # failures = np.where(coverage < 2)[0]
        # if len(failures) == 0:
        #     return MAX_L
        # else:
        #     return failures[0]
        
        # Vectorized find first:
        # This is tricky in pure NumPy without a loop.
        # But we can do:
        # mask = coverage < 2
        # if not np.any(mask):
        #     return MAX_L
        # else:
        #     return np.argmax(mask)  # First True is at argmax
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_L
        return int(np.argmax(mask))
    
    # Initial solution: random
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    # Better initial solution: Use the seed strategy?
    # Seed: r_i = ceil(i/2) % p.
    # Let's start with seed to have a decent baseline, then improve.
    seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    residues = seed_residues.copy()
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    
    # We will run until deadline.
    # Each eval takes ~10-20ms. So ~5000-10000 iterations possible.
    
    while time.time() < deadline:
        # Pick a random prime index to change
        idx = random.randint(0, n_primes - 1)
        
        # Pick a new random residue for that prime
        old_r = residues[idx]
        new_r = random.randint(0, primes[idx] - 1)
        if new_r == old_r:
            continue
            
        residues[idx] = new_r
        
        new_score = evaluate(residues)
        
        # Accept if better, or with probability exp(-(delta)/T) if worse
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
        else:
            # Delta <= 0
            # Probability of accepting
            # Since scores are integers, delta is integer <= 0.
            # If delta == 0, accept with prob 1 (or high).
            if delta == 0:
                accept_prob = 1.0
            else:
                # exp(delta / T). delta is negative.
                # If T is small, this is very small.
                try:
                    accept_prob = np.exp(delta / T)
                except OverflowError:
                    accept_prob = 0.0
                    
            if random.random() < accept_prob:
                current_score = new_score
            else:
                # Reject: revert
                residues[idx] = old_r
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: restart with higher T if stuck?
            # For now, just keep going.
            
        # Check time frequently
        if time.time() > deadline - 5.0:
            break
            
    return best_residues.tolist()

# To ensure the function is callable and returns the right type
# The construct function above is the main entry point.
