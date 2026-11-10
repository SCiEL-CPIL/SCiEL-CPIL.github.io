"""Optimized local search with fast NumPy evaluation."""
import time
import numpy as np

# --- primes up to 1000 (168 primes) ---
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return np.array([i for i, v in enumerate(sieve) if v], dtype=np.int64)

PRIMES = _primes_upto(1000)
NPRIMES = len(PRIMES)

# Precompute prime indices for faster access? No, PRIMES is small.

def evaluate(res, max_L=3000):
    """
    res: array of 168 residues.
    Returns the largest L such that every integer 1..L is covered by at least 2 congruences.
    """
    L = max_L
    x = np.arange(1, L + 1, dtype=np.int64)
    counts = np.zeros(L, dtype=np.int16)
    
    # Vectorize the update? 
    # For each prime p and residue r:
    # indices i in 0..L-1 such that (i+1) % p == r
    # i+1 = k*p + r => i = k*p + r - 1
    # i >= 0 => k*p >= 1-r. Since r < p, 1-r <= p-1.
    # Smallest i:
    # if r == 1: i=0
    # if r > 1: i = p - (r-1) = p - r + 1? 
    # Let's check: x = i+1. x ≡ r (mod p).
    # x = r, r+p, ...
    # i = x-1 = r-1, r-1+p, ...
    # If r-1 < 0? No, r>=0. If r=0, x=p, 2p... i=p-1, 2p-1...
    # If r=1, x=1, 1+p... i=0, p...
    # So i_start = (r - 1) % p.
    # If i_start < L:
    #   counts[i_start::p] += 1
    
    for r, p in zip(res, PRIMES):
        i_start = (r - 1) % p
        if i_start < L:
            counts[i_start::p] += 1
            
    # Find first index where counts < 2
    bad = np.where(counts < 2)[0]
    if len(bad) == 0:
        return L
    first_fail_idx = bad[0]
    # The integer value is first_fail_idx + 1
    # The score is the largest L such that 1..L are covered.
    # So if index k is the first bad one (0-based), then integer k+1 is bad.
    # Integers 1..k are good.
    # So score is k.
    return first_fail_idx

def construct():
    deadline = time.time() + 110  # 10s safety margin
    rng = np.random.default_rng(42)
    
    # Initial guess: random
    res = rng.integers(0, PRIMES, size=NPRIMES)
    best_res = res.copy()
    best_L = evaluate(best_res)
    
    # Local search
    # Strategy: Find the first failing integer x. 
    # Try to change one prime's residue to cover x.
    # If it improves L, accept.
    
    # To speed up, we can limit the search to primes p > x/2 ? 
    # If p > x, changing r_p to x%p covers x and doesn't affect any other y < x (since next is x+p > x).
    # If x/2 < p <= x, changing r_p affects x and x-p.
    # If p <= x/2, it affects more.
    # Focusing on large primes is efficient for fixing specific bottlenecks.
    
    max_eval_L = 3000
    
    while time.time() < deadline:
        L_cur = evaluate(res)
        if L_cur > best_L:
            best_L = L_cur
            best_res = res.copy()
            if best_L >= 1000:
                # We achieved the goal, but maybe we can go higher?
                # The task says "Maximize". But 1000 is the target.
                # Let's keep searching a bit more if we have time.
                pass
        
        if L_cur >= max_eval_L:
             # If we cover the whole evaluation range, we can't distinguish further with this L.
             # We could increase max_eval_L, but for now, just break or restart?
             # Let's just break if we hit the cap, as we likely have a good solution.
             # Or we can just return.
             break
        
        # Find first fail
        # evaluate returns the score (last good integer). 
        # So first_fail = L_cur + 1.
        x = L_cur + 1
        
        # Candidate primes to mutate
        # Prioritize primes p > x (no side effects on 1..x-1)
        # Then p > x/2
        # Then others
        
        # Get indices of primes
        # Instead of sorting every time, we can pre-sort primes by value.
        # But PRIMES is already sorted.
        
        # Find indices where PRIMES > x
        mask_large = PRIMES > x
        mask_med = (PRIMES > x // 2) & (PRIMES <= x)
        mask_small = PRIMES <= x // 2
        
        candidates = []
        if np.any(mask_large):
            candidates.append(np.where(mask_large)[0])
        if np.any(mask_med):
            candidates.append(np.where(mask_med)[0])
        if np.any(mask_small):
            candidates.append(np.where(mask_small)[0])
            
        if not candidates:
            # Should not happen as there are always primes > x for x < 1000?
            # If x is large, maybe not. But x starts small.
            candidates.append(np.arange(NPRIMES))
            
        # Try a few mutations
        # Pick a random candidate group, then a random prime from it
        # Bias towards large primes
        # Weights: large 0.5, med 0.3, small 0.2
        # But number of elements varies.
        
        # Simple strategy: 
        # 1. Try one random large prime (if exists)
        # 2. Try one random med prime (if exists)
        # 3. Try one random small prime (if exists)
        
        attempts = 0
        for group in candidates:
            if len(group) == 0:
                continue
            i = group[rng.integers(0, len(group))]
            new_res = res.copy()
            new_res[i] = x % PRIMES[i]
            
            L_new = evaluate(new_res)
            if L_new > L_cur:
                res = new_res
                attempts += 1
                # If we improved, we can re-evaluate L_cur for the next step
                # But we already have L_new.
                # We break out of the inner loop to re-calculate x based on new L?
                # Or just continue trying to improve further?
                # Let's restart the inner loop with the new L to fix the new bottleneck.
                break
                
        if attempts == 0:
            # No improvement from targeted mutations.
            # Try a random perturbation to escape local optima
            if rng.random() < 0.1:
                i = rng.integers(0, NPRIMES)
                new_res = res.copy()
                new_res[i] = rng.integers(0, PRIMES[i])
                L_new = evaluate(new_res)
                if L_new > L_cur:
                    res = new_res
                else:
                    # Simulated annealing style? Or just reject?
                    # Reject to keep it simple and fast.
                    pass

    return best_res.tolist()
