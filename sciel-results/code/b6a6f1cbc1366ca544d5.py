"""Rewritten program for covering interval with at least 2 congruences."""
import time
import random
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock limit


def _primes_up_to(n):
    """Sieve of Eratosthenes for primes up to n."""
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def _compute_score(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has count >= 2."""
    n = len(primes)
    # For each integer x in 1..cap, count how many congruences it satisfies
    # Efficient approach: for each prime p with residue r, mark x = r mod p
    # Use an array to accumulate counts
    counts = np.zeros(cap + 2, dtype=np.int32)
    for p, r in zip(primes, residues):
        # x ≡ r (mod p) means x = r, r+p, r+2p, ...
        # But we only care about x in 1..cap
        start = r if r >= 1 else r + p  # first x >= 1
        if start <= 0:
            start = 1
        # Actually, x ≡ r (mod p): the values are r, r+p, r+2p, ...
        # If r == 0, then x = p, 2p, 3p, ...
        if r == 0:
            start = p
        else:
            start = r
        # Count all values in 1..cap
        if start <= cap:
            # Use slicing to add 1 to all positions start, start+p, start+2p, ...
            idx = np.arange(start - 1, cap, p)  # 0-indexed positions
            counts[idx] += 1
    
    # Find the largest L such that counts[1..L] all >= 2
    # counts is 0-indexed, so counts[x] is the count for integer x+1
    # Actually, let me re-think: I'll use 1-indexed thinking
    # Let me redo this more carefully
    
    # For integer x (1-indexed), count how many primes p have x ≡ r_p (mod p)
    # x ≡ r (mod p) means x = r + kp for some integer k >= 0 (if r > 0) or x = kp (if r == 0)
    
    # Let's use a different approach: for each (p, r), the set of x in [1, cap] satisfying x ≡ r (mod p)
    # We accumulate these counts.
    
    # Reset and redo
    counts = np.zeros(cap + 1, dtype=np.int32)  # counts[x] for x = 0..cap, we use 1..cap
    for p, r in zip(primes, residues):
        if r == 0:
            # x = p, 2p, 3p, ...
            xs = np.arange(p, cap + 1, p)
        else:
            # x = r, r+p, r+2p, ...
            if r > cap:
                continue
            xs = np.arange(r, cap + 1, p)
        if len(xs) > 0:
            counts[xs] += 1
    
    # Find largest L such that counts[1..L] all >= 2
    # Use cumulative product or find first position where count < 2
    below_two = np.where(counts[1:cap+1] < 2)[0]
    if len(below_two) == 0:
        return cap
    return below_two[0]  # This is 0-indexed relative to [1..cap], so the first failing x is below_two[0] + 1, meaning L = below_two[0]


def construct():
    start_time = time.time()
    primes = _primes_up_to(1000)
    n = len(primes)  # 168
    
    # Initial seed: the given construction
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    best_score = _compute_score(residues, primes)
    best_residues = residues[:]
    
    if time.time() - start_time > _DEADLINE - 10:
        return best_residues
    
    # Local search: try changing one residue at a time
    random.seed(42)
    primes_arr = np.array(primes)
    
    # Precompute for speed: for each prime p and residue r, which positions in 1..cap are covered
    # This is expensive, so we'll do incremental updates
    
    def _evaluate_single_change(residues, p_idx, new_r, primes):
        """Evaluate the score if we change residues[p_idx] to new_r."""
        old_r = residues[p_idx]
        if old_r == new_r:
            return _compute_score(residues, primes)
        p = primes[p_idx]
        
        # Recompute counts with the change
        counts = np.zeros(100001, dtype=np.int32)
        for i, (pp, rr) in enumerate(zip(primes, residues)):
            if i == p_idx:
                rr = new_r
            if rr == 0:
                xs = np.arange(pp, 100001, pp)
            else:
                if rr > 100000:
                    continue
                xs = np.arange(rr, 100001, pp)
            if len(xs) > 0:
                counts[xs] += 1
        
        below_two = np.where(counts[1:100001] < 2)[0]
        if len(below_two) == 0:
            return 100000
        return below_two[0]
    
    # Hill climbing with random restarts
    improvement_found = True
    max_iters = 5000
    iters = 0
    
    while improvement_found and time.time() - start_time < _DEADLINE - 5 and iters < max_iters:
        improvement_found = False
        iters += 1
        
        # Try random single-residue changes
        tried = set()
        for _ in range(200):
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            p_idx = random.randrange(n)
            if p_idx in tried:
                continue
            tried.add(p_idx)
            
            p = primes[p_idx]
            new_r = random.randrange(p)
            
            new_score = _evaluate_single_change(residues, p_idx, new_r, primes)
            
            if new_score > best_score:
                best_score = new_score
                best_residues = residues[:]
                residues[p_idx] = new_r
                improvement_found = True
                
                # Try to improve further with the new base
                for _ in range(50):
                    if time.time() - start_time > _DEADLINE - 5:
                        break
                    p_idx2 = random.randrange(n)
                    p2 = primes[p_idx2]
                    new_r2 = random.randrange(p2)
                    new_score2 = _evaluate_single_change(residues, p_idx2, new_r2, primes)
                    if new_score2 > best_score:
                        best_score = new_score2
                        best_residues = residues[:]
                        residues[p_idx2] = new_r2
                    else:
                        break
            
            if best_score >= 1000:
                break
    
    return best_residues


if __name__ == "__main__":
    result = construct()
    print(f"Got {len(result)} residues")
