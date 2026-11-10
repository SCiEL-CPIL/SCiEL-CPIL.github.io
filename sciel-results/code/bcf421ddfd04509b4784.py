import time
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, v in enumerate(sieve) if v]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)  # 168
    max_check = 100000
    
    # Initial assignment: try to cover small integers densely
    # Strategy: for integer k, assign residues from two primes to cover k
    # Greedy initial: for each prime i, set r_p = (i % p) initially, then refine
    
    # Better initial: pair primes to cover consecutive integers
    # Prime i covers integer floor(i/2)+1 via r = (floor(i/2)+1) % p
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        target = (i // 2) + 1
        residues[i] = target % p
    
    def compute_score(res):
        """Return largest L such that all 1..L have >=2 covers."""
        # For each prime p with residue r, it covers numbers r, r+p, r+2p, ...
        # We need to count for each x in 1..max_check how many primes cover it
        # Efficient: for each prime, add 1 to positions r, r+p, r+2p, ...
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for idx, p in enumerate(primes):
            r = int(res[idx])
            if r == 0:
                start = p
            else:
                start = r
            # Add 1 at positions start, start+p, start+2p, ...
            if start <= max_check:
                counts[start::p] += 1
        # Find largest L where all 1..L have counts >= 2
        # Find first position where count < 2
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return max_check
        return int(below[0])  # position is below[0]+1, so L = below[0]
    
    best_score = compute_score(residues)
    best_res = residues.copy()
    
    # Local search: try changing one residue at a time
    def try_improve(res, score, deadline):
        """Hill climbing: try single-residue changes."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(n_primes):
                if time.time() >= deadline:
                    break
                p = primes[idx]
                old_r = int(res[idx])
                # Try a few candidate residues
                # Try residues that cover the current bottleneck
                # First, find the current score's bottleneck
                # Try old_r +/- small offsets, and also 0
                candidates = set()
                candidates.add(0)
                candidates.add(old_r)
                # Try to cover the next uncovered position
                # But computing that each time is expensive; just try random-ish
                for offset in [-2, -1, 1, 2, 3, 5, 7, 10, 15, 20]:
                    candidates.add((old_r + offset) % p)
                # Also try to cover small integers
                for target in range(1, min(50, p)):
                    candidates.add(target % p)
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    res[idx] = new_r
                    new_score = compute_score(res)
                    if new_score > score:
                        score = new_score
                        improved = True
                        break
                    res[idx] = old_r
        return res, score
    
    best_res, best_score = try_improve(best_res, best_score, deadline)
    
    # Second pass: more systematic local search
    # For each prime, try ALL residues if time permits
    if time.time() < deadline - 5:
        for idx in range(n_primes):
            if time.time() >= deadline - 2:
                break
            p = primes[idx]
            old_r = int(best_res[idx])
            best_local_r = old_r
            best_local_score = best_score
            # Sample residues rather than all (p can be up to 997)
            if p <= 100:
                test_residues = range(p)
            else:
                # Sample
                import random
                random.seed(42 + idx)
                test_residues = random.sample(range(p), min(50, p))
                test_residues = list(test_residues)
                if old_r not in test_residues:
                    test_residues.append(old_r)
            
            for new_r in test_residues:
                if time.time() >= deadline - 2:
                    break
                if new_r == old_r:
                    continue
                best_res[idx] = new_r
                s = compute_score(best_res)
                if s > best_local_score:
                    best_local_score = s
                    best_local_r = new_r
            best_res[idx] = best_local_r
            if best_local_score > best_score:
                best_score = best_local_score
    
    # Convert to list of ints
    result = [int(r) for r in best_res]
    return result
