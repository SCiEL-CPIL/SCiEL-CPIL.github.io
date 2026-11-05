"""Optimized solver for the covering problem with incremental evaluation."""
import random
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _count_covered(residues, primes, max_L):
    """
    Count how many integers in 1..max_L are covered at least twice.
    Returns the largest L such that every integer 1..L is covered at least twice.
    """
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def _compute_counts(residues, primes, max_L):
    """Compute coverage counts array for given residues."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def _get_positions(r, p, max_L):
    """Get positions covered by residue r mod p in 1..max_L."""
    if r == 0:
        start = p
    else:
        start = r
    if start > max_L:
        return np.array([], dtype=np.int32)
    return np.arange(start, max_L + 1, p, dtype=np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # Initial solution from the seed
    initial_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_residues = initial_residues[:]
    counts = _compute_counts(best_residues, primes, max_L)
    best_L = _find_L(counts, max_L)
    
    # Precompute position arrays for each prime and all possible residues
    # This is expensive in memory but saves time in the inner loop
    # positions_cache[i][r] = positions covered by residue r mod primes[i]
    # For p up to 1000, this is manageable
    
    # Actually, let's compute on the fly but cache the results
    # positions_cache[i] = {r: positions_array}
    
    # Better approach: for each prime i, precompute positions for all r in 0..p-1
    # Memory: sum of p_i * (max_L / p_i) = 168 * max_L ≈ 16.8M entries * 4 bytes ≈ 67MB
    # That's a lot. Let's be smarter.
    
    # Alternative: for each prime i, positions for residue r are just r, r+p, r+2p, ...
    # We can compute the delta quickly without storing all positions.
    
    # When we change residue from old_r to new_r for prime i:
    # - Remove coverage from old_r positions
    # - Add coverage to new_r positions
    # - The net change is: -1 at old_r positions, +1 at new_r positions
    # - But positions that are in BOTH old_r and new_r sets cancel out
    # - Two arithmetic progressions r1, r1+p, r1+2p, ... and r2, r2+p, r2+2p, ...
    #   intersect at positions that are ≡ r1 (mod p) AND ≡ r2 (mod p)
    #   This only happens if r1 ≡ r2 (mod p), i.e., r1 = r2.
    #   So if old_r != new_r, the two sets are disjoint!
    
    # This is a key insight: for a given prime p, different residues give disjoint sets of positions.
    # So the delta is simply: -1 at old_r positions, +1 at new_r positions.
    
    # Let's verify: if r1 != r2 (mod p), can a number be ≡ r1 (mod p) AND ≡ r2 (mod p)?
    # No, because that would require r1 ≡ r2 (mod p). So yes, disjoint.
    
    # So the update is clean:
    # counts_new = counts - 1 at old_r positions + 1 at new_r positions
    
    # For evaluation, we need to find the new L after this update.
    # We can do this incrementally:
    # 1. Apply the delta to a copy of counts
    # 2. Find the new L
    
    # But copying the entire counts array (100001 elements) for each candidate is expensive.
    # Instead, let's think about how to find the new L more efficiently.
    
    # The new L is the first position where counts_new < 2.
    # counts_new[x] = counts[x] - 1 if x ≡ old_r (mod p) else counts[x] + 1 if x ≡ new_r (mod p) else counts[x]
    
    # Let's precompute for the current counts:
    # - The first position where counts < 2 (this is best_L + 1)
    # - For each prime i and residue r, which positions are affected
    
    # Actually, let's just do a fast evaluation:
    # For a candidate (i, new_r):
    # 1. Compute old_r = best_residues[i]
    # 2. If old_r == new_r, skip
    # 3. Create a modified counts array (or just track the changes)
    # 4. Find the new L
    
    # To make this fast, let's use numpy operations:
    # old_pos = _get_positions(old_r, p, max_L)
    # new_pos = _get_positions(new_r, p, max_L)
    # counts_modified = counts.copy()
    # counts_modified[old_pos] -= 1
    # counts_modified[new_pos] += 1
    # new_L = _find_L(counts_modified, max_L)
    
    # The copy is 100001 * 4 = 400KB, which is fast.
    # The np.where in _find_L is O(max_L) = 100000, which is fast.
    # So each evaluation is ~O(max_L) ≈ 100k operations, very fast.
    
    # With this, we can do many more evaluations per second.
    # Let's estimate: if each eval takes ~0.1ms, we can do ~10000 evals/sec.
    # In 140 seconds, that's ~1.4M evals.
    # For each of 168 primes, we can try ~8000 candidates each.
    
    improved = True
    iteration = 0
    while improved and (time.time() - start_time) < deadline:
        improved = False
        iteration += 1
        for i in range(n_primes):
            if (time.time() - start_time) > deadline - 5:
                break
            p = primes[i]
            current_r = best_residues[i]
            
            # Try all residues for this prime (p <= 1000, so at most 1000 trials)
            best_new_r = current_r
            best_new_L = best_L
            
            for new_r in range(p):
                if new_r == current_r:
                    continue
                if (time.time() - start_time) > deadline - 5:
                    break
                
                # Compute the delta
                old_pos = _get_positions(current_r, p, max_L)
                new_pos = _get_positions(new_r, p, max_L)
                
                # Apply delta to a copy
                counts_mod = counts.copy()
                if len(old_pos) > 0:
                    counts_mod[old_pos] -= 1
                if len(new_pos) > 0:
                    counts_mod[new_pos] += 1
                
                new_L = _find_L(counts_mod, max_L)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
            
            if best_new_r != current_r:
                # Apply the best change
                old_pos = _get_positions(current_r, p, max_L)
                new_pos = _get_positions(best_new_r, p, max_L)
                if len(old_pos) > 0:
                    counts[old_pos] -= 1
                if len(new_pos) > 0:
                    counts[new_pos] += 1
                best_residues[i] = best_new_r
                best_L = best_new_L
                improved = True
    
    return best_residues
