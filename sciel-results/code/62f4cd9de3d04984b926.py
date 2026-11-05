"""
Strategy: Local-search over a small, high-impact subset of primes (small primes),
with a fast vectorized single-move evaluation.

Rationale:
- The current program attempts a full 168-variable search but is too slow to do
  many iterations, and its pairwise step is expensive.
- The "score" L is the largest integer such that every x in [1, L] is covered by
  at least two congruences. The bottleneck (smallest count) is often determined
  by the interplay of the smallest primes (2, 3, 5, ...). The large primes (near 1000)
  have a low frequency (1/1000) and are less critical for the initial coverage.
- We focus the optimization on the first ~20-30 primes. We fix the rest to a
  reasonable heuristic (e.g., r_p = 0 or random) and only optimize the small ones.
- We use a very fast vectorized single-move hill climb:
  For a given prime index i, we compute the "base" counts (without prime i).
  Then for all residues r in [0, p_i), we add the contribution of r mod p_i
  to the base counts (vectorized over batches) and find the resulting L.
  This is O(p_i * max_L / batch_size) operations per prime, which is very fast
  for small primes.
- We iterate this local search until no improvement is found or time runs out.
- We also try a few different random initializations for the small primes.
"""
import time
import numpy as np
import random

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

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

def _optimize_small_primes(primes, max_L, n_focus, time_limit, seed=0):
    """
    Optimize the first n_focus primes, keeping the rest fixed (to 0 for simplicity,
    or we could keep them from a good initial state).
    """
    random.seed(seed)
    n_primes = len(primes)
    
    # Initialize all residues to 0 (or random for the focus ones)
    residues = [0] * n_primes
    # For the focus primes, we might want a better start.
    # Let's start with random for the focus primes.
    for i in range(n_focus):
        residues[i] = random.randint(0, primes[i] - 1)
    
    counts = _compute_counts(residues, primes, max_L)
    current_L = _find_L(counts, max_L)
    
    best_residues = residues[:]
    best_L = current_L
    
    # Precompute positions for the focus primes
    # positions[i][r] is the array of positions covered by residues[i]=r
    # We only need this for i < n_focus
    focus_positions = []
    for i in range(n_focus):
        p = primes[i]
        pos_list = []
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_L:
                pos_list.append(np.arange(start, max_L + 1, p, dtype=np.int32))
            else:
                pos_list.append(np.array([], dtype=np.int32))
        focus_positions.append(pos_list)
    
    start_time = time.time()
    
    # Hill climbing on the focus primes
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle the order of focus primes to avoid local minima
        order = list(range(n_focus))
        random.shuffle(order)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
                
            p = primes[i]
            current_r = residues[i]
            
            # Compute base counts (without prime i)
            old_pos = focus_positions[i][current_r]
            base = counts.copy()
            if len(old_pos) > 0:
                base[old_pos] -= 1
            
            # Evaluate all possible residues for prime i
            best_new_r = current_r
            best_new_L = current_L
            
            # Vectorized batch evaluation
            BATCH_SIZE = 50
            candidates = list(range(p))
            
            for batch_start in range(0, len(candidates), BATCH_SIZE):
                if time.time() - start_time > time_limit:
                    break
                batch = candidates[batch_start:batch_start + BATCH_SIZE]
                B = len(batch)
                
                # Create a (B, max_L+1) array
                # This is memory intensive if max_L is large, but 50 * 100k * 4 bytes = 20MB, which is fine.
                batch_arr = np.tile(base, (B, 1))
                
                for j, new_r in enumerate(batch):
                    new_pos = focus_positions[i][new_r]
                    if len(new_pos) > 0:
                        batch_arr[j, new_pos] += 1
                
                # Find L for each candidate
                below_two_mask = batch_arr[:, 1:] < 2
                first_below = np.argmax(below_two_mask, axis=1)
                has_below = below_two_mask.any(axis=1)
                Ls = np.where(has_below, first_below + 1, max_L)
                
                best_idx_in_batch = np.argmax(Ls)
                best_L_in_batch = Ls[best_idx_in_batch]
                
                if best_L_in_batch > best_new_L:
                    best_new_L = best_L_in_batch
                    best_new_r = batch[best_idx_in_batch]
            
            if best_new_r != current_r:
                # Apply the change
                if len(old_pos) > 0:
                    counts[old_pos] -= 1
                new_pos = focus_positions[i][best_new_r]
                if len(new_pos) > 0:
                    counts[new_pos] += 1
                residues[i] = best_new_r
                current_L = best_new_L
                improved = True
                
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # We will focus on the first 20 primes.
    # The first 20 primes are: 2, 3, 5, ..., 71.
    # Their product is huge, but the coverage density is high.
    # The "bottleneck" for L is likely in the first few hundred integers.
    
    n_focus = 20
    
    best_residues = None
    best_L = -1
    
    # Try a few different seeds/initializations
    # We have ~145s. Each run of _optimize_small_primes should be fast (< 10s).
    # So we can do ~10-20 restarts.
    
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        
        residues, L = _optimize_small_primes(primes, max_L, n_focus, time_limit=10, seed=seed)
        
        if L > best_L:
            best_L = L
            best_residues = residues[:]
            
        # If we reached the target, we can stop early
        if best_L >= 1000:
            break
    
    # If we didn't find a good solution, fall back to a simple heuristic
    if best_residues is None:
        best_residues = [0] * n_primes
        best_L = _find_L(_compute_counts(best_residues, primes, max_L), max_L)
        
    return best_residues
