"""Combined approach: Greedy frontier extension with safety checks and simulated annealing for escape."""
import numpy as np
import time
import math
import random

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _build_counts(residues, primes, N):
    """Build coverage counts array for x in 0..N-1."""
    counts = np.zeros(N, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            if p < N:
                counts[p::p] += 1
        else:
            if r < N:
                counts[r::p] += 1
    return counts

def _get_L(counts, N):
    """Find the largest L such that counts[x] >= 2 for all x in 1..L."""
    if N <= 1:
        return 0
    # We need counts[1] >= 2, counts[2] >= 2, etc.
    # Find first index i in 1..N-1 where counts[i] < 2
    # If none, L = N-1 (since counts is 0..N-1, x goes up to N-1)
    # Actually, if counts[N-1] >= 2, we can say L >= N-1.
    # The problem asks for L. If we cover 1..L, score is L.
    # Our array covers 0..N-1. So x=1..N-1.
    # If all 1..N-1 are covered, score is at least N-1.
    # We can extend N if needed, but let's fix N=20000.
    
    # Vectorized check
    # failures = np.nonzero(counts[1:N] < 2)[0]
    # This creates a copy? No, view. But nonzero creates a new array.
    # Better: use argmax on a boolean array?
    # Or just loop? N=20000 is small.
    # Let's use numpy for speed.
    mask = counts[1:N] < 2
    if not np.any(mask):
        return N - 1
    first_fail = np.argmax(mask)
    return first_fail + 1

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    N = 20000
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    
    # We will run multiple restarts
    # Each restart: Random init -> Greedy Extension -> SA Escape -> Greedy Extension
    
    while time.time() < deadline:
        # Initialize
        residues = np.array([int(rng.integers(0, p)) for p in primes])
        counts = _build_counts(residues, primes, N)
        current_L = _get_L(counts, N)
        
        if current_L > best_score:
            best_score = current_L
            best_residues = residues.copy()
            
        # Greedy Phase
        stuck_counter = 0
        max_stuck = 5000 # If we don't improve L for 5000 steps, try SA
        
        while time.time() < deadline:
            if stuck_counter > max_stuck:
                break
                
            # Find best move to extend L
            best_move_L = current_L
            best_move_idx = -1
            best_move_r = -1
            
            # Target is current_L + 1
            x_target = current_L + 1
            
            if x_target >= N:
                break # Reached limit of our array, assume we can go further? 
                      # For now, stop.
            
            # Evaluate all primes
            for i, p in enumerate(primes):
                r_old = residues[i]
                r_new = x_target % p
                
                if r_old == r_new:
                    continue
                
                # Check safety: Does removing r_old break the prefix [1, current_L]?
                # We subtract 1 from indices r_old, r_old+p, ... <= current_L
                # If any of these had count 2, it becomes 1 -> Invalid.
                
                # Indices in [1, current_L] congruent to r_old mod p
                # Start index:
                # If r_old == 0: starts at p, 2p... (since x>=1)
                # If r_old > 0: starts at r_old, r_old+p... (if r_old >= 1)
                
                if r_old == 0:
                    start_idx = p
                else:
                    start_idx = r_old
                    
                if start_idx > current_L:
                    # No indices in prefix affected by subtraction
                    is_safe = True
                else:
                    # Generate indices or use slicing
                    # end_idx is the largest multiple/residue <= current_L
                    # num_steps = (current_L - start_idx) // p + 1
                    # indices = np.arange(start_idx, current_L + 1, p)
                    # But we can just slice counts
                    # Slice from start_idx to current_L with step p
                    # Note: counts is 0-indexed.
                    # We need min(counts[start_idx : current_L+1 : p])
                    # If min <= 2, unsafe.
                    
                    # Optimization: If p is large, the slice is small.
                    # If p is small, the slice is large.
                    # Using numpy min is fast.
                    
                    # Handle case where start_idx might be 0? No, x>=1.
                    # If r_old == 0, start_idx = p.
                    # If r_old > 0, start_idx = r_old.
                    
                    segment = counts[start_idx : current_L + 1 : p]
                    if len(segment) > 0 and np.min(segment) <= 2:
                        is_safe = False
                    else:
                        is_safe = True
                
                if not is_safe:
                    continue
                
                # Estimate new L
                # We add 1 to indices r_new, r_new+p, ...
                # We subtract 1 from indices r_old, r_old+p, ...
                
                # We start checking from x_target
                new_L = current_L
                x = x_target
                # We can limit the check to a reasonable extension, e.g., 1000 steps
                # Or until we hit a failure.
                # Since we only changed one prime, the change in count is at most +/- 1.
                # So if counts[x] >= 3, it's safe regardless of delta.
                # If counts[x] == 2, it's safe if delta >= 0.
                # If counts[x] == 1, it's safe if delta == 1.
                
                # We can break early if we find a failure.
                # To speed up, we can check in chunks?
                # For now, simple loop.
                
                # To avoid infinite loops or long loops if we extend a lot,
                # we can cap the search for new_L extension.
                # But usually extension is small.
                
                # We need to simulate the counts for x, x+1, ...
                # delta(x) = (1 if x % p == r_new else 0) - (1 if x % p == r_old else 0)
                
                # We can iterate x.
                # Optimization: If counts[x] >= 3, delta doesn't matter for validity (>=2).
                # We only care if counts[x] + delta < 2.
                # i.e., counts[x] == 1 and delta == -1 -> fail.
                # counts[x] == 2 and delta == -1 -> fail.
                # counts[x] == 0 and delta == 0 or -1 -> fail.
                
                # So we only need to check x where counts[x] <= 2.
                # We can precompute indices where counts < 3?
                # That might be expensive to update.
                
                # Let's just loop x.
                # Max extension per step is usually small.
                
                while x < N:
                    c = counts[x]
                    if c >= 3:
                        # Safe, increment
                        new_L = x
                        x += 1
                        continue
                    
                    # c is 0, 1, or 2
                    delta = 0
                    if x % p == r_new:
                        delta += 1
                    if x % p == r_old:
                        delta -= 1
                    
                    if c + delta < 2:
                        break
                    
                    new_L = x
                    x += 1
                    
                if new_L > best_move_L:
                    best_move_L = new_L
                    best_move_idx = i
                    best_move_r = r_new
            
            if best_move_idx == -1:
                # No move extends L
                stuck_counter += 1
                # Try a random safe move to shuffle?
                # Or just break and use SA
                # For now, break to SA
                break
            else:
                # Apply move
                i = best_move_idx
                p = primes[i]
                r_old = residues[i]
                r_new = best_move_r
                
                # Update counts
                if r_old == 0:
                    counts[p::p] -= 1
                else:
                    counts[r_old::p] -= 1
                
                if r_new == 0:
                    counts[p::p] += 1
                else:
                    counts[r_new::p] += 1
                    
                residues[i] = r_new
                current_L = best_move_L
                stuck_counter = 0
                
                if current_L > best_score:
                    best_score = current_L
                    best_residues = residues.copy()
        
        # SA Phase to escape local optimum
        if time.time() < deadline:
            T = 10.0
            T_min = 0.1
            alpha = 0.999
            sa_steps = 5000
            for _ in range(sa_steps):
                if time.time() > deadline - 5.0:
                    break
                i = int(rng.integers(n_primes))
                p = primes[i]
                r_old = residues[i]
                r_new = int(rng.integers(0, p))
                
                if r_old == r_new:
                    continue
                
                # Check safety for SA? 
                # SA allows worsening, but we should avoid breaking the prefix completely?
                # If we break the prefix, L drops to 0 or small.
                # It's better to restrict SA to "safe" moves (don't break prefix)
                # or accept any move but revert if L drops too much?
                # Let's restrict to safe moves to maintain L.
                
                # Same safety check as greedy
                if r_old == 0:
                    start_idx = p
                else:
                    start_idx = r_old
                    
                if start_idx <= current_L:
                    segment = counts[start_idx : current_L + 1 : p]
                    if len(segment) > 0 and np.min(segment) <= 2:
                        continue # Unsafe, skip
                
                # Apply
                if r_old == 0:
                    counts[p::p] -= 1
                else:
                    counts[r_old::p] -= 1
                    
                if r_new == 0:
                    counts[p::p] += 1
                else:
                    counts[r_new::p] += 1
                    
                residues[i] = r_new
                
                new_L = _get_L(counts, N)
                
                # Accept if better or with prob exp(-delta/T)
                # Since we restricted to safe moves, new_L >= current_L is not guaranteed?
                # Wait, if we don't break prefix, L should be at least current_L?
                # No, changing r_old might reduce counts in the prefix?
                # We checked min(segment) > 2, so counts in prefix remain >= 2.
                # So prefix [1, current_L] is still covered.
                # So new_L >= current_L.
                # So delta >= 0.
                # So we always accept?
                # Yes, if we only make safe moves, L never decreases.
                # So this is just a random walk on the plateau of L.
                # This helps escape local optima where no single move extends L,
                # but a sequence of safe moves might.
                
                current_L = new_L
                if current_L > best_score:
                    best_score = current_L
                    best_residues = residues.copy()
                
                T *= alpha
                if T < T_min:
                    T = T_min

    return best_residues.tolist()
