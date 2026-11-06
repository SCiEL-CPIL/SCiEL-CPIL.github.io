import time
import numpy as np
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(counts, max_L):
    # counts is 0-indexed, we care about 1..max_L
    # Find first index i in 1..max_L where counts[i] < 2
    # np.where returns indices. If empty, L is max_L.
    # Note: counts[0] is unused/ignored for L calculation as L starts at 1.
    below = np.where(counts[1:max_L + 1] < 2)[0]
    if len(below) == 0:
        return max_L
    # below[0] is the 0-based index in the slice [1:max_L+1]
    # So the actual x is below[0] + 1
    # The L is the largest x such that 1..x are covered.
    # If the first failure is at x_fail, then L = x_fail - 1.
    return int(below[0])

def _build_counts(residues, primes, L_max):
    counts = np.zeros(L_max + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = int(residues[i])
        if r == 0:
            if p <= L_max:
                counts[p::p] += 1
        else:
            if r <= L_max:
                counts[r::p] += 1
    return counts

def _update_counts(counts, p, old_r, new_r, L_max):
    # Subtract old contribution
    if old_r == 0:
        if p <= L_max:
            counts[p::p] -= 1
    else:
        if old_r <= L_max:
            counts[old_r::p] -= 1
    # Add new contribution
    if new_r == 0:
        if p <= L_max:
            counts[p::p] += 1
    else:
        if new_r <= L_max:
            counts[new_r::p] += 1

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    primes = _primes_up_to(1000)
    n_primes = len(primes)
    L_search = 20000  # Search in a larger space to allow for annealing jumps

    # Initialize with a strong pattern
    # The pattern (i+2)//2 % p was good in the seed.
    # Let's use a few initializations and pick the best.
    best_residues = None
    best_L = 0
    best_counts = None
    
    # Deterministic initializations
    inits = [
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: 0,
        lambda i, p: 1,
        lambda i, p: (i * 7 + 3) % p,
    ]
    
    for init_func in inits:
        residues = np.array([init_func(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = _build_counts(residues, primes, L_search)
        L = _compute_L(counts, L_search)
        if L > best_L:
            best_L = L
            best_residues = residues.copy()
            best_counts = counts.copy()

    if best_residues is None:
        best_residues = np.zeros(n_primes, dtype=np.int64)
        best_counts = _build_counts(best_residues, primes, L_search)
        best_L = _compute_L(best_counts, L_search)

    residues = best_residues.copy()
    counts = best_counts.copy()
    current_L = best_L

    # Simulated Annealing Parameters
    # Temperature T controls the probability of accepting worse moves.
    # We want to accept moves that drop L by small amounts early on.
    T_start = 10.0
    T_end = 0.1
    total_steps = 50000 # Rough estimate of steps we can do
    
    step_count = 0
    random.seed(42)

    while time.time() < deadline - 2.0:
        step_count += 1
        # Cooling schedule: Linear or Exponential? Exponential is standard.
        # Let's use a time-based cooling to be robust to variable step speeds.
        elapsed = time.time() - start_time
        remaining = deadline - 2.0 - elapsed
        total_time_budget = deadline - 2.0 - (start_time + 5.0) # Approx budget after init
        if total_time_budget <= 0:
            total_time_budget = 1.0
        progress = min(1.0, elapsed / (deadline - 2.0 - start_time))
        T = T_start * (T_end / T_start) ** progress
        
        # Determine move
        # Strategy: 
        # 1. If we have time, try to fix the first hole (L+1).
        # 2. Find a prime p such that changing r_p to (L+1)%p is "safe" or "acceptable".
        
        target_x = current_L + 1
        
        if target_x > L_search:
            # We reached the end of our search space. 
            # Try to expand? No, we just verify.
            # If we are at L_search, we can try to see if we can go further by increasing L_search?
            # For simplicity, we stop searching and return.
            # Or we can try random moves to see if L increases? Unlikely if L_search is a hard cap in counts array.
            # Actually, if current_L == L_search, we should probably return.
            # But maybe we can extend L_search? 
            # Let's just break and do final verification.
            break

        # Find candidate primes to move
        # We want to pick a prime p and set r_p = target_x % p.
        # We prioritize primes where this move is "safe" (doesn't break prefix 1..current_L).
        
        # Efficient safety check:
        # A move is safe if for all y in 1..current_L where old_r_p == y % p, counts[y] > 2.
        # i.e., no y in 1..current_L covered by old_r_p has counts[y] == 2.
        
        # We can check this for a specific p quickly.
        # But checking all 168 primes every step is expensive.
        # Let's pick a random subset of primes or just one random prime first?
        # No, let's pick a few random primes and check them.
        
        move_accepted = False
        attempts = 0
        max_attempts = 10 # Check up to 10 random primes for a valid move
        
        while attempts < max_attempts and not move_accepted:
            attempts += 1
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = int(residues[i])
            new_r = target_x % p
            
            if old_r == new_r:
                # Already covers target_x. This means target_x is already covered by p.
                # But counts[target_x] < 2. So we need another prime to cover it.
                # Changing p to the same residue does nothing.
                continue
            
            # Check safety
            # We need to check if any y < current_L that is covered by (p, old_r) has counts[y] == 2.
            # The positions covered by (p, old_r) in 1..current_L are:
            # y = old_r + k*p. 
            # If old_r == 0, y = p, 2p, ...
            # If old_r > 0, y = old_r, old_r+p, ... (if old_r >= 1)
            
            safe = True
            if old_r == 0:
                start_y = p
            else:
                start_y = old_r
            
            if start_y <= current_L:
                # Get the slice of counts for these positions
                # positions = np.arange(start_y, current_L + 1, p)
                # if np.any(counts[positions] == 2): safe = False
                
                # Using numpy slicing is faster than arange+indexing
                # counts[start_y:current_L+1:p]
                sub_counts = counts[start_y:current_L+1:p]
                if np.any(sub_counts == 2):
                    safe = False
            
            if safe:
                # Accept safe move immediately (it will improve or maintain L)
                _update_counts(counts, p, old_r, new_r, L_search)
                residues[i] = new_r
                
                new_L = _compute_L(counts, L_search)
                
                # It should be >= current_L. If it is > current_L, great.
                # If it is == current_L, it means we fixed a hole but created another at same L?
                # No, if safe, it doesn't break 1..current_L.
                # And it covers target_x = current_L + 1.
                # So counts[target_x] increases by 1.
                # If counts[target_x] was 1, it becomes 2. Then L might extend.
                # If counts[target_x] was 0, it becomes 1. Then L stays same?
                # Wait, if counts[target_x] was 0, and we add 1, it becomes 1. Still < 2.
                # So L doesn't change. But we made a move that didn't break anything.
                # This is a "neutral" move. We accept it to explore.
                
                if new_L >= current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                        best_counts = counts.copy()
                    move_accepted = True
                else:
                    # Should not happen for a safe move, but revert just in case
                    _update_counts(counts, p, new_r, old_r, L_search)
                    residues[i] = old_r
                    move_accepted = True # Count as an attempt

            else:
                # Unsafe move. It breaks the prefix.
                # Calculate the new L if we were to make this move.
                # We can estimate the cost: the first hole created.
                # To be fast, we might just use a probability based on T.
                # But we need to know the new L to calculate delta E.
                # Doing a full update and L computation for every unsafe candidate is expensive.
                # Let's only do this for the first unsafe candidate we find?
                # Or just use a random acceptance probability based on T?
                
                # Let's try the move and see.
                _update_counts(counts, p, old_r, new_r, L_search)
                residues[i] = new_r
                
                new_L = _compute_L(counts, L_search)
                
                delta_E = current_L - new_L # Positive if worse
                
                if delta_E <= 0:
                    # Actually improved? (Maybe it fixed a hole before current_L+1?)
                    # Unlikely if we targeted current_L+1, but possible if it had side effects.
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                        best_counts = counts.copy()
                    move_accepted = True
                else:
                    # Worse. Accept with prob exp(-delta_E / T)
                    if random.random() < np.exp(-delta_E / T):
                        current_L = new_L
                        # Don't update best if it's worse
                        move_accepted = True
                    else:
                        # Revert
                        _update_counts(counts, p, new_r, old_r, L_search)
                        residues[i] = old_r
                        move_accepted = True # Attempt made, rejected

        if not move_accepted:
            # No move accepted in max_attempts. 
            # Perform a random kick to escape.
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = int(residues[i])
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                _update_counts(counts, p, old_r, new_r, L_search)
                residues[i] = new_r
                new_L = _compute_L(counts, L_search)
                # Accept kick if it's not catastrophically bad? Or just accept?
                # Let's accept if it's within T tolerance or just always accept to explore.
                # Given we are stuck, accept it.
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    best_counts = counts.copy()

    # Final verification with larger L
    # We found a good configuration in L_search space.
    # Let's check how far it goes in a larger space.
    L_verify = 50000
    if time.time() < deadline - 1.0:
        counts_v = _build_counts(best_residues, primes, L_verify)
        final_L = _compute_L(counts_v, L_verify)
        
        if final_L >= L_verify and time.time() < deadline - 0.5:
            L_verify2 = 100000
            counts_v2 = _build_counts(best_residues, primes, L_verify2)
            final_L = _compute_L(counts_v2, L_verify2)
            
    return [int(r) for r in best_residues]
