import numpy as np
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes, dtype=np.int64)

    # Dynamic window strategy
    # Start with a moderate window. Expand if we cover it.
    # The inspiration suggests that expanding allows escaping local minima
    # and pushing the frontier further.
    N_current = 5000
    N_max = 100000
    N_expansion_factor = 2
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    random.seed(42)

    # Helper: Build counts array for a given residues and window size N
    # counts[x] = number of primes p such that x % p == r_p
    def build_counts(residues, N):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts

    # Helper: Get score (L) from counts
    def get_score(counts, N):
        # Find first index >= 1 where counts < 2
        # If no failure, return N
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            return N
        return int(np.argmax(failures_mask)) + 1

    # Initial Strategies
    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
    
    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)
        
    def init_one():
        return np.ones(n_primes, dtype=np.int64)
        
    def init_half():
        return np.array([p // 2 for p in primes], dtype=np.int64)
        
    initial_strategies = [
        init_random,
        init_zero,
        init_one,
        init_half,
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
    ]
    
    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break
            
        residues = strat()
        N_cur = N_current
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Local Search with Dynamic Expansion
        stall_count = 0
        max_stall = 200
        
        while time.time() < deadline - 5.0:
            # If we covered the current window, try to expand
            if current_score == N_cur:
                # Expand window
                if N_cur * N_expansion_factor > N_max:
                    break # Reached max size, can't expand further
                    
                N_cur = min(N_cur * N_expansion_factor, N_max)
                # Rebuild counts for the new range
                counts = build_counts(residues, N_cur)
                current_score = get_score(counts, N_cur)
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                
                # If we still cover the new window, loop will expand again
                continue
            
            # Find the first failure
            x_fail = current_score
            gap_idx = current_score - 1  # index in counts array (1-based x -> 0-based index x-1)
            # Wait, counts is 1-based? No, counts is 0..N.
            # x_fail is the value of x.
            # counts[x_fail] < 2.
            
            # Move Selection Strategy:
            # We want to increase counts[x_fail] to >= 2.
            # Current count is C = counts[x_fail].
            # We need to add (2 - C) matches.
            # If C == 1, we need 1 match.
            # If C == 0, we need 2 matches.
            
            # Candidate primes: those p where x_fail % p != residues[i]
            # Changing residues[i] to x_fail % p will add 1 to counts[x_fail].
            
            # To be efficient, we pick a random prime that does NOT currently cover x_fail.
            # If we pick one that DOES cover, we are wasting the move (or breaking it).
            
            # Optimization: Precompute which primes cover x_fail?
            # x_fail is fixed for this iteration.
            # cover_mask = (x_fail % primes_np == residues)
            # candidates = np.where(~cover_mask)[0]
            
            cover_mask = (x_fail % primes_np == residues)
            candidates = np.where(~cover_mask)[0]
            
            if len(candidates) == 0:
                # All primes cover x_fail. But count < 2?
                # Impossible since n_primes = 168.
                break
            
            # Pick a candidate
            # Heuristic: Pick a prime that covers many other numbers in the vicinity?
            # For now, random is good for exploration, but we can bias towards
            # primes that are "small" (cover more numbers) or "large" (less interference).
            # Let's stick to random selection from candidates for robustness.
            
            if rng.random() < 0.7:
                # Pick a candidate that doesn't cover x_fail
                i = int(candidates[rng.integers(len(candidates))])
            else:
                # Random prime (might cover x_fail, acting as perturbation)
                i = int(rng.integers(n_primes))
            
            p = primes[i]
            r_old = int(residues[i])
            
            # Determine new residue
            # If we picked a candidate (doesn't cover x_fail), set r_new = x_fail % p
            # This guarantees adding a match at x_fail.
            # If we picked a random one that covers x_fail, changing it might reduce count at x_fail.
            # So for the "perturbation" case, let's just pick a random residue.
            
            if not cover_mask[i]:
                r_new = x_fail % p
            else:
                r_new = int(rng.integers(0, p))
                if r_new == r_old:
                    continue
            
            # Apply move to counts
            # Subtract old contribution
            if r_old == 0:
                if p <= N_cur:
                    counts[p::p] -= 1
            else:
                if r_old <= N_cur:
                    counts[r_old::p] -= 1
            
            # Add new contribution
            if r_new == 0:
                if p <= N_cur:
                    counts[p::p] += 1
            else:
                if r_new <= N_cur:
                    counts[r_new::p] += 1
            
            residues[i] = r_new
            
            # Check new score
            new_score = get_score(counts, N_cur)
            
            # Acceptance Criteria
            if new_score > current_score:
                current_score = new_score
                stall_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            elif new_score == current_score:
                # Accept with some probability to escape plateaus
                if rng.random() < 0.3:
                    stall_count = 0
                else:
                    stall_count += 1
                    # Revert?
                    # Reverting is safer to maintain quality, but might get stuck.
                    # Let's revert if we've stalled too much.
                    if stall_count > max_stall:
                        # Revert
                        if r_new == 0:
                            if p <= N_cur:
                                counts[p::p] -= 1
                        else:
                            if r_new <= N_cur:
                                counts[r_new::p] -= 1
                            
                        if r_old == 0:
                            if p <= N_cur:
                                counts[p::p] += 1
                        else:
                            if r_old <= N_cur:
                                counts[r_old::p] += 1
                            
                        residues[i] = r_old
                        stall_count = 0
            else:
                # Score dropped. Revert.
                if r_new == 0:
                    if p <= N_cur:
                        counts[p::p] -= 1
                else:
                    if r_new <= N_cur:
                        counts[r_new::p] -= 1
                
                if r_old == 0:
                    if p <= N_cur:
                        counts[p::p] += 1
                else:
                    if r_old <= N_cur:
                        counts[r_old::p] += 1
                
                residues[i] = r_old
                stall_count += 1
                
                if stall_count > max_stall:
                    # Random restart or large perturbation
                    # Let's try a random restart of the residues
                    residues = init_random()
                    counts = build_counts(residues, N_cur)
                    current_score = get_score(counts, N_cur)
                    stall_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()

    # Ensure best_residues is not None
    if best_residues is None:
        best_residues = init_zero()
        
    return [int(r) for r in best_residues]

if __name__ == "__main__":
    res = construct()
    print(len(res))

