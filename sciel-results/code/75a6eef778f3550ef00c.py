"""Combined approach: greedy frontier extension with vectorized candidate evaluation,
plateau-walking, and simulated annealing for escape."""
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
    # counts[1:N] corresponds to x=1..N-1
    if N <= 1:
        return 0
    failures = np.nonzero(counts[1:N] < 2)[0]
    if len(failures) == 0:
        return N
    return failures[0] + 1

def _greedy_frontier_extension(residues, primes, N, max_iters=50000):
    """Greedily extend the frontier by finding the most impactful single-residue change."""
    counts = _build_counts(residues, primes, N)
    current_L = _get_L(counts, N)
    n_primes = len(primes)
    
    # Precompute for each prime the positions it covers
    # For efficiency, we'll evaluate candidates for the bottleneck position
    for it in range(max_iters):
        if current_L >= N:
            break
        
        bottleneck = current_L + 1
        if bottleneck >= N:
            break
        
        best_gain = 0
        best_prime_idx = -1
        best_new_r = -1
        
        # For each prime, compute what residue would cover the bottleneck
        # and how much additional coverage it provides
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            
            # The residue that would cover the bottleneck
            r_new = bottleneck % p
            
            if r_new == r_old:
                continue
            
            # Compute the change in coverage
            # Old coverage: positions r_old, r_old+p, r_old+2p, ...
            # New coverage: positions r_new, r_new+p, r_new+2p, ...
            
            # Count how many positions in 1..N are newly covered (were not covered before)
            # and how many are lost
            
            # For the new residue, find positions in 1..N
            if r_new == 0:
                new_positions = np.arange(p, N, p)
            else:
                new_positions = np.arange(r_new, N, p)
            
            # For the old residue, find positions in 1..N
            if r_old == 0:
                old_positions = np.arange(p, N, p)
            else:
                old_positions = np.arange(r_old, N, p)
            
            # Net gain: positions in new_positions not in old_positions that have count < 2
            # minus positions in old_positions not in new_positions that have count == 2
            
            # This is expensive to compute exactly, so use a heuristic:
            # Count positions in new_positions where counts < 2 (potential new coverage)
            if len(new_positions) > 0:
                newly_covered = np.sum(counts[new_positions] < 2)
            else:
                newly_covered = 0
            
            if len(old_positions) > 0:
                lost_coverage = np.sum(counts[old_positions] == 2)
            else:
                lost_coverage = 0
            
            gain = newly_covered - lost_coverage
            
            if gain > best_gain:
                best_gain = gain
                best_prime_idx = i
                best_new_r = r_new
        
        if best_prime_idx == -1 or best_gain <= 0:
            break
        
        # Apply the change
        i = best_prime_idx
        p = primes[i]
        r_old = residues[i]
        r_new = best_new_r
        
        if r_old == 0:
            if p < N:
                counts[p::p] -= 1
        else:
            if r_old < N:
                counts[r_old::p] -= 1
        
        if r_new == 0:
            if p < N:
                counts[p::p] += 1
        else:
            if r_new < N:
                counts[r_new::p] += 1
        
        residues[i] = r_new
        new_L = _get_L(counts, N)
        
        if new_L > current_L:
            current_L = new_L
        else:
            # No improvement in L, but we made a change
            # Check if we should continue (plateau walking)
            pass
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)  # 168
    
    # Search range for L
    N = 30000
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Initialize residues
        if restart == 0:
            # Heuristic initial: assign residues to cover small numbers
            residues = np.zeros(n_primes, dtype=np.int64)
            for i, p in enumerate(primes):
                # Try to cover position i+1
                residues[i] = (i + 1) % p
        elif restart % 3 == 1:
            # Random initialization
            residues = np.array([int(rng.integers(0, p)) for p in primes])
        else:
            # Start from best found so far with some perturbation
            if best_residues is not None:
                residues = best_residues.copy()
                # Perturb a few random residues
                for _ in range(5):
                    i = int(rng.integers(n_primes))
                    p = primes[i]
                    residues[i] = int(rng.integers(0, p))
            else:
                residues = np.array([int(rng.integers(0, p)) for p in primes])
        
        # Build counts
        counts = _build_counts(residues, primes, N)
        current_L = _get_L(counts, N)
        
        if current_L > best_score:
            best_score = current_L
            best_residues = residues.copy()
        
        # Phase 1: Greedy frontier extension
        if time.time() < deadline - 2.0:
            residues, current_L = _greedy_frontier_extension(residues, primes, N, max_iters=10000)
            counts = _build_counts(residues, primes, N)
            current_L = _get_L(counts, N)
            
            if current_L > best_score:
                best_score = current_L
                best_residues = residues.copy()
        
        # Phase 2: Simulated annealing for further improvement
        if time.time() < deadline - 2.0:
            T = 10.0
            T_min = 0.1
            alpha = 0.9995
            
            while time.time() < deadline - 2.0:
                # Pick a random prime
                i = int(rng.integers(n_primes))
                p = primes[i]
                r_old = residues[i]
                
                # Pick a new residue
                if current_L < N and rng.random() < 0.4:
                    # Target the bottleneck
                    bottleneck = current_L + 1
                    if bottleneck < N:
                        r_new = bottleneck % p
                    else:
                        r_new = int(rng.integers(0, p))
                else:
                    r_new = int(rng.integers(0, p))
                
                if r_old == r_new:
                    continue
                
                # Update counts
                if r_old == 0:
                    if p < N:
                        counts[p::p] -= 1
                else:
                    if r_old < N:
                        counts[r_old::p] -= 1
                
                if r_new == 0:
                    if p < N:
                        counts[p::p] += 1
                else:
                    if r_new < N:
                        counts[r_new::p] += 1
                
                new_L = _get_L(counts, N)
                delta = new_L - current_L
                
                # Acceptance
                if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-10)):
                    current_L = new_L
                    residues[i] = r_new
                    if current_L > best_score:
                        best_score = current_L
                        best_residues = residues.copy()
                else:
                    # Revert
                    if r_new == 0:
                        if p < N:
                            counts[p::p] -= 1
                    else:
                        if r_new < N:
                            counts[r_new::p] -= 1
                    
                    if r_old == 0:
                        if p < N:
                            counts[p::p] += 1
                    else:
                        if r_old < N:
                            counts[r_old::p] += 1
                
                # Cool down
                T *= alpha
                if T < T_min:
                    T = T_min
        
        # After each restart, try greedy extension from the best found
        if best_residues is not None and time.time() < deadline - 2.0:
            residues2 = best_residues.copy()
            residues2, L2 = _greedy_frontier_extension(residues2, primes, N, max_iters=5000)
            if L2 > best_score:
                best_score = L2
                best_residues = residues2
    
    return best_residues.tolist()
