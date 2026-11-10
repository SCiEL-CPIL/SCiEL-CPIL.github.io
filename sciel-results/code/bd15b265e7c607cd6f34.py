import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _build_count(residues, primes, M):
    """Build count array: counts[x] = number of congruences satisfied by x."""
    counts = np.zeros(M + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            counts[p:M+1:p] += 1
        else:
            counts[r:M+1:p] += 1
    return counts

def _find_prefix_L(counts, M):
    """Find largest L such that counts[x] >= 2 for all x in 1..L."""
    if M <= 0:
        return 0
    # Find first index in 1..M where count < 2
    uncovered = np.where(counts[1:M+1] < 2)[0]
    if len(uncovered) > 0:
        return int(uncovered[0])
    return M

def _greedy_init(primes, M):
    """Greedy initialization: for each prime, pick residue that maximizes coverage of early numbers."""
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(M + 1, dtype=np.int32)
    
    # Process primes in order, choosing residue to cover as many uncovered early numbers as possible
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        # Try all residues, score by how many numbers in 1..min(p, M) get covered
        # that currently have count < 2
        for r in range(p):
            # Count how many numbers x in [r, r+p, r+2p, ...] <= M have counts[x] < 2
            # and x >= 1
            start = r if r >= 1 else p
            if start > M:
                score = 0
            else:
                # Vectorized: check counts at positions start, start+p, ...
                idx = np.arange(start, M + 1, p)
                score = np.sum(counts[idx] < 2)
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        # Update counts
        r = best_r
        if r == 0:
            counts[p:M+1:p] += 1
        else:
            counts[r:M+1:p] += 1
    
    return residues, counts

def _local_search(residues, primes, counts, M, deadline, max_iters=50000):
    """Local search: repeatedly try to improve by changing one residue.
    Focus on fixing the first uncovered position."""
    n_primes = len(primes)
    best_L = _find_prefix_L(counts, M)
    best_residues = residues.copy()
    best_counts = counts.copy()
    
    iters = 0
    no_improve = 0
    
    while iters < max_iters and time.time() < deadline - 5:
        iters += 1
        
        # Find current first uncovered
        L = _find_prefix_L(counts, M)
        
        if L >= M:
            break
        
        # Target: fix position L+1 (first uncovered)
        target = L + 1
        
        # For each prime, find if changing its residue could help cover target
        # while not breaking any previously covered position
        best_move = None
        best_gain = 0
        
        # Sample a subset of primes for speed
        sample_size = min(n_primes, 50)
        sample_idx = random.sample(range(n_primes), sample_size)
        
        for idx in sample_idx:
            p = primes[idx]
            old_r = residues[idx]
            
            # Check if target is in the residue class of this prime
            # target mod p gives the residue that would cover target
            target_r = target % p
            
            if target_r == old_r:
                # Already covers target, skip
                continue
            
            # Try changing to target_r
            new_r = target_r
            
            # Check if this change is safe: does it break any position in 1..L?
            # Positions covered by old_r: old_r, old_r+p, old_r+2p, ...
            # Positions covered by new_r: new_r, new_r+p, new_r+2p, ...
            
            # Find positions that lose coverage (were covered by old_r but not new_r)
            # and positions that gain coverage (covered by new_r but not old_r)
            
            # For safety, we need: for all x in 1..L, if x was covered by this prime
            # (i.e., x % p == old_r), then after change x must still be covered by at least
            # one other prime OR x % p == new_r (still covered by this prime)
            
            # Actually simpler: simulate the change and check if any position in 1..L drops below 2
            
            # Apply change temporarily
            if old_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[old_r:M+1:p] -= 1
            
            if new_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[new_r:M+1:p] += 1
            
            # Check if any position in 1..L now has count < 2
            if L > 0:
                min_count = np.min(counts[1:L+1])
                if min_count < 2:
                    # Revert
                    if new_r == 0:
                        counts[p:M+1:p] -= 1
                    else:
                        counts[new_r:M+1:p] -= 1
                    if old_r == 0:
                        counts[p:M+1:p] += 1
                    else:
                        counts[old_r:M+1:p] += 1
                    continue
            
            # Safe move! Compute new L
            new_L = _find_prefix_L(counts, M)
            gain = new_L - L
            
            if gain > best_gain:
                best_gain = gain
                best_move = (idx, old_r, new_r, new_L)
            
            # Revert for next trial
            if new_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[new_r:M+1:p] -= 1
            if old_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[old_r:M+1:p] += 1
        
        if best_move is not None and best_gain > 0:
            idx, old_r, new_r, new_L = best_move
            p = primes[idx]
            
            # Apply the move
            if old_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[old_r:M+1:p] -= 1
            
            if new_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[new_r:M+1:p] += 1
            
            residues[idx] = new_r
            no_improve = 0
            
            # Update best if improved
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                best_counts = counts.copy()
                
                # Expand M if needed
                if best_L >= M:
                    old_M = M
                    M = min(M * 2, 100000)
                    # Rebuild counts for new M
                    counts = _build_count(residues, primes, M)
                    best_counts = counts.copy()
                    best_L = _find_prefix_L(counts, M)
        else:
            no_improve += 1
            if no_improve > 100:
                # Restart from best
                residues = best_residues.copy()
                counts = best_counts.copy()
                no_improve = 0
    
    return best_residues, best_L

def construct():
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial M
    M = 2000
    
    # Greedy initialization
    residues, counts = _greedy_init(primes, M)
    L = _find_prefix_L(counts, M)
    
    best_L = L
    best_residues = residues.copy()
    
    # Local search
    best_residues, best_L = _local_search(residues, primes, counts, M, deadline)
    
    # If we have time, try more iterations with different random seeds
    while time.time() < deadline - 10:
        # Restart with perturbation
        residues = best_residues.copy()
        # Perturb a few residues
        for _ in range(3):
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            residues[idx] = random.randint(0, p - 1)
        
        counts = _build_count(residues, primes, M)
        L = _find_prefix_L(counts, M)
        
        # Local search from here
        res, L2 = _local_search(residues, primes, counts, M, deadline)
        if L2 > best_L:
            best_L = L2
            best_residues = res
    
    # Ensure residues are Python integers
    best_residues = [int(r) for r in best_residues]
    
    return best_residues
# EVOLVE-BLOCK-END
