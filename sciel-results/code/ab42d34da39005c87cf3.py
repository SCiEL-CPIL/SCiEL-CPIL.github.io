"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _primes_up_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES = _primes_up_to(1000)
N_PRIMES = len(PRIMES)  # 168

def _compute_L(residues, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    # Build a coverage count array using vectorized approach
    # For each prime p with residue r, mark positions r, r+p, r+2p, ... up to max_check
    # We need count[x] >= 2 for all x in 1..L
    
    # Efficient approach: use numpy
    import numpy as np
    
    counts = np.zeros(max_check + 2, dtype=np.int32)
    for p, r in zip(PRIMES, residues):
        if r == 0:
            r = p  # 0 mod p means p, 2p, ...
        # Mark all positions congruent to r mod p
        # positions: r, r+p, r+2p, ... <= max_check
        if r > max_check:
            continue
        indices = np.arange(r, max_check + 1, p)
        counts[indices] += 1
    
    # Find the first position where count < 2
    # We need positions 1..L all to have count >= 2
    # Find first index i >= 1 where counts[i] < 2
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]  # this is the 0-indexed offset from 1, so L = below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial solution: the seed approach
    # The i-th prime covers ceil((i+1)/2), so residues are chosen so that
    # each integer k is hit by primes at positions 2k-1 and 2k
    # residue for prime at index i (0-based) covering value v = ceil((i+1)/2)
    # v = (i+1+1)//2 = (i+2)//2
    # r_p = v % p
    best_residues = [(i + 2) // 2 % PRIMES[i] for i in range(N_PRIMES)]
    best_L = _compute_L(best_residues)
    
    # Local search: try changing one residue at a time
    # For each position i, try all possible residues for prime PRIMES[i]
    # and see if L improves
    
    import numpy as np
    
    def _quick_L(residues, current_best):
        """Quick check: verify that all positions up to current_best still have count >= 2,
        then try to extend further."""
        # This is a fast check for whether a candidate is at least as good
        # Build counts only up to current_best + some extra
        check_limit = min(100000, current_best + 5000)
        counts = np.zeros(check_limit + 2, dtype=np.int32)
        for p, r in zip(PRIMES, residues):
            if r == 0:
                r = p
            if r > check_limit:
                continue
            indices = np.arange(r, check_limit + 1, p)
            counts[indices] += 1
        
        # Check all positions 1..current_best have count >= 2
        if np.any(counts[1:current_best+1] < 2):
            return -1  # worse than current
        
        # Find how far we can extend
        below_two = np.where(counts[1:check_limit+1] < 2)[0]
        if len(below_two) == 0:
            return check_limit
        return below_two[0]
    
    # Local search with multiple passes
    improved = True
    pass_count = 0
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Shuffle order for diversity
        order = list(range(N_PRIMES))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline:
                break
            
            p = PRIMES[i]
            old_r = best_residues[i]
            
            # Try candidate residues
            # Heuristic: focus on residues that cover small integers
            # The most impactful residues are those that cover positions near the current L boundary
            # Try all residues if p is small, sample if p is large
            if p <= 50:
                candidates = list(range(p))
            else:
                # Sample: include old_r, and residues that cover positions near L
                candidates = set()
                candidates.add(old_r)
                # Residues that would cover positions near best_L
                for target in range(max(1, best_L - 10), min(100000, best_L + 20) + 1):
                    r_candidate = target % p
                    candidates.add(r_candidate)
                # Also add some random ones
                for _ in range(min(20, p)):
                    candidates.add(random.randint(0, p - 1))
                candidates = list(candidates)
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                
                best_residues[i] = new_r
                L_candidate = _quick_L(best_residues, best_L)
                
                if L_candidate > best_L:
                    best_L = L_candidate
                    improved = True
                    break
                else:
                    best_residues[i] = old_r
    
    # Final full computation to get exact L
    final_L = _compute_L(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues
# EVOLVE-BLOCK-END
