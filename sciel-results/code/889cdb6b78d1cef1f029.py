"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Generate primes up to 1000
    primes = _primes_up_to(1000)
    n_primes = len(primes)  # 168
    
    # Precompute for fast coverage counting
    # For each prime p, we need to count how many x in 1..L satisfy x ≡ r (mod p)
    # We'll use a vectorized approach with numpy
    
    def _build_counts(residues, primes, max_L=100000):
        """Build coverage counts for x=1..max_L using vectorized operations."""
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            # x ≡ r (mod p) means x = r + k*p for k >= 0, x >= 1
            # First valid x: if r == 0, then x = p, 2p, ...; else x = r, r+p, ...
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L:
                continue
            # Use slice assignment for speed
            counts[start::p] += 1
        return counts
    
    def _compute_L(residues, primes, max_L=100000):
        """Compute the largest L such that every integer 1..L has count >= 2."""
        counts = _build_counts(residues, primes, max_L)
        # Find the first index where count < 2
        # counts[0] is unused (we care about 1..L)
        below = np.where(counts[1:max_L+1] < 2)[0]
        if len(below) == 0:
            return max_L
        return below[0]  # 0-indexed, so the actual L is below[0]
    
    def _local_search(residues, primes, max_steps=5000, max_L=100000):
        """Vectorized local search: try changing each residue and accept if L improves."""
        residues = np.array(residues, dtype=np.int32)
        current_L = _compute_L(residues, primes, max_L)
        best_L = current_L
        best_residues = residues.copy()
        
        # Precompute prime list as numpy array
        primes_arr = np.array(primes, dtype=np.int64)
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            
            improved = False
            # Try random subset of positions
            n_try = min(n_primes, 50)
            indices = np.random.choice(n_primes, size=n_try, replace=False)
            
            for i in indices:
                p = primes[i]
                old_r = residues[i]
                
                # Try a few random new residues
                for _ in range(3):
                    new_r = np.random.randint(0, p)
                    if new_r == old_r:
                        continue
                    
                    residues[i] = new_r
                    new_L = _compute_L(residues, primes, max_L)
                    
                    if new_L > current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        improved = True
                        break
                    else:
                        residues[i] = old_r
            
            steps += 1
            if not improved and steps % 100 == 0:
                # No improvement in a while, try a bigger kick
                for i in np.random.choice(n_primes, size=10, replace=False):
                    p = primes[i]
                    residues[i] = np.random.randint(0, p)
                current_L = _compute_L(residues, primes, max_L)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
        
        return best_residues, best_L
    
    def _greedy_construct(primes):
        """Greedy: assign residues to maximize coverage of small integers."""
        residues = np.zeros(len(primes), dtype=np.int32)
        max_L = 100000
        counts = np.zeros(max_L + 1, dtype=np.int32)
        
        for i, p in enumerate(primes):
            best_r = 0
            best_score = -1
            # Try all residues? Too slow for large p. Try a subset.
            if p <= 100:
                candidates = range(p)
            else:
                # Focus on residues that cover small numbers
                candidates = list(range(0, min(p, 200)))
                if p > 200:
                    candidates += [np.random.randint(0, p) for _ in range(50)]
            
            for r in candidates:
                # Count how many new x in 1..2000 would get their 2nd coverage
                if r == 0:
                    start = p
                else:
                    start = r
                if start > max_L:
                    continue
                # Count x in start, start+p, ... up to 2000 where counts[x] == 1
                xs = np.arange(start, min(2001, max_L+1), p)
                score = np.sum(counts[xs] == 1)
                # Also penalize if we're creating gaps
                if score > best_score:
                    best_score = score
                    best_r = r
            
            residues[i] = best_r
            # Update counts
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_L:
                counts[start::p] += 1
        
        return residues
    
    # Strategy 1: Greedy construction
    best_residues = None
    best_L = 0
    
    residues_greedy = _greedy_construct(primes)
    L_greedy = _compute_L(residues_greedy, primes)
    best_L = L_greedy
    best_residues = residues_greedy.copy()
    
    # Local search from greedy
    if time.time() < deadline - 10:
        res, L = _local_search(residues_greedy, primes, max_steps=3000)
        if L > best_L:
            best_L = L
            best_residues = res
    
    # Strategy 2: Multiple random restarts with local search
    random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Random initial residues
        residues_rand = np.array([np.random.randint(0, p) for p in primes], dtype=np.int32)
        
        # Quick local search
        if time.time() < deadline - 10:
            res, L = _local_search(residues_rand, primes, max_steps=2000)
            if L > best_L:
                best_L = L
                best_residues = res
    
    # Strategy 3: Hybrid - start from seed pattern, then local search
    if time.time() < deadline - 10:
        seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
        res, L = _local_search(seed_residues, primes, max_steps=2000)
        if L > best_L:
            best_L = L
            best_residues = res
    
    # Final polish: try to extend L by fixing specific gaps
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            current_L = _compute_L(best_residues, primes)
            if current_L >= 100000:
                break
            # Find the first gap
            counts = _build_counts(best_residues, primes, min(current_L + 1000, 100000))
            below = np.where(counts[1:min(current_L + 1000, 100000)+1] < 2)[0]
            if len(below) == 0:
                break
            gap_pos = below[0] + 1  # 1-indexed
            # Try to fix by changing residues of primes that divide near gap_pos
            # This is complex; just do another local search round
            res, L = _local_search(best_residues, primes, max_steps=1000)
            if L > best_L:
                best_L = L
                best_residues = res
    
    # Ensure we're within deadline
    if time.time() > deadline - 1:
        # Return best found
        pass
    
    return [int(r) for r in best_residues]

def _primes_up_to(n):
    """Generate all primes up to n using sieve."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]
# EVOLVE-BLOCK-END
