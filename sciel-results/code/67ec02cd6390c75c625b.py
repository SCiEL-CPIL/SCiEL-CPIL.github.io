"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _get_primes(n=1000):
    """Return list of primes <= n in increasing order."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_L(counts, max_check=100000):
    """Find largest L such that every integer 1..L has count >= 2."""
    bad_mask = counts[1:max_check+1] < 2
    if not np.any(bad_mask):
        return max_check
    first_bad = np.argmax(bad_mask)
    return (first_bad + 1) - 1

def construct():
    primes = _get_primes(1000)
    n_primes = len(primes)  # 168
    max_check = 100000
    
    # Initialize residues randomly
    residues = np.array([np.random.randint(p) for p in primes], dtype=np.int64)
    
    # Build initial counts
    counts = np.zeros(max_check + 2, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = p if r == 0 else r
        counts[start:max_check+1:p] += 1
    
    current_L = _compute_L(counts, max_check)
    
    # Hybrid Search: Targeted Moves + Simulated Annealing
    # We will alternate between trying to fix the current bottleneck and random exploration
    
    best_L = current_L
    best_residues = residues.copy()
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    steps = 50000
    
    for step in range(steps):
        T *= cooling
        if T < T_min:
            break
            
        # Identify current bottleneck K = L + 1
        # If L == max_check, we are done (or at max)
        if current_L >= max_check:
            break
            
        K = current_L + 1
        
        # Strategy 1: Targeted Move
        # Find primes that do NOT cover K.
        # For a prime p, K is covered if K % p == residues[i] (with 0 handling).
        # Actually, residues[i] is in 0..p-1.
        # K % p gives the residue.
        # If residues[i] == K % p, it covers K.
        
        # We want to pick a prime i such that residues[i] != K % p.
        # Then set residues[i] = K % p.
        # This increases count at K by 1.
        
        # Let's pick a random prime that doesn't cover K
        candidates = []
        for i, p in enumerate(primes):
            r_k = K % p
            if residues[i] != r_k:
                candidates.append(i)
                
        if candidates:
            i = np.random.choice(candidates)
            p = primes[i]
            old_r = residues[i]
            new_r = K % p
            
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            # Check if this move is safe for indices < K
            # We need to ensure that for all j in 1..K-1, count[j] doesn't drop below 2.
            # The move removes coverage from old_r and adds to new_r.
            # So counts decrease at positions where x % p == old_r.
            # We only care if any of those positions are < K and had count == 2.
            
            # Efficient check:
            # Get indices in 1..K-1 where x % p == old_r
            # start_old = old_r (or p)
            # We can check the slice counts[old_start:K:p]
            # If any value in this slice is == 2, then decreasing it will make it 1, which is bad.
            
            if old_start < K:
                slice_old = counts[old_start:K:p]
                if np.any(slice_old == 2):
                    # This move would break coverage for some j < K
                    # Reject this targeted move, try random or skip
                    pass
                else:
                    # Safe move!
                    counts[old_start:max_check+1:p] -= 1
                    counts[new_start:max_check+1:p] += 1
                    new_L = _compute_L(counts, max_check)
                    
                    if new_L > current_L:
                        current_L = new_L
                        residues[i] = new_r
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                    else:
                        # Revert
                        counts[old_start:max_check+1:p] += 1
                        counts[new_start:max_check+1:p] -= 1
            else:
                # old_start >= K, so no impact on 1..K-1 from the removal part
                # (Wait, old_start is the first index >= 1. If old_start >= K, then no x in 1..K-1 is removed)
                counts[old_start:max_check+1:p] -= 1
                counts[new_start:max_check+1:p] += 1
                new_L = _compute_L(counts, max_check)
                
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    counts[old_start:max_check+1:p] += 1
                    counts[new_start:max_check+1:p] -= 1
        else:
            # All primes cover K. We can't increase count at K by adding a new prime.
            # We must rely on random moves or accept that K might be stuck.
            # Try a random move to escape local optimum
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            new_r = np.random.randint(p)
            
            if new_r == old_r:
                continue
                
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            counts[old_start:max_check+1:p] -= 1
            counts[new_start:max_check+1:p] += 1
            new_L = _compute_L(counts, max_check)
            
            if new_L >= current_L or np.random.rand() < np.exp((current_L - new_L) / T):
                current_L = new_L
                residues[i] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                counts[old_start:max_check+1:p] += 1
                counts[new_start:max_check+1:p] -= 1

    return best_residues.tolist()
# EVOLVE-BLOCK-END
