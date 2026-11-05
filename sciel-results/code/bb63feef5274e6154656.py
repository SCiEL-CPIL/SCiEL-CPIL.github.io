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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # Initial solution: random
    np.random.seed(42)
    best_residues = [np.random.randint(0, p) for p in primes]
    best_L = _count_covered(best_residues, primes, MAX_L)
    
    # Precompute coverage counts for the current solution
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for r, p in zip(best_residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= MAX_L:
            pos = np.arange(start, MAX_L + 1, p)
            if len(pos) > 0:
                counts[pos] += 1
    
    # Simulated Annealing Parameters
    T = 100.0
    alpha = 0.999
    min_T = 1e-5
    K = 50  # Number of candidates to evaluate per step
    
    current_residues = best_residues[:]
    current_L = best_L
    current_counts = counts.copy()
    
    # Precompute positions for each prime for fast access? 
    # Actually, generating arange is fast enough for p <= 1000.
    # But we can precompute the masks for all residues of small primes? No, too much memory.
    
    # Strategy:
    # 1. Pick a random prime index.
    # 2. Generate K random candidate residues.
    # 3. For each candidate, compute the new L using vectorized ops on the counts array.
    # 4. Accept the best candidate if it improves, or with probability exp((L_new - L_old)/T) if it worsens.
    
    while (time.time() - start_time) < deadline and T > min_T:
        i = np.random.randint(0, n_primes)
        p = primes[i]
        current_r = current_residues[i]
        
        # Get old positions
        if current_r == 0:
            old_start = p
        else:
            old_start = current_r
        
        if old_start <= MAX_L:
            old_pos = np.arange(old_start, MAX_L + 1, p)
            old_mask = np.zeros(MAX_L + 1, dtype=np.int8)
            if len(old_pos) > 0:
                old_mask[old_pos] = 1
        else:
            old_pos = np.array([], dtype=np.int32)
            old_mask = np.zeros(MAX_L + 1, dtype=np.int8)
        
        # Generate candidates
        # For small primes, sample without replacement. For large, just random.
        if p <= K:
            candidates = np.random.choice(p, size=p, replace=False)
            # Remove current if present? Not strictly necessary, but good to avoid no-ops
            if current_r in candidates:
                candidates = candidates[candidates != current_r]
        else:
            candidates = np.random.randint(0, p, size=K)
            # Remove duplicates and current
            unique_candidates = np.unique(candidates)
            unique_candidates = unique_candidates[unique_candidates != current_r]
            if len(unique_candidates) > K:
                unique_candidates = unique_candidates[:K]
            candidates = unique_candidates
            
        if len(candidates) == 0:
            T *= alpha
            continue
            
        best_candidate_r = current_r
        best_candidate_L = current_L
        best_candidate_mask = None
        
        # Evaluate all candidates
        # To speed this up, we can compute the "delta" in counts.
        # new_counts = current_counts - old_mask + new_mask
        # We want to find L_new = first index where new_counts < 2.
        
        # Vectorizing the L-finding for multiple candidates is tricky because 
        # the "first index" operation is not easily vectorizable across different arrays.
        # However, we can compute the new_counts array for each candidate.
        
        # Optimization: 
        # Instead of creating a full MAX_L+1 array for each candidate, 
        # we can note that only positions in old_pos or new_pos change.
        # But finding the "first index < 2" requires checking the whole array 
        # unless we maintain a segment tree or similar structure.
        
        # Given the 140s deadline and K=50, n_primes=168, 
        # iterations ~ 140 / (time_per_iter).
        # time_per_iter ~ K * (cost of updating array + finding L).
        # Cost of finding L on 100k array is ~100k ops.
        # 50 * 100k = 5M ops per iteration. 
        # 140s / (5M ops / 100M ops/s) = 140 / 0.05 = 2800 iterations.
        # This seems feasible.
        
        for new_r in candidates:
            if new_r == 0:
                new_start = p
            else:
                new_start = new_r
            
            if new_start <= MAX_L:
                new_pos = np.arange(new_start, MAX_L + 1, p)
                new_mask = np.zeros(MAX_L + 1, dtype=np.int8)
                if len(new_pos) > 0:
                    new_mask[new_pos] = 1
            else:
                new_mask = np.zeros(MAX_L + 1, dtype=np.int8)
            
            new_counts = current_counts - old_mask + new_mask
            
            # Find L
            below_two = np.where(new_counts[1:] < 2)[0]
            if len(below_two) == 0:
                new_L = MAX_L
            else:
                new_L = below_two[0]
                
            if new_L > best_candidate_L:
                best_candidate_L = new_L
                best_candidate_r = new_r
                best_candidate_mask = new_mask
        
        # Acceptance criterion
        if best_candidate_L > current_L:
            # Always accept improvement
            current_residues[i] = best_candidate_r
            current_L = best_candidate_L
            # Update counts
            current_counts = current_counts - old_mask + best_candidate_mask
            
            # Update best
            if current_L > best_L:
                best_L = current_L
                best_residues = current_residues[:]
                
        else:
            # Simulated Annealing acceptance
            delta_L = best_candidate_L - current_L
            if delta_L < 0:
                prob = np.exp(delta_L / T)
                if np.random.rand() < prob:
                    current_residues[i] = best_candidate_r
                    current_L = best_candidate_L
                    current_counts = current_counts - old_mask + best_candidate_mask
                    
        T *= alpha

    return best_residues
