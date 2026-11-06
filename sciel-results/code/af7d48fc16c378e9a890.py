import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We aim for a score of at least 1000. 
    # Let's set MAX_CHECK to 2000 to allow some headroom for optimization, 
    # but the primary target is the prefix [1, 1000].
    MAX_CHECK = 2000
    
    # Precompute coverage masks for each prime and residue? 
    # That would be sum(p) booleans ~ 76k * 2000 = 150M booleans ~ 150MB. 
    # Might be too much memory or slow to init. 
    # Instead, we compute counts on the fly or maintain a dense array.
    
    # Strategy:
    # 1. Initialize residues using a "density" heuristic.
    #    For each prime p, choose r_p to maximize the number of x in [1, MAX_CHECK] 
    #    that are currently covered by 0 or 1 other primes? 
    #    Actually, a simpler initialization: 
    #    We want to cover [1, 1000] twice. 
    #    Random initialization is often bad. 
    #    Let's try a greedy "fill the gaps" initialization but optimized.
    
    counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Greedy Initialization:
    # Process primes in increasing order.
    # For each prime p, find the residue r that maximizes the "double coverage" potential.
    # Heuristic: Maximize sum(1 for x in 1..MAX_CHECK if counts[x] < 2 and x % p == r)
    # This is equivalent to: for each r, count how many x in [1, MAX_CHECK] with x%p==r have counts[x] < 2.
    
    for i, p in enumerate(primes):
        if time.time() > deadline - 20:
            break
            
        # We want to pick r to maximize coverage of "under-covered" numbers.
        # under_covered = [x for x in 1..MAX_CHECK if counts[x] < 2]
        # But computing this list every time is O(MAX_CHECK). 
        # We can vectorize the gain calculation.
        
        # Create a mask of under-covered numbers
        # Note: counts[0] is unused, we care about 1..MAX_CHECK
        under_mask = counts[1:MAX_CHECK+1] < 2
        
        if not np.any(under_mask):
            # Already double covered everything? Unlikely in init.
            residues[i] = 0
            # Update counts for r=0
            start = p
            if start <= MAX_CHECK:
                counts[start::p] += 1
            continue
            
        # For each residue r in 0..p-1, count how many under-covered numbers x satisfy x % p == r.
        # We can get the indices of under-covered numbers.
        under_indices = np.where(under_mask)[0] + 1
        
        if len(under_indices) == 0:
             residues[i] = 0
             start = p
             if start <= MAX_CHECK:
                 counts[start::p] += 1
             continue
             
        # Calculate residues of under-covered numbers mod p
        mods = under_indices % p
        
        # Count frequency of each residue
        # bincount is fast
        freqs = np.bincount(mods, minlength=p)
        
        # Pick the residue with max frequency
        best_r = np.argmax(freqs)
        residues[i] = best_r
        
        # Update counts
        r = best_r
        start = r if r > 0 else p
        if start <= MAX_CHECK:
            counts[start::p] += 1
            
    # Now we have an initial solution.
    # Calculate initial score
    def get_score(counts_arr):
        # Find first index in 1..MAX_CHECK where count < 2
        # If none, return MAX_CHECK (or higher if we checked further, but we capped at MAX_CHECK)
        below_two = np.where(counts_arr[1:MAX_CHECK+1] < 2)[0]
        if len(below_two) == 0:
            return MAX_CHECK
        return below_two[0]
        
    current_score = get_score(counts)
    best_score = current_score
    best_residues = residues.copy()
    
    # Local Search / Simulated Annealing Phase
    # We will use a vectorized approach to evaluate moves.
    # Changing residue of prime i from r_old to r_new:
    # 1. Subtract coverage of r_old: counts[r_old::p] -= 1 (adjust for 0)
    # 2. Add coverage of r_new: counts[r_new::p] += 1
    # 3. Calculate new score.
    
    # To speed up, we can precompute the "slices" for each prime? 
    # Or just use numpy slicing which is fast.
    
    # We will perform random restarts or hill climbing.
    # Given 120s, we can do many steps.
    
    # Let's use a simple hill climbing with random restarts if stuck.
    # Or just aggressive hill climbing with multiple candidates per step.
    
    # To be faster, we can batch the score calculation? 
    # No, score depends on the global state.
    
    # Optimization: 
    # Instead of checking the whole array for score, we can maintain a list of "bad" positions (count < 2).
    # But updating that list is tricky.
    # Let's stick to vectorized score calculation: np.where is O(N).
    # 2000 elements is tiny. 100,000 iterations * 2000 ops = 200M ops. 
    # In Python/NumPy, this might be slow if done in a tight loop with Python overhead.
    # Let's estimate: 10k iterations * 1ms/iter = 10s. 
    # We can do ~100k-500k iterations.
    
    # Let's try to make the inner loop as fast as possible.
    
    # Precompute slices? 
    # slices[i] = list of (start, step) for each residue? No, just one slice per prime for a given residue.
    
    # We'll iterate through primes in random order.
    
    no_improve = 0
    max_no_improve = 5000 # High threshold to allow long runs
    
    while time.time() < deadline - 5:
        improved = False
        
        # Pick a random prime to modify
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Try a few candidate new residues
        # 1. Random
        # 2. Residue that covers the current bottleneck (first x with count < 2)
        
        # Find current bottleneck
        below_two = np.where(counts[1:MAX_CHECK+1] < 2)[0]
        if len(below_two) > 0:
            bottleneck_x = below_two[0] + 1
            # The residue that covers this bottleneck is bottleneck_x % p
            target_r = bottleneck_x % p
            # Try target_r and a few neighbors
            candidates = [target_r]
            # Add some randoms
            candidates.extend(np.random.randint(0, p, 5))
        else:
            # No bottleneck in [1, MAX_CHECK], we are at max score for this MAX_CHECK
            # Try to extend? Or just random shuffle to find better structure for future?
            # Since we capped at MAX_CHECK, if we hit MAX_CHECK, we can't score higher unless we increase MAX_CHECK.
            # Let's increase MAX_CHECK if we hit it? 
            # No, that changes the problem. The task asks for L >= 1000. 
            # If we reach 2000, that's great. 
            # If we are stuck at 2000, we can't improve. 
            # But if we are stuck at, say, 1500, we need to improve.
            candidates = np.random.randint(0, p, 10)
            
        for new_r in candidates:
            if new_r == old_r:
                continue
            
            # Apply change
            # Remove old
            start_old = old_r if old_r > 0 else p
            if start_old <= MAX_CHECK:
                counts[start_old::p] -= 1
                
            # Add new
            start_new = new_r if new_r > 0 else p
            if start_new <= MAX_CHECK:
                counts[start_new::p] += 1
                
            # Calculate score
            new_score = get_score(counts)
            
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
                residues[i] = new_r
                improved = True
                no_improve = 0
                break
            else:
                # Revert
                if start_new <= MAX_CHECK:
                    counts[start_new::p] -= 1
                if start_old <= MAX_CHECK:
                    counts[start_old::p] += 1
                residues[i] = old_r
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart with a new random initialization? 
                # Or just keep going, maybe it's a local optimum.
                # Let's try a "kick": change 10 random residues to escape local optimum.
                for _ in range(10):
                    k = np.random.randint(n_primes)
                    pk = primes[k]
                    old_rk = residues[k]
                    new_rk = np.random.randint(pk)
                    
                    start_old_k = old_rk if old_rk > 0 else pk
                    if start_old_k <= MAX_CHECK:
                        counts[start_old_k::pk] -= 1
                    residues[k] = new_rk
                    start_new_k = new_rk if new_rk > 0 else pk
                    if start_new_k <= MAX_CHECK:
                        counts[start_new_k::pk] += 1
                no_improve = 0
                # Recalculate best score after kick? 
                # The kick might have lowered the score, but we keep best_residues.
                # We continue from the kicked state.
                
    return [int(r) for r in best_residues]
