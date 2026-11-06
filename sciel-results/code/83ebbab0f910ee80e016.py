"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We need to determine the residues r_p for each prime p.
    # The goal is to maximize L such that every x in [1, L] is covered by at least 2 residues.
    
    # Strategy: Greedy Set-Cover with Forward Pass and Lookahead
    # 1. Process primes in increasing order.
    # 2. Maintain an array `coverage` where coverage[x] is the number of primes 
    #    processed so far that cover x.
    # 3. For the current prime p, we choose a residue r to maximize a "potential" score.
    #    The potential of a residue r is the number of integers x in a lookahead window 
    #    [1, W] such that:
    #    - x is covered by this new residue (x % p == r)
    #    - coverage[x] is currently 1 (so this new residue makes it 2, which is good)
    #    We penalize if coverage[x] is already >= 2 (redundant) or 0 (not helpful for prefix yet, 
    #    unless we are building up for later).
    #    Actually, simply maximizing the number of x with coverage[x]==1 that we cover is a strong heuristic.
    #    Let's also consider x with coverage[x]==0 if we are early on, but coverage[x]==1 is the key to "double cover".
    
    # Lookahead window: How far ahead should we look?
    # The final score L might be around 1000-2000. 
    # If we look too far, we might over-optimize for distant points.
    # If we look too close, we might get stuck.
    # Let's use a window that expands as we go, or a fixed large window like 5000.
    # Given 168 primes, and we want L>=1000, a window of 5000 seems reasonable.
    # However, calculating coverage for a large window for every prime (168 * 5000) is 840k ops, which is fast.
    
    W = 5000  # Lookahead window size
    coverage = np.zeros(W + 1, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Precompute positions for each prime? 
    # For a prime p, the positions covered by residue r are r, r+p, r+2p...
    # We can just iterate.
    
    for idx, p in enumerate(primes):
        # Choose best residue r in [0, p-1]
        best_r = 0
        best_score = -1
        
        # Optimization: 
        # For a given p, the positions covered by r are r, r+p, ...
        # The "potential" is sum(1 for x in covered(r) if coverage[x] == 1)
        # Let's compute this for all r efficiently?
        # It's hard to do for all r at once with numpy without creating a huge matrix.
        # p is at most 997. W is 5000.
        # Iterating r from 0 to p-1, and for each r iterating x... 
        # Total ops per prime: sum_{r=0}^{p-1} (W/p) = W.
        # So total ops for one prime is ~W.
        # Total ops for all primes: 168 * 5000 = 840,000. This is very fast.
        
        # Let's implement the inner loop in Python, it might be slow if not careful.
        # W=5000, p~100. 50 iterations per r. 100 r's. 5000 ops per prime.
        # 168 primes * 5000 = 840k ops. Python can do ~10M-50M simple ops/sec.
        # So this should take < 1 second.
        
        # But we can do better with numpy.
        # For each r, we want to count how many x in [r, W] with step p have coverage[x]==1.
        
        # Let's try a numpy approach for the scoring of residues.
        # Create a boolean mask of where coverage == 1.
        mask_one = (coverage == 1)
        
        # For each r, we look at indices r, r+p, r+2p...
        # We can precompute the sum of mask_one over these strides?
        # No, because the start index r changes.
        
        # Let's just loop over r. It's small (p < 1000).
        for r in range(p):
            # Indices: r, r+p, ... up to W
            # Note: if r==0, indices are p, 2p... (since x>=1, and 0 is not in 1..W usually, 
            # but our coverage array is 0..W. The problem says x in 1..L.
            # So we should only consider x >= 1.
            # If r == 0, the first index is p (since 0 is not in 1..W).
            # If r > 0, the first index is r.
            
            start = r if r > 0 else p
            if start > W:
                score = 0
            else:
                # Get the slice
                # We want sum of mask_one[start::p]
                score = np.sum(mask_one[start::p])
            
            # Tie-breaking: 
            # If scores are equal, prefer residues that cover more x with coverage==0? 
            # Or just keep the first one.
            # Let's add a small tie-breaker: prefer smaller r? Or random?
            # Smaller r might be better for "early" coverage? Not necessarily.
            # Let's stick to max score.
            
            if score > best_score:
                best_score = score
                best_r = r
            elif score == best_score and r < best_r:
                # Tie-break: prefer smaller residue? 
                # This is arbitrary but deterministic.
                best_r = r
        
        residues[idx] = best_r
        
        # Update coverage array
        # Add 1 to positions covered by best_r
        start = best_r if best_r > 0 else p
        if start <= W:
            coverage[start::p] += 1
        
        # Time check
        if time.time() > deadline - 10.0:
            break
    
    # Now we have a set of residues.
    # Let's evaluate the actual prefix score L.
    # We need to find the largest L such that coverage[x] >= 2 for all x in 1..L.
    # Note: coverage here is based on the first W integers.
    # If the prefix extends beyond W, we need to extend the calculation.
    # But W=5000 is likely enough.
    
    # Let's verify the score on the full coverage array.
    # We only computed coverage up to W.
    # If the score is W, we might need to check further.
    # But let's assume W is large enough.
    
    # Find first x in 1..W where coverage[x] < 2
    # If none, score is at least W.
    
    # To be safe, let's extend the coverage check if needed.
    # But for now, let's just find the prefix in 1..W.
    
    below_two = np.where(coverage[1:W+1] < 2)[0]
    if len(below_two) == 0:
        L = W
    else:
        L = below_two[0]
        
    # If L is close to W, we might want to try to extend it?
    # Or just return.
    
    # Let's do a quick local search improvement if we have time.
    # The greedy pass might have local minima.
    # We can try to swap residues of a few primes to fix the first gap.
    
    if time.time() < deadline - 5.0:
        # Local search phase
        # Re-initialize coverage for local search? 
        # We have the coverage from the greedy pass.
        # We can use it.
        
        # Find the first gap
        gap_idx = L + 1
        if gap_idx > W:
            # Extend coverage?
            # For now, if we hit the wall, we stop.
            pass
        else:
            # We want to increase coverage[gap_idx] to >= 2.
            # Currently coverage[gap_idx] is 0 or 1.
            # If it is 1, we need to add one more cover.
            # If it is 0, we need to add two covers.
            
            # Try to find a prime that can cover gap_idx without breaking the prefix.
            # A prime p covers gap_idx if gap_idx % p == r_p.
            # We can change r_p to gap_idx % p.
            # But we must ensure that for all x < gap_idx, coverage[x] remains >= 2.
            
            # This is a constraint satisfaction problem.
            # Let's try a simple hill climb:
            # Pick a prime, try a new residue, check if prefix extends.
            
            current_residues = residues.copy()
            current_coverage = coverage.copy()
            current_L = L
            
            # We'll do a few iterations of local search
            # To speed up, we only look at primes that are "small" or random.
            
            for _ in range(500): # Max 500 local search steps
                if time.time() > deadline - 2.0:
                    break
                
                # Pick a random prime
                i = np.random.randint(n_primes)
                p = primes[i]
                old_r = current_residues[i]
                
                # Try to set residue to cover the current gap
                target_r = gap_idx % p
                
                if target_r == old_r:
                    continue
                
                # Check if changing r to target_r breaks the prefix
                # Affected positions:
                # Lose: old_r, old_r+p, ... < gap_idx
                # Gain: target_r, target_r+p, ... < gap_idx (and >= gap_idx)
                
                # Actually, we only care about x < gap_idx for validity.
                # And we care about x = gap_idx for extension.
                
                # Let's simulate the change on a copy? No, too slow.
                # Let's check counts.
                
                # Positions losing coverage (x < gap_idx, x % p == old_r)
                if old_r > 0:
                    pos_lose = np.arange(old_r, gap_idx, p)
                else:
                    pos_lose = np.arange(p, gap_idx, p)
                
                # If any of these have coverage == 2, they will become 1, breaking prefix.
                if len(pos_lose) > 0:
                    if np.any(current_coverage[pos_lose] == 2):
                        continue # Invalid move
                
                # Positions gaining coverage (x < gap_idx, x % p == target_r)
                if target_r > 0:
                    pos_gain = np.arange(target_r, gap_idx, p)
                else:
                    pos_gain = np.arange(p, gap_idx, p)
                
                # This move is valid if no prefix point drops below 2.
                # We already checked the losing points.
                # The gaining points will increase, so they are safe.
                
                # Now, does this extend the prefix?
                # We need coverage[gap_idx] to become >= 2.
                # Current coverage[gap_idx] is C.
                # If gap_idx % p == target_r, then this prime adds 1 to coverage[gap_idx].
                # So new coverage[gap_idx] = C + 1 (if it wasn't already covered by this prime) 
                # or C (if it was, but we changed r, so it wasn't).
                # Wait, if old_r != target_r, then the prime was NOT covering gap_idx before.
                # So it adds 1.
                
                # If C == 1, new is 2. Prefix extends to at least gap_idx.
                # If C == 0, new is 1. Prefix does not extend yet.
                
                # Let's apply the change if it's valid.
                # We don't necessarily need to extend immediately, but it helps.
                
                # Apply change
                # Update current_coverage
                if old_r > 0:
                    current_coverage[old_r::p] -= 1
                else:
                    current_coverage[p::p] -= 1
                
                if target_r > 0:
                    current_coverage[target_r::p] += 1
                else:
                    current_coverage[p::p] += 1
                
                current_residues[i] = target_r
                
                # Recalculate prefix
                # We only need to check from 1 to gap_idx (or further)
                # Actually, the prefix might have extended further if we fixed multiple gaps?
                # No, we only fixed gap_idx.
                # But maybe gap_idx+1 was also 1-covered and is now 2-covered?
                # Unlikely, but possible.
                
                # Let's find the new prefix.
                # We can just scan from 1 to W.
                # But that's O(W). 500 * 5000 = 2.5M ops. OK.
                
                below_two_new = np.where(current_coverage[1:W+1] < 2)[0]
                if len(below_two_new) == 0:
                    new_L = W
                else:
                    new_L = below_two_new[0]
                
                if new_L > current_L:
                    current_L = new_L
                    gap_idx = current_L + 1
                    # Update global best
                    if current_L > L:
                        L = current_L
                        residues = current_residues.copy()
                        coverage = current_coverage.copy()
                else:
                    # Revert? 
                    # If it didn't help, maybe revert to avoid noise?
                    # Or keep it? Keeping it might help later.
                    # Let's keep it for now, as it's a valid state.
                    pass

    return [int(r) for r in residues]
# EVOLVE-BLOCK-END
