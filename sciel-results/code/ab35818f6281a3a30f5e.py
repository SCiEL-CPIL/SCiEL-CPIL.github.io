import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    p_list = np.array(primes)
    
    # We will work with a fixed limit for search, but verify with a larger one at the end
    SEARCH_LIMIT = 5000
    VERIFY_LIMIT = 100000
    
    def compute_coverage(residues, limit):
        """
        Compute coverage array for x in 1..limit.
        residues: array of length n_primes
        """
        cov = np.zeros(limit + 1, dtype=np.int16)
        for i in range(n_primes):
            p = p_list[i]
            r = int(residues[i])
            if r == 0:
                start = p
            else:
                start = r
            if start <= limit:
                cov[start::p] += 1
        return cov

    def get_prefix_L(cov, limit):
        """
        Find largest L such that cov[x] >= 2 for all x in 1..L.
        Returns L.
        """
        # cov[1] to cov[limit]
        # We want the first index where cov < 2
        # np.where returns indices where condition is True
        bad = np.where(cov[1:limit+1] < 2)[0]
        if len(bad) == 0:
            return limit
        return int(bad[0])  # First bad index is bad[0], so L = bad[0] (since 1-based, if cov[1] is bad, L=0? No. 
        # If cov[1] < 2, bad[0]=0, L=0. 
        # If cov[2] < 2, bad[0]=1, L=1.
        # So L = bad[0] is correct for 1-based indexing where L is the count of good integers.
        
    def full_score(residues, limit=VERIFY_LIMIT):
        cov = compute_coverage(residues, limit)
        return get_prefix_L(cov, limit)

    # Initialize residues randomly
    residues = np.array([random.randint(0, p-1) for p in p_list], dtype=np.int32)
    
    # --- PHASE 1: Greedy Density Maximization ---
    # Goal: Maximize the number of integers in [1, SEARCH_LIMIT] with coverage >= 2.
    # Strategy: For each prime, choose the residue that adds the most "value".
    # Value is defined as: +2 for each integer that goes from cov<2 to cov>=2, 
    # +1 for each integer that goes from cov>=2 to cov>2 (less important),
    # -100 for each integer that goes from cov>=2 to cov<2 (penalty for breaking coverage).
    
    # To make this fast, we approximate. 
    # Let's just do a simple greedy: 
    # Start with random residues.
    # Iterate through primes. For each prime, try all possible residues (or a subset if p is large).
    # Pick the one that maximizes the score: count of x in [1, SEARCH_LIMIT] with cov[x] >= 2.
    
    # Optimization: We can't try all residues for large p (up to 997). 
    # But we only care about the first SEARCH_LIMIT integers.
    # For a given prime p, the residue r affects positions r, r+p, r+2p...
    # We can evaluate the impact of changing residue for prime i.
    
    # Let's do a few passes of "Best Residue Selection"
    cov = compute_coverage(residues, SEARCH_LIMIT)
    current_good_count = np.sum(cov[1:SEARCH_LIMIT+1] >= 2)
    
    for pass_num in range(5):
        if time.time() > deadline - 30: break
        
        improved = False
        # Iterate in random order to avoid bias
        indices = list(range(n_primes))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 30: break
            
            p = p_list[i]
            old_r = int(residues[i])
            
            # Determine candidate residues to try
            # If p is small, try all. If p is large, try random samples + old.
            if p <= 100:
                candidates = range(p)
            else:
                # Sample 20 random residues + old
                candidates = [old_r] + [random.randint(0, p-1) for _ in range(20)]
            
            best_r = old_r
            best_score = current_good_count
            
            # Remove current contribution to compute baseline for this prime
            if old_r == 0:
                old_start = p
            else:
                old_start = old_r
            if old_start <= SEARCH_LIMIT:
                cov[old_start::p] -= 1
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                
                if new_start <= SEARCH_LIMIT:
                    cov[new_start::p] += 1
                    # Calculate new good count
                    # We only need to check the changed positions? 
                    # No, because "good count" is a global sum. 
                    # But we can update it incrementally.
                    # Let's just compute sum for now, it's O(SEARCH_LIMIT) per candidate.
                    # 5000 * 20 candidates * 168 primes * 5 passes = 8.4M ops. Fast enough.
                    new_count = np.sum(cov[1:SEARCH_LIMIT+1] >= 2)
                    cov[new_start::p] -= 1
                    
                    if new_count > best_score:
                        best_score = new_count
                        best_r = new_r
                else:
                    # No change in coverage for this range
                    new_count = current_good_count
                    if new_count > best_score:
                        best_score = new_count
                        best_r = new_r
            
            # Apply best
            if best_r != old_r:
                # Re-add old contribution (it was removed)
                if old_start <= SEARCH_LIMIT:
                    cov[old_start::p] += 1
                # Remove old, add best
                if old_start <= SEARCH_LIMIT:
                    cov[old_start::p] -= 1
                if best_r == 0:
                    best_start = p
                else:
                    best_start = best_r
                if best_start <= SEARCH_LIMIT:
                    cov[best_start::p] += 1
                    current_good_count = best_score
                    residues[i] = best_r
                    improved = True
                else:
                    # If best_r doesn't affect range, we don't need to update cov, but we should update residues
                    # However, if it doesn't affect range, score didn't change, so best_r would likely be old_r unless tie-breaking favors it.
                    # Let's just keep old_r if no improvement in score.
                    pass
            else:
                # Revert removal of old
                if old_start <= SEARCH_LIMIT:
                    cov[old_start::p] += 1
        
        if not improved:
            break
            
    # --- PHASE 2: Bottleneck Repair ---
    # Now we have a good density. Let's find the longest prefix with cov >= 2.
    cov = compute_coverage(residues, SEARCH_LIMIT)
    current_L = get_prefix_L(cov, SEARCH_LIMIT)
    
    # We want to extend current_L.
    # The bottleneck is x = current_L + 1.
    # We need to increase coverage of x.
    # Which primes can cover x? Primes p where we can choose r = x % p.
    # But changing r for prime i might break coverage of other y <= current_L.
    
    # Strategy:
    # 1. Identify x = current_L + 1.
    # 2. Find all primes i such that x is NOT currently covered by residue i. 
    #    (i.e., residues[i] != x % p_list[i])
    # 3. For each such prime i, calculate the cost of changing residues[i] to x % p_list[i].
    #    Cost = number of y in [1, current_L] such that cov[y] == 2 and y is covered by prime i's OLD residue but NOT new residue.
    #    If Cost == 0, the move is "safe" (doesn't break existing prefix).
    # 4. If a safe move exists, make it. This will increase cov[x] by 1.
    #    If cov[x] was 1, it becomes 2. If it was 0, it becomes 1.
    #    We need cov[x] >= 2. So we might need multiple moves.
    
    while time.time() < deadline - 10:
        cov = compute_coverage(residues, SEARCH_LIMIT)
        current_L = get_prefix_L(cov, SEARCH_LIMIT)
        
        if current_L >= SEARCH_LIMIT:
            # We've covered the search limit. We can try to expand limit or stop.
            # For now, if we hit SEARCH_LIMIT, we assume the full score is at least SEARCH_LIMIT.
            # We can break and verify.
            break
        
        x = current_L + 1
        
        # Find primes that do NOT cover x
        # x % p == residues[i] means it covers x.
        # We want i where residues[i] != x % p_list[i]
        
        safe_move_found = False
        
        # Iterate through primes. Smaller primes have more frequent occurrences, 
        # so changing them is more likely to break existing coverage. 
        # Larger primes have fewer occurrences, so they are "safer" to change.
        # Let's try larger primes first.
        indices = list(range(n_primes))
        indices.sort(key=lambda i: -p_list[i])
        
        for i in indices:
            if time.time() > deadline - 10: break
            
            p = p_list[i]
            old_r = int(residues[i])
            new_r = int(x % p)
            
            if old_r == new_r:
                continue # Already covers x
            
            # Check if changing old_r to new_r breaks any y in [1, current_L] with cov[y] == 2
            # Positions covered by old_r: old_start, old_start+p, ...
            # Positions covered by new_r: new_start, new_start+p, ...
            
            # We need to check intersection of [1, current_L] with old_positions \ new_positions
            # And ensure cov[y] > 2 for those y. If cov[y] == 2, it becomes 1 -> BAD.
            
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            # Generate positions
            old_pos = np.arange(old_start, current_L + 1, p)
            new_pos = np.arange(new_start, current_L + 1, p)
            
            # We only care about positions in old_pos that are NOT in new_pos
            # And where cov[pos] == 2
            
            # Efficient check:
            # Create a set of new positions for fast lookup? Or use boolean mask.
            # Since current_L <= 5000, sets are fine.
            new_pos_set = set(new_pos)
            
            bad_count = 0
            for pos in old_pos:
                if pos not in new_pos_set:
                    if cov[pos] == 2:
                        bad_count += 1
                        if bad_count > 0:
                            break
            
            if bad_count == 0:
                # Safe move!
                # Apply it
                if old_start <= SEARCH_LIMIT:
                    cov[old_start::p] -= 1
                if new_start <= SEARCH_LIMIT:
                    cov[new_start::p] += 1
                residues[i] = new_r
                safe_move_found = True
                break
        
        if not safe_move_found:
            # No single safe move. 
            # Try a "risky" move: pick the prime that minimizes bad_count, 
            # but only if it increases cov[x] significantly or if we are stuck.
            # Actually, if no safe move exists, we might be stuck in a local optimum 
            # where extending the prefix requires temporarily breaking it or a 2-move sequence.
            
            # Let's try a 2-move sequence:
            # Find i1, i2 such that:
            # 1. i1 covers x (after move)
            # 2. i2 covers x (after move)
            # 3. The union of their "breaks" is covered by the other's "fixes" or cov>2.
            
            # This is complex. Let's try a simpler heuristic:
            # Find the prime i that has the MINIMUM bad_count.
            # If min_bad_count is small (e.g., <= 2), we might accept it if it helps us progress later?
            # No, breaking the prefix reduces L. We want to INCREASE L.
            
            # Alternative: Use random perturbation to escape local optimum.
            # Pick a random prime, change its residue, and see if L improves after a few steps.
            
            # Let's do a small local search:
            # Pick 3 random primes, try all combinations of their residues? No, too big.
            # Pick 1 random prime, try all residues.
            
            i_rand = random.randint(0, n_primes - 1)
            p_rand = p_list[i_rand]
            old_r_rand = int(residues[i_rand])
            
            best_r_rand = old_r_rand
            best_L_rand = current_L
            
            # Try a few random residues
            for _ in range(10):
                new_r_rand = random.randint(0, p_rand - 1)
                if new_r_rand == old_r_rand: continue
                
                old_start_r = p_rand if old_r_rand == 0 else old_r_rand
                new_start_r = p_rand if new_r_rand == 0 else new_r_rand
                
                if old_start_r <= SEARCH_LIMIT:
                    cov[old_start_r::p_rand] -= 1
                if new_start_r <= SEARCH_LIMIT:
                    cov[new_start_r::p_rand] += 1
                
                new_L = get_prefix_L(cov, SEARCH_LIMIT)
                
                if new_L > best_L_rand:
                    best_L_rand = new_L
                    best_r_rand = new_r_rand
                else:
                    # Revert
                    if new_start_r <= SEARCH_LIMIT:
                        cov[new_start_r::p_rand] -= 1
                    if old_start_r <= SEARCH_LIMIT:
                        cov[old_start_r::p_rand] += 1
            
            if best_r_rand != old_r_rand:
                # Apply best
                old_start_r = p_rand if old_r_rand == 0 else old_r_rand
                new_start_r = p_rand if best_r_rand == 0 else best_r_rand
                if old_start_r <= SEARCH_LIMIT:
                    cov[old_start_r::p_rand] -= 1
                if new_start_r <= SEARCH_LIMIT:
                    cov[new_start_r::p_rand] += 1
                residues[i_rand] = best_r_rand
            else:
                # Stuck. Break and finalize.
                break

    # Final Verification with larger limit
    final_score = full_score(residues, VERIFY_LIMIT)
    
    return residues.tolist()

# EVOLVE-BLOCK-END
