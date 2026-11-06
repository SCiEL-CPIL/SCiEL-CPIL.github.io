import time

def construct():
    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    n_primes = len(primes)
    LIMIT = 2000  # We look ahead a bit to guide the greedy choice
    
    # counts[x] = number of primes covering x
    counts = [0] * (LIMIT + 1)
    residues = [0] * n_primes
    
    current_L = 0
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score_tuple = (-1, -1, -1) # (new_L, num_single_in_frontier, -num_zero_in_frontier)
        
        # We only need to check a subset of residues if p is large?
        # No, p <= 1000, so p residues is at most 1000.
        # 168 primes * 1000 residues * O(LIMIT/p) updates?
        # LIMIT=2000.
        # Total ops ~ 168 * 1000 * (2000/p).
        # Sum of 1/p for primes <= 1000 is roughly log(log(1000)) ~ 2.
        # So ~ 168 * 1000 * 2 * 2000? No.
        # For a specific p, the number of updates is LIMIT/p.
        # So cost per residue is LIMIT/p.
        # Total cost for prime p is p * (LIMIT/p) = LIMIT.
        # Total cost for all primes is n_primes * LIMIT = 168 * 2000 = 336,000.
        # This is very fast.
        
        for r in range(p):
            # Simulate adding r
            # We need to calculate the new L and the frontier stats
            
            # Instead of updating an array and reverting, we can just calculate the stats directly?
            # No, updating an array is easier to track L.
            # But we need to revert.
            
            # Optimization: Pre-calculate the positions covered by r
            # positions = [x for x in range(1, LIMIT+1) if x % p == r % p]
            # But we can just iterate.
            
            # Let's use a temporary array or just track changes?
            # Since LIMIT is small, we can copy the counts array?
            # 168 * 1000 * 2000 = 336,000,000 operations. Might be too slow in Python if not optimized.
            # 336M simple ops in Python might take ~30-60s. Too risky.
            
            # Better: Update in place, calculate score, revert.
            # Update cost: LIMIT/p.
            # Score calculation: Scan LIMIT to find new_L?
            # If we scan LIMIT for every residue, that's LIMIT * LIMIT/p = LIMIT^2 / p.
            # Total: n_primes * LIMIT^2 / p_avg?
            # 168 * 4,000,000 / 500 ~ 1.3M. That's fine.
            # Wait, scanning LIMIT for L is O(LIMIT).
            # So for each residue, cost is O(LIMIT/p) for update + O(LIMIT) for scan.
            # Dominated by O(LIMIT).
            # Total cost: n_primes * p * LIMIT.
            # 168 * 1000 * 2000 = 336,000,000.
            # In Python, 336M iterations is likely > 10s. Probably 30-50s.
            # We have 120s limit. It might pass.
            
            # Let's optimize the score calculation.
            # We don't need to scan the whole LIMIT if we know current_L.
            # The new L can only increase.
            # So we start scanning from current_L + 1.
            
            # Let's try to be more efficient.
            # We can maintain the counts array.
            
            # Update counts
            start = r if r > 0 else p
            if start <= LIMIT:
                for x in range(start, LIMIT + 1, p):
                    counts[x] += 1
            
            # Calculate new_L
            new_L = current_L
            # Check if we can extend
            # Actually, we should check from 1? No, current_L is valid for previous primes.
            # Adding a prime only increases counts. So [1, current_L] remains >= 2?
            # Wait. If counts[x] was 2, it becomes 3. Still >= 2.
            # If counts[x] was 1, it becomes 2.
            # If counts[x] was 0, it becomes 1.
            # So [1, current_L] was >= 2? No, current_L is defined as the largest L such that [1, L] is >= 2.
            # So yes, [1, current_L] is still >= 2.
            # We just need to find how far we can extend from current_L.
            
            # However, what if adding this prime *breaks* something? No, counts only increase.
            # So new_L >= current_L.
            
            x = current_L + 1
            while x <= LIMIT and counts[x] >= 2:
                new_L = x
                x += 1
            
            # Calculate frontier stats
            # We want to maximize the number of "1"s in the frontier [new_L+1, new_L+100]
            # And minimize "0"s?
            
            frontier_start = new_L + 1
            frontier_end = min(LIMIT, new_L + 100)
            
            num_ones = 0
            num_zeros = 0
            for x in range(frontier_start, frontier_end + 1):
                if counts[x] == 1:
                    num_ones += 1
                elif counts[x] == 0:
                    num_zeros += 1
            
            score_tuple = (new_L, num_ones, -num_zeros)
            
            if score_tuple > best_score_tuple:
                best_score_tuple = score_tuple
                best_r = r
            
            # Revert counts
            if start <= LIMIT:
                for x in range(start, LIMIT + 1, p):
                    counts[x] -= 1
        
        # Apply the best residue
        residues[i] = best_r
        start = best_r if best_r > 0 else p
        if start <= LIMIT:
            for x in range(start, LIMIT + 1, p):
                counts[x] += 1
        
        # Update current_L
        x = current_L + 1
        while x <= LIMIT and counts[x] >= 2:
            current_L = x
            x += 1

    return residues
