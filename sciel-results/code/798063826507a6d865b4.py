import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    # Initial residues
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=int)
    
    N = 1000
    x = np.arange(1, N + 1)
    
    counts = np.zeros(N, dtype=int)
    for i, p in enumerate(primes):
        counts += (x % p == residues[i])
        
    def get_bottleneck(c):
        # Returns index of first element < 2, or -1 if none
        is_fail = c < 2
        if not np.any(is_fail):
            return -1
        return np.argmax(is_fail)
        
    current_bottleneck = get_bottleneck(counts)
    current_L = 0 if current_bottleneck == -1 else current_bottleneck
    
    while time.time() < deadline:
        improved = False
        order = np.random.permutation(num_primes)
        
        # Recompute bottleneck at start of pass? 
        # It changes if we accept a move.
        
        for i in order:
            if time.time() > deadline:
                break
            
            # Get current bottleneck
            b_idx = get_bottleneck(counts)
            if b_idx == -1:
                # Maxed out N
                break
                
            # We want to cover x = b_idx + 1
            target_x = b_idx + 1
            p = primes[i]
            
            # Does p currently cover target_x?
            current_r = residues[i]
            if (target_x % p) == current_r:
                # p already covers it. Changing p won't help cover it.
                # It might hurt. Skip.
                continue
                
            # Try to change p to cover target_x
            new_r = target_x % p
            
            # Check if this move is valid (doesn't break prefix)
            # We need to ensure that for all j < b_idx, count[j] >= 2 remains.
            # Actually, we just need to check if the new L is > current L.
            # Since we are fixing the first failure, if the move is safe for the prefix,
            # the new L will be at least b_idx + 1 (or more).
            
            # Fast check:
            # The move removes coverage at positions where x % p == current_r
            # and adds coverage at positions where x % p == new_r.
            # Positions removed: R = {x | x % p == current_r}
            # Positions added: A = {x | x % p == new_r}
            
            # We only care about x <= b_idx (the valid prefix) and x = b_idx+1 (the target).
            # Actually, we care about the whole array to compute new L.
            # But we can optimize:
            # If any x in prefix (1..b_idx) had count 2, and we remove its only other cover?
            # No, we remove cover from R.
            # If x in R and x <= b_idx, and counts[x] == 2, and x is NOT in A (which is true since R and A are disjoint mod p),
            # then counts[x] becomes 1. Failure.
            
            # So, check if any x in (1..b_idx) such that x % p == current_r has counts[x] == 2.
            # If yes, move is bad.
            
            # Let's check this condition.
            # mask_R = (x % p == current_r)
            # prefix_mask = (x <= b_idx)
            # danger_mask = mask_R & prefix_mask & (counts == 2)
            # if np.any(danger_mask): skip
            
            # Also, we must ensure we don't break x = b_idx+1? 
            # We are adding cover to b_idx+1. So counts[b_idx+1] increases. Good.
            
            # What about x > b_idx+1?
            # If we break a cover there, it doesn't affect L immediately, as long as prefix is safe.
            
            # So the condition is:
            # No x in 1..b_idx has (x % p == current_r) AND counts[x] == 2.
            
            # Let's implement this check.
            mask_R = (x % p == current_r)
            # We only care about prefix 1..b_idx
            # x is 1..N. Indices 0..b_idx-1 correspond to 1..b_idx.
            # So check indices 0..b_idx-1.
            
            # counts[:b_idx] are the counts for the prefix.
            # We need to check if any of these are 2 AND covered by current_r.
            
            # mask_R_prefix = mask_R[:b_idx]
            # counts_prefix = counts[:b_idx]
            # if np.any(mask_R_prefix & (counts_prefix == 2)):
            #    continue
            
            # If safe, apply move.
            # But wait, is it possible that this move fixes b_idx+1 but creates a new failure at b_idx+2?
            # Yes. But L will still increase to at least b_idx+1.
            # So it's an improvement.
            
            # Let's verify safety.
            if b_idx > 0:
                mask_R_prefix = (x[:b_idx] % p == current_r)
                if np.any(mask_R_prefix & (counts[:b_idx] == 2)):
                    continue
            
            # Apply move
            residues[i] = new_r
            # Update counts
            counts -= mask_R
            mask_A = (x % p == new_r)
            counts += mask_A
            
            # Update L
            new_b_idx = get_bottleneck(counts)
            new_L = 0 if new_b_idx == -1 else new_b_idx
            
            if new_L > current_L:
                current_L = new_L
                improved = True
            else:
                # If L didn't improve, revert?
                # It shouldn't happen if our safety check was correct.
                # But maybe the safety check was insufficient?
                # If L didn't improve, it means we broke something earlier?
                # Our check only looked at counts==2.
                # What if counts was 1? Then it becomes 0. But prefix was valid, so counts >= 2.
                # So safety check should be sufficient.
                
                # However, if new_L == current_L, it means we didn't fix the bottleneck?
                # We added cover to b_idx+1. So counts[b_idx+1] increased.
                # So bottleneck should move.
                # Unless b_idx+1 was already covered by 2 others? No, it was the bottleneck (count 1 or 0).
                # Wait, bottleneck is first index with count < 2.
                # So counts[b_idx] < 2.
                # We added cover to x = b_idx+1.
                # So counts[b_idx] is still < 2.
                # So L should NOT increase?
                # Ah! The bottleneck is at index `b_idx`.
                # The integer is `x = b_idx + 1`.
                # If we cover it, its count increases.
                # But the *previous* integer `x = b_idx` (index `b_idx-1`) was valid.
                # The *first* failure is at `b_idx`.
                # So L is `b_idx`.
                # If we fix `b_idx+1`, the first failure might still be `b_idx`?
                # No, `b_idx` is the index. The integer is `b_idx+1`.
                # If we fix integer `b_idx+1`, does the first failure move?
                # Yes, if `b_idx+1` was the first failure.
                # So new L should be > b_idx.
                
                pass
                
        if not improved:
            # Stuck. Try a random restart or different strategy?
            # For now, break.
            break
            
    return [int(r) for r in residues]
