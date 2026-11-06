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
    LIMIT = 10000 
    
    # counts[x] is the number of congruences satisfied by x
    counts = [0] * (LIMIT + 1)
    current_L = 0
    
    # We process primes in increasing order
    for p in primes:
        # Determine target x to cover
        # We want to extend current_L.
        # The bottleneck is current_L + 1.
        # If counts[current_L + 1] >= 1, covering it makes it >= 2, potentially extending L.
        # If counts[current_L + 1] == 0, covering it makes it 1. L doesn't extend.
        # In that case, it might be better to cover a point x > current_L where counts[x] == 1,
        # so that when current_L + 1 is eventually fixed, L can jump further.
        
        target_x = current_L + 1
        if target_x <= LIMIT and counts[target_x] == 0:
            # Look for a point x > current_L with count 1
            # We only need to look a bit ahead, or just the next one?
            # Scanning up to LIMIT is O(N), which is fine (10000 * 168 ~ 1.6M ops)
            found = False
            for x in range(current_L + 1, LIMIT + 1):
                if counts[x] == 1:
                    target_x = x
                    found = True
                    break
            if not found:
                target_x = current_L + 1
        
        # Choose residue
        r = target_x % p
        
        # Update counts
        # x = r, r+p, r+2p...
        # Note: if r == 0, multiples are p, 2p...
        # The condition is x % p == r.
        # So x = r, r+p, ...
        # If r == 0, x = 0, p, 2p... but we care about 1..LIMIT.
        # So start = r if r > 0 else p.
        
        start = r if r > 0 else p
        for x in range(start, LIMIT + 1, p):
            counts[x] += 1
        
        # Update current_L
        # Check if we can extend
        while current_L < LIMIT and counts[current_L + 1] >= 2:
            current_L += 1
            
        if current_L >= LIMIT:
            break

    # Construct final residues list
    # We need to return the residues chosen.
    # We didn't store them in the loop above!
    # We need to re-run or store them.
    
    # Let's re-run to store residues, or modify loop to store.
    # Since the logic is deterministic, we can just re-run it quickly.
    # Or better, store in the first pass.
    
    residues = []
    counts = [0] * (LIMIT + 1)
    current_L = 0
    
    for p in primes:
        target_x = current_L + 1
        if target_x <= LIMIT and counts[target_x] == 0:
            found = False
            for x in range(current_L + 1, LIMIT + 1):
                if counts[x] == 1:
                    target_x = x
                    found = True
                    break
            if not found:
                target_x = current_L + 1
        
        r = target_x % p
        residues.append(r)
        
        start = r if r > 0 else p
        for x in range(start, LIMIT + 1, p):
            counts[x] += 1
        
        while current_L < LIMIT and counts[current_L + 1] >= 2:
            current_L += 1
            
        if current_L >= LIMIT:
            # Fill remaining residues with 0 (or arbitrary)
            # But we must return 168 residues.
            # If we break early, we need to fill the rest.
            # However, the loop iterates over all primes.
            # If current_L >= LIMIT, we can just append 0 for remaining.
            pass

    # If we broke early or just finished, ensure length is 168
    # The loop runs for all primes in `primes` (168 primes).
    # So `residues` will have length 168.
    
    return residues
