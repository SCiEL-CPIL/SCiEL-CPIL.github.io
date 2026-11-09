import time
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    MAX_L = 100000
    
    # Initialize residues to 0
    residues = np.zeros(n_primes, dtype=int)
    
    # Initialize coverage array
    cov = np.zeros(MAX_L + 1, dtype=int)
    
    # Calculate initial coverage
    for i, p in enumerate(primes):
        r = residues[i]
        if r <= MAX_L:
            cov[r::p] += 1
        else:
            # If r > MAX_L, it doesn't affect 1..MAX_L directly?
            # Wait, x % p == r. If r >= p, this is impossible for x < p?
            # No, r < p always.
            # So r is always < p <= 1000.
            # So r is always small.
            pass

    def advance_L():
        # Find the largest L such that cov[1..L] >= 2
        # We know cov[1..L-1] >= 2 from previous step.
        # So we just need to extend from current L.
        # But if we broke something, we might need to go back?
        # With the safety check, we shouldn't break anything.
        # So we can just scan forward.
        nonlocal L
        while L < MAX_L and cov[L+1] >= 2:
            L += 1
        return L
        
    L = 1
    # Initial check: is cov[1] >= 2?
    # If not, we need to fix 1.
    # But advance_L starts at 1.
    # If cov[1] < 2, L stays 1.
    # Then x = 2.
    # Wait, if cov[1] < 2, we are stuck at L=0?
    # The problem asks for 1..L.
    # If cov[1] < 2, L should be 0?
    # "largest L such that every integer 1..L satisfies..."
    # If 1 fails, L=0.
    # But we want to maximize.
    # So we must fix 1 first.
    
    # Let's initialize L=0 and start fixing from 1.
    L = 0
    
    # Main loop
    while time.time() < deadline:
        x = L + 1
        if x > MAX_L:
            break
            
        current_cov = cov[x]
        
        if current_cov >= 2:
            L += 1
            continue
            
        # Need to fix x
        # Find candidates
        needed = 2 - current_cov
        
        candidates = []
        for i, p in enumerate(primes):
            # Prefer p > x to avoid affecting 1..L
            if p > x:
                target_r = x % p
                if residues[i] != target_r:
                    old_r = residues[i]
                    # Safety check:
                    # If old_r <= L, we must ensure cov[old_r] > 2
                    # If old_r > L, it's safe
                    if old_r > L or cov[old_r] > 2:
                        candidates.append((i, p, old_r, target_r))
            
        if not candidates:
            # Try smaller primes? Risky.
            # Or break.
            # Let's try any prime that is safe.
            for i, p in enumerate(primes):
                target_r = x % p
                if residues[i] != target_r:
                    old_r = residues[i]
                    if old_r > L or cov[old_r] > 2:
                        candidates.append((i, p, old_r, target_r))
            
            if not candidates:
                # No safe move. Break.
                break
        
        # Sort by prime descending
        candidates.sort(key=lambda t: t[1], reverse=True)
        
        applied = 0
        for i, p, old_r, new_r in candidates:
            if applied >= needed:
                break
                
            # Apply change
            # Decrement old_r
            if old_r <= MAX_L:
                cov[old_r::p] -= 1
            # Increment new_r
            if new_r <= MAX_L:
                cov[new_r::p] += 1
            residues[i] = new_r
            applied += 1
            
        # Advance L
        L = advance_L()

    return [int(r) for r in residues]
