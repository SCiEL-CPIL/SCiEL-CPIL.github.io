"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.array(np.nonzero(sieve)[0])

def _compute_L(residues, primes, cap=100000):
    """
    Compute the largest L such that every integer 1..L is covered by at least 2 congruences.
    residues: array of length len(primes), residues[i] is r for prime primes[i]
    primes: array of primes
    """
    n = len(primes)
    # For each x in 1..cap, count how many primes p have x % p == residues[i]
    # We need count >= 2 for all x in 1..L
    
    # Efficient approach: for each x, we need to check all primes.
    # But cap can be up to 100000, and n=168, so 16.8M checks. That's fine.
    
    # Let's use vectorized numpy where possible.
    # For each prime index i, the values x in 1..cap where x % p_i == r_i
    # are x = r_i + k*p_i for k >= 0, with x >= 1.
    
    # Build a coverage count array of size cap+1 (index 0 unused, or use 1..cap)
    counts = np.zeros(cap + 1, dtype=np.int32)
    
    for i in range(n):
        p = primes[i]
        r = residues[i]
        # Values: r, r+p, r+2p, ... but we need x >= 1
        # If r == 0, first value is p (since x=0 not in 1..cap, but x=p is)
        # Actually x = r + k*p for k such that x >= 1 and x <= cap
        if r == 0:
            start = p
        else:
            start = r
        # But we need start >= 1
        if start < 1:
            start = 1 + ((1 - r) % p) if r != 0 else p
            # Simpler: start = r if r >= 1 else p
            start = r if r >= 1 else p
        
        if start > cap:
            continue
        
        # All values: start, start+p, start+2p, ... <= cap
        idx = np.arange(start, cap + 1, p)
        counts[idx] += 1
    
    # Now find the largest L such that counts[1..L] are all >= 2
    # Find the first index where count < 2
    bad = np.where(counts[1:] < 2)[0]
    if len(bad) == 0:
        return cap
    else:
        return bad[0]  # This is the 0-based index in counts[1:], so the value is bad[0] + 1 - 1 = bad[0]? No.
        # counts[1:] has indices 0..cap-1 corresponding to x=1..cap
        # If bad[0] = j, then counts[1+j] < 2, i.e., x = 1+j is bad.
        # So L = 1+j - 1 = j. Wait, if x=1+j is the first bad, then L = 1+j-1 = j.
        # But j is 0-based in counts[1:], so x = 1 + j.
        # If j=0, x=1 is bad, L=0.
        # If j=5, x=6 is bad, L=5.
        # So L = j.
        return bad[0]

def _greedy_construct(primes, max_time=140.0):
    """
    Greedy construction: assign residues one prime at a time to maximize L.
    Strategy: for each prime in order, try all possible residues and pick the one
    that maximizes the resulting L (or a proxy).
    
    To be efficient, we use a first-bad-point heuristic:
    - Maintain current coverage counts for x in 1..some_window
    - When adding a new prime, focus on covering the current first-bad-point
    """
    n = len(primes)
    deadline = time.time() + max_time
    
    # Start with no residues assigned
    residues = np.zeros(n, dtype=np.int64)
    
    # We'll build incrementally. For efficiency, we track coverage for a window.
    # Window size: start small, expand as needed.
    # Actually, let's just compute L directly but cache the counts array.
    
    # Alternative: use a smarter approach.
    # For each prime i (in order), choose r_i to maximize L.
    # To evaluate a candidate r_i, we need to compute L with all previous primes + this one.
    # This is O(n * cap) per candidate, and p_i candidates per prime.
    # Total: sum(p_i * n * cap) which is too slow.
    
    # Better: maintain a counts array incrementally.
    # When we fix prime i to residue r, we add 1 to counts at positions r, r+p, ...
    # We can track the first-bad-point dynamically.
    
    # Let's use a window of size W that we expand as needed.
    W = 2000  # initial window
    counts = np.zeros(W + 1, dtype=np.int32)
    
    def first_bad():
        """Find the first x in 1..W where counts[x] < 2. Return x, or W+1 if none."""
        bad = np.where(counts[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W + 1
        return bad[0] + 1
    
    # Process primes in order
    for i in range(n):
        if time.time() > deadline - 5:
            break
        
        p = primes[i]
        best_r = 0
        best_fb = 0
        
        # Try a subset of residues to save time
        # Focus on residues that help cover the current first-bad-point
        current_fb = first_bad()
        
        # Candidate residues: r such that current_fb % p == r (to cover the first bad point)
        # Also try r=0 and a few others
        candidates = set()
        if current_fb <= W:
            candidates.add(current_fb % p)
        candidates.add(0)
        candidates.add(1)
        # Add a few random ones
        for _ in range(5):
            candidates.add(np.random.randint(0, p))
        
        # Limit candidates for large p
        if len(candidates) > 20:
            candidates = list(candidates)[:20]
        
        for r in candidates:
            # Tentatively add this residue
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start = p if r == 0 else r
            
            if start > W:
                # This residue doesn't affect the window, so L stays the same
                # But it might help later. For now, skip or handle.
                continue
            
            idx = np.arange(start, W + 1, p)
            counts[idx] += 1
            
            fb = first_bad()
            counts[idx] -= 1
            
            if fb > best_fb:
                best_fb = fb
                best_r = r
        
        # Commit the best residue
        residues[i] = best_r
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start = p if r == 0 else r
        
        if start <= W:
            idx = np.arange(start, W + 1, p)
            counts[idx] += 1
    
    # Compute final L
    L = _compute_L(residues, primes)
    
    # Now do iterative repair
    for iteration in range(10):
        if time.time() > deadline - 10:
            break
        
        # Find the first bad point
        # Compute counts for 1..L+100
        check_L = min(L + 200, 100000)
        counts_full = np.zeros(check_L + 1, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start = p if r == 0 else r
            if start > check_L:
                continue
            idx = np.arange(start, check_L + 1, p)
            counts_full[idx] += 1
        
        bad_idx = np.where(counts_full[1:check_L+1] < 2)[0]
        if len(bad_idx) == 0:
            L = min(check_L, 100000)
            continue
        
        # First bad point
        fb = bad_idx[0] + 1
        
        # Try to fix by reassigning some primes
        # Find primes that currently cover fb (i.e., fb % p == residues[i])
        # and primes that don't
        covering = []
        not_covering = []
        for i in range(n):
            p = primes[i]
            if fb % p == residues[i]:
                covering.append(i)
            else:
                not_covering.append(i)
        
        # Try reassigning one prime from not_covering to cover fb
        for i in not_covering[:50]:  # limit work
            if time.time() > deadline - 10:
                break
            p = primes[i]
            old_r = residues[i]
            new_r = fb % p
            
            if new_r == old_r:
                continue
            
            # Check if this improves L
            # Temporarily change
            residues[i] = new_r
            new_L = _compute_L(residues, primes)
            if new_L > L:
                L = new_L
            else:
                residues[i] = old_r
    
    return residues, L

def construct():
    """
    Return 168 residues, one for each prime p <= 1000 in increasing order.
    """
    primes = _primes_up_to(1000)
    n = len(primes)
    
    # Use greedy construction
    residues, L = _greedy_construct(primes, max_time=140.0)
    
    # Ensure correct length and types
    residues = np.array(residues, dtype=np.int64)
    if len(residues) != n:
        # Pad or truncate
        if len(residues) < n:
            residues = np.concatenate([residues, np.zeros(n - len(residues), dtype=np.int64)])
        else:
            residues = residues[:n]
    
    # Ensure 0 <= r < p
    for i in range(n):
        residues[i] = residues[i] % primes[i]
    
    return residues.tolist()

# EVOLVE-BLOCK-END
