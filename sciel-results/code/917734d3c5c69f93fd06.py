import numpy as np
import time

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]

def _compute_L(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has coverage >= 2."""
    counts = np.zeros(cap + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start < 1:
            start = r + p
        if start > cap:
            continue
        counts[start::p] += 1
    bad = np.where(counts[1:cap+1] < 2)[0]
    if len(bad) == 0:
        return cap
    return bad[0]

def _score(residues, primes, cap=100000):
    return _compute_L(residues, primes, cap)

def _compute_counts(residues, primes, cap=100000):
    """Compute coverage counts for 1..cap."""
    counts = np.zeros(cap + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start < 1:
            start = r + p
        if start > cap:
            continue
        counts[start::p] += 1
    return counts

def _greedy_set_cover(primes, cap=100000):
    """
    Greedy set-cover heuristic: process primes in increasing order.
    For each prime, select the residue that maximizes the number of positions
    in [1, cap] that are currently covered exactly once (count == 1),
    thereby converting them to double-covered (count == 2).
    
    This is a structural change from hill-climbing to a deterministic greedy
    construction that builds coverage systematically.
    """
    n_primes = len(primes)
    residues = np.zeros(n_primes, dtype=np.int64)
    counts = np.zeros(cap + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        
        # Try all possible residues for this prime
        for r in range(p):
            # Compute how many positions in [1, cap] would go from count==1 to count==2
            # Position x is covered by this prime with residue r iff x ≡ r (mod p)
            # i.e., x = r, r+p, r+2p, ... but we need x >= 1
            start = r
            if start < 1:
                start = r + p
            
            if start > cap:
                gain = 0
            else:
                # Positions: start, start+p, start+2p, ...
                positions = np.arange(start, cap + 1, p)
                # Count how many of these have count == 1
                gain = np.sum(counts[positions] == 1)
            
            # Also consider: we want to avoid making any position go from count >= 2 to count >= 3
            # (that would be wasted), but more importantly we want to fix positions with count == 1
            # Actually, the gain should also account for positions that are currently count == 0
            # and would become count == 1 (not helpful for double coverage)
            # And positions with count >= 2 that would become count >= 3 (neutral or slightly bad)
            
            # Let's refine: gain = (# positions going from 1 to 2) - 0.5*(# positions going from 0 to 1)
            # Actually, simpler: just maximize # positions going from 1 to 2
            # If tie, prefer residues that also cover some 0->1 positions (for future)
            
            if gain > best_gain or (gain == best_gain and r < best_r):
                best_gain = gain
                best_r = r
        
        residues[i] = best_r
        
        # Update counts with the chosen residue
        r = best_r
        start = r
        if start < 1:
            start = r + p
        if start <= cap:
            counts[start::p] += 1
    
    return residues

def _greedy_critical_interval(primes, cap=100000):
    """
    Greedy that focuses on the critical interval [1, L] where L is the current
    double-covered prefix. For each prime, select the residue that maximizes
    coverage of positions in [1, L] that are currently covered exactly once.
    """
    n_primes = len(primes)
    residues = np.zeros(n_primes, dtype=np.int64)
    counts = np.zeros(cap + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        # Find current L (double-covered prefix)
        if i == 0:
            L = 0
        else:
            bad = np.where(counts[1:min(cap, 100000)+1] < 2)[0]
            if len(bad) == 0:
                L = min(cap, 100000)
            else:
                L = bad[0]
        
        best_r = 0
        best_gain = -1
        
        # Focus on the critical interval [1, max(L, p)]
        # But we also want to extend L, so consider a window beyond L
        search_cap = min(cap, max(L + 2*p, p*2))
        
        for r in range(p):
            start = r
            if start < 1:
                start = r + p
            
            if start > search_cap:
                gain = 0
            else:
                positions = np.arange(start, search_cap + 1, p)
                # Gain: positions in [1, search_cap] with count == 1
                gain = np.sum(counts[positions] == 1)
            
            if gain > best_gain or (gain == best_gain and r < best_r):
                best_gain = gain
                best_r = r
        
        residues[i] = best_r
        
        # Update counts
        r = best_r
        start = r
        if start < 1:
            start = r + p
        if start <= cap:
            counts[start::p] += 1
    
    return residues

def _local_search(residues, primes, cap=100000, deadline=None, rng=None):
    """
    Local search to improve the greedy solution.
    Uses fast prefix scoring for efficiency.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    n_primes = len(primes)
    current = residues.copy()
    current_score = _score(current, primes, cap)
    best_score = current_score
    best_residues = current.copy()
    
    no_improve = 0
    max_no_improve = 200
    
    while no_improve < max_no_improve:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random prime to modify
        i = rng.randint(n_primes)
        p = primes[i]
        old_r = current[i]
        
        # Generate candidates
        candidates = set()
        candidates.add(old_r)
        
        # Random candidates
        for _ in range(5):
            candidates.add(rng.randint(p))
        
        # Targeted: find first gap and target it
        # Use a smaller cap for speed
        small_cap = min(cap, 20000)
        counts_small = np.zeros(small_cap + 1, dtype=np.int32)
        for j, pj in enumerate(primes):
            rj = current[j]
            start = rj
            if start < 1:
                start = rj + pj
            if start > small_cap:
                continue
            counts_small[start::pj] += 1
        
        bad = np.where(counts_small[1:small_cap+1] < 2)[0]
        if len(bad) > 0:
            u = bad[0] + 1  # first uncovered position
            candidates.add(u % p)
            for offset in [-3, -2, -1, 1, 2, 3]:
                if 1 <= u + offset <= small_cap:
                    candidates.add((u + offset) % p)
        
        best_new_r = old_r
        best_new_score = current_score
        
        for new_r in candidates:
            if new_r == old_r:
                continue
            current[i] = new_r
            s = _score(current, primes, cap)
            if s > best_new_score:
                best_new_score = s
                best_new_r = new_r
            current[i] = old_r
        
        if best_new_r != old_r:
            current[i] = best_new_r
            current_score = best_new_score
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_residues = current.copy()
        else:
            no_improve += 1
    
    return best_residues, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Phase 1: Greedy set-cover initialization
    # Try both variants and pick the better one
    res1 = _greedy_set_cover(primes, cap=100000)
    score1 = _score(res1, primes)
    
    if time.time() > deadline - 20:
        return [int(r) for r in res1]
    
    res2 = _greedy_critical_interval(primes, cap=100000)
    score2 = _score(res2, primes)
    
    if score1 >= score2:
        best_residues = res1
        best_score = score1
    else:
        best_residues = res2
        best_score = score2
    
    # Phase 2: Local search improvement
    rng = np.random.RandomState(42)
    
    # Multiple local search passes with different random seeds
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 10:
            break
        
        rng = np.random.RandomState(seed)
        res, score = _local_search(best_residues, primes, cap=100000, 
                                     deadline=deadline - 5, rng=rng)
        if score > best_score:
            best_score = score
            best_residues = res
    
    # Phase 3: If time remains, try more aggressive perturbation + local search
    if time.time() < deadline - 15:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            current = best_residues.copy()
            n_change = max(1, n_primes // 5)
            idx = rng.choice(n_primes, n_change, replace=False)
            for i in idx:
                current[i] = rng.randint(primes[i])
            
            res, score = _local_search(current, primes, cap=100000,
                                         deadline=deadline - 5, rng=rng)
            if score > best_score:
                best_score = score
                best_residues = res
    
    return [int(r) for r in best_residues]
