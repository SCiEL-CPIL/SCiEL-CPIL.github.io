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

def _fast_score_prefix(residues, primes, prefix_len):
    """Fast score computation only up to prefix_len (for local search)."""
    if prefix_len > 100000:
        prefix_len = 100000
    counts = np.zeros(prefix_len + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start < 1:
            start = r + p
        if start > prefix_len:
            continue
        counts[start::p] += 1
    bad = np.where(counts[1:prefix_len+1] < 2)[0]
    if len(bad) == 0:
        return prefix_len
    return bad[0]

def _greedy_frontier_init(primes, window=5000):
    """
    Greedy initialization: process primes in order, for each prime choose the residue
    that maximizes the number of positions in the frontier that go from coverage 1 to 2.
    """
    n_primes = len(primes)
    residues = np.zeros(n_primes, dtype=np.int64)
    cov = np.zeros(window + 1, dtype=np.int32)
    
    for idx, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        # For smaller primes, try all residues; for larger, sample
        if p <= 200:
            candidates = range(p)
        else:
            # Sample residues: include 0, and some spread out ones
            candidates = list(set([0, 1, 2, p-1, p//2, (p*3)//4, p//3]))
            # Add some random ones
            rng = np.random.RandomState(idx)
            candidates.extend(rng.randint(p, size=20).tolist())
        
        for r in candidates:
            first = r if r >= 1 else p
            if first > window:
                gain = 0
            else:
                positions = np.arange(first, window + 1, p)
                gain = int(np.sum(cov[positions] == 1))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = best_r
        first = best_r if best_r >= 1 else p
        if first <= window:
            positions = np.arange(first, window + 1, p)
            cov[positions] += 1
    
    return residues

def _targeted_repair(residues, primes, current_L, time_budget=30):
    """
    Targeted repair: repeatedly find the first gap, and try to fix it by
    changing residues of primes that can cover the gap position.
    """
    start_time = time.time()
    n_primes = len(primes)
    best_residues = residues.copy()
    best_L = current_L
    max_iterations = 500
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        if best_L >= 100000:
            break
        
        # Find first gap
        gap_pos = best_L + 1
        if gap_pos > 100000:
            break
        
        # Find primes that could cover gap_pos (i.e., where gap_pos % p != current residue)
        # and also primes that already cover it
        covering = []
        not_covering = []
        for i, p in enumerate(primes):
            if gap_pos % p == best_residues[i]:
                covering.append(i)
            else:
                not_covering.append(i)
        
        if len(covering) >= 2:
            # Already covered, recompute L
            new_L = _score(best_residues, primes)
            if new_L > best_L:
                best_L = new_L
                continue
            else:
                # The gap is actually later, recompute
                break
        
        # Need to add at least (2 - len(covering)) more primes covering gap_pos
        # Try changing one prime from not_covering to cover gap_pos
        best_new_L = best_L
        best_change = None
        
        # Sort not_covering by priority: smaller primes first (more frequent coverage)
        not_covering_sorted = sorted(not_covering, key=lambda i: primes[i])
        
        # Try top candidates
        for i in not_covering_sorted[:20]:
            old_r = best_residues[i]
            new_r = gap_pos % primes[i]
            if new_r == old_r:
                continue
            best_residues[i] = new_r
            new_L = _score(best_residues, primes)
            best_residues[i] = old_r
            if new_L > best_new_L:
                best_new_L = new_L
                best_change = (i, new_r)
        
        if best_change is not None:
            i, new_r = best_change
            best_residues[i] = new_r
            best_L = best_new_L
        else:
            # Try changing a covering prime to also cover gap_pos+1 (extend further)
            # Or try random changes near the gap
            improved = False
            for i in not_covering_sorted[:5]:
                old_r = best_residues[i]
                # Try a few residues around gap_pos
                for offset in [0, 1, -1, 2, -2]:
                    target = gap_pos + offset
                    if target < 1:
                        continue
                    new_r = target % primes[i]
                    if new_r == old_r:
                        continue
                    best_residues[i] = new_r
                    new_L = _score(best_residues, primes)
                    best_residues[i] = old_r
                    if new_L > best_L:
                        best_residues[i] = new_r
                        best_L = new_L
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    return best_residues, best_L

def _hill_climb(residues, primes, current_L, time_budget=60):
    """
    Hill climbing with targeted candidate generation.
    """
    start_time = time.time()
    n_primes = len(primes)
    rng = np.random.RandomState(42)
    best_residues = residues.copy()
    best_L = current_L
    no_improve = 0
    max_no_improve = 200
    
    while no_improve < max_no_improve:
        if time.time() - start_time > time_budget:
            break
        if best_L >= 100000:
            break
        
        # Pick a prime to modify
        # Bias towards primes that are "active" near the frontier
        i = rng.randint(n_primes)
        p = primes[i]
        old_r = best_residues[i]
        
        # Generate candidates
        candidates = set()
        candidates.add(old_r)
        
        # Target the frontier
        frontier = best_L + 1
        if frontier <= 100000:
            candidates.add(frontier % p)
            for offset in [-3, -2, -1, 1, 2, 3]:
                t = frontier + offset
                if 1 <= t <= 100000:
                    candidates.add(t % p)
        
        # Random candidates
        for _ in range(5):
            candidates.add(rng.randint(p))
        
        # Try each candidate
        best_new_r = old_r
        best_new_L = best_L
        
        for new_r in candidates:
            if new_r == old_r:
                continue
            best_residues[i] = new_r
            # Use fast prefix score for speed
            check_len = min(best_L + 200, 100000)
            s = _fast_score_prefix(best_residues, primes, check_len)
            if s > best_new_L:
                # Verify with full score
                full_s = _score(best_residues, primes)
                if full_s > best_new_L:
                    best_new_L = full_s
                    best_new_r = new_r
            best_residues[i] = old_r
        
        if best_new_r != old_r:
            best_residues[i] = best_new_r
            best_L = best_new_L
            no_improve = 0
        else:
            no_improve += 1
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Phase 1: Greedy frontier initialization
    residues = _greedy_frontier_init(primes, window=5000)
    current_L = _score(residues, primes)
    best_residues = residues.copy()
    best_L = current_L
    
    if time.time() > deadline - 5:
        return [int(r) for r in best_residues]
    
    # Phase 2: Targeted repair
    residues, current_L = _targeted_repair(residues, primes, current_L, time_budget=25)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues.copy()
    
    if time.time() > deadline - 5:
        return [int(r) for r in best_residues]
    
    # Phase 3: Hill climbing
    residues, current_L = _hill_climb(residues, primes, current_L, time_budget=40)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues.copy()
    
    if time.time() > deadline - 5:
        return [int(r) for r in best_residues]
    
    # Phase 4: Multiple restarts with perturbation
    rng = np.random.RandomState(123)
    remaining_time = deadline - time.time()
    n_restarts = max(1, int(remaining_time / 15))
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Perturb best solution
        current = best_residues.copy()
        n_change = max(3, n_primes // 15)
        idx = rng.choice(n_primes, n_change, replace=False)
        for i in idx:
            current[i] = rng.randint(primes[i])
        
        current_L = _score(current, primes)
        
        # Quick hill climb
        remaining = deadline - time.time() - 5
        if remaining > 5:
            current, current_L = _hill_climb(current, primes, current_L, time_budget=remaining)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current.copy()
    
    # Final targeted repair on best
    if time.time() < deadline - 10:
        final_residues, final_L = _targeted_repair(best_residues, primes, best_L, time_budget=deadline - time.time() - 5)
        if final_L > best_L:
            best_L = final_L
            best_residues = final_residues
    
    return [int(r) for r in best_residues]
