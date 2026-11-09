import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Evaluation range
    MAX_EVAL = 10000
    
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)
    
    def compute_coverage(residues):
        """Compute coverage array using vectorized operations."""
        coverage = np.zeros(MAX_EVAL, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
                if first < 1:
                    first = r + p
            if first <= MAX_EVAL:
                coverage[first-1::p] += 1
        return coverage

    def get_score(coverage):
        """Find the largest L such that coverage[0:L] >= 2."""
        mask = coverage < 2
        if not np.any(mask):
            return MAX_EVAL
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    def apply_change(coverage, i, old_r, new_r):
        """Apply a single residue change to coverage."""
        p = primes[i]
        if old_r == 0:
            first_old = p
        else:
            first_old = old_r
            if first_old < 1:
                first_old = old_r + p
        if first_old <= MAX_EVAL:
            coverage[first_old-1::p] -= 1
        
        if new_r == 0:
            first_new = p
        else:
            first_new = new_r
            if first_new < 1:
                first_new = new_r + p
        if first_new <= MAX_EVAL:
            coverage[first_new-1::p] += 1
    
    def revert_change(coverage, i, old_r, new_r):
        """Revert a single residue change."""
        p = primes[i]
        if new_r == 0:
            first_new = p
        else:
            first_new = new_r
            if first_new < 1:
                first_new = new_r + p
        if first_new <= MAX_EVAL:
            coverage[first_new-1::p] -= 1
        
        if old_r == 0:
            first_old = p
        else:
            first_old = old_r
            if first_old < 1:
                first_old = old_r + p
        if first_old <= MAX_EVAL:
            coverage[first_old-1::p] += 1

    # Phase 1: Greedy initialization with focus on small primes
    residues = np.zeros(n_primes, dtype=np.int64)
    coverage = np.zeros(MAX_EVAL, dtype=np.int32)
    
    # Process small primes first (they have more impact)
    prime_order = sorted(range(n_primes), key=lambda i: primes[i])
    
    for idx in prime_order:
        p = primes[idx]
        best_r = 0
        best_gain = -1
        for r in range(p):
            if r == 0:
                first = p
            else:
                first = r
                if first < 1:
                    first = r + p
            if first > 1000:
                continue
            count = 0
            x = first
            while x <= 1000:
                if coverage[x-1] < 2:
                    count += 1
                x += p
            if count > best_gain:
                best_gain = count
                best_r = r
        residues[idx] = best_r
        apply_change(coverage, idx, 0 if best_r == 0 else best_r, best_r) if best_r != 0 else None
        # Actually need to apply from scratch for first prime
        # Let me redo this more carefully
    
    # Redo Phase 1 more carefully
    residues = np.zeros(n_primes, dtype=np.int64)
    coverage = np.zeros(MAX_EVAL, dtype=np.int32)
    
    for idx in prime_order:
        p = primes[idx]
        best_r = 0
        best_gain = -1
        for r in range(p):
            if r == 0:
                first = p
            else:
                first = r
                if first < 1:
                    first = r + p
            if first > 1000:
                continue
            count = 0
            x = first
            while x <= 1000:
                if coverage[x-1] < 2:
                    count += 1
                x += p
            if count > best_gain:
                best_gain = count
                best_r = r
        residues[idx] = best_r
        # Apply the change
        apply_change(coverage, idx, 0, best_r) if best_r == 0 else apply_change(coverage, idx, best_r, best_r)
        # Wait, this is wrong. Let me just set it directly
        coverage = np.zeros(MAX_EVAL, dtype=np.int32)
        for j in range(n_primes):
            if residues[j] != 0 or j == idx:
                rj = residues[j]
                pj = primes[j]
                if rj == 0:
                    first_j = pj
                else:
                    first_j = rj
                    if first_j < 1:
                        first_j = rj + pj
                if first_j <= MAX_EVAL:
                    coverage[first_j-1::pj] += 1
        # This is too slow. Let me just track it properly
    
    # Let me use a cleaner approach
    residues = np.zeros(n_primes, dtype=np.int64)
    coverage = np.zeros(MAX_EVAL, dtype=np.int32)
    
    for idx in prime_order:
        p = primes[idx]
        best_r = 0
        best_gain = -1
        for r in range(p):
            if r == 0:
                first = p
            else:
                first = r
                if first < 1:
                    first = r + p
            if first > 1000:
                continue
            count = 0
            x = first
            while x <= 1000:
                if coverage[x-1] < 2:
                    count += 1
                x += p
            if count > best_gain:
                best_gain = count
                best_r = r
        residues[idx] = best_r
        # Apply
        if best_r == 0:
            first = p
        else:
            first = best_r
            if first < 1:
                first = best_r + p
        if first <= MAX_EVAL:
            coverage[first-1::p] += 1
    
    current_score = get_score(coverage)
    best_score = current_score
    best_residues = residues.copy()
    
    # Phase 2: Simulated Annealing with incremental updates
    T = 100.0
    T_min = 0.5
    cooling_rate = 0.9995
    
    # Precompute for speed: for each prime, the indices it affects
    prime_indices = {}
    for i in range(n_primes):
        p = primes[i]
        idxs = []
        for x in range(1, MAX_EVAL + 1):
            if x % p == residues[i] % p:  # This is wrong, residue matters
                pass
        # Actually, the indices depend on the residue, so we can't precompute easily
        # Just use the slice approach
    
    step = 0
    while time.time() < deadline and T > T_min:
        if time.time() > deadline - 5:
            break
        
        # Pick a random prime to change
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Apply change
        apply_change(coverage, i, old_r, new_r)
        residues[i] = new_r
        
        new_score = get_score(coverage)
        delta = new_score - current_score
        
        # Metropolis acceptance
        if delta >= 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
        else:
            # Accept with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current_score = new_score
            else:
                # Revert
                revert_change(coverage, i, old_r, new_r)
                residues[i] = old_r
        
        T *= cooling_rate
        step += 1
    
    # Phase 3: Conflict-driven repair at the end
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Find first gap and try to fix
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            mask = coverage < 2
            if not np.any(mask):
                break
            
            gap_idx = np.argmax(mask)
            gap_x = int(xs[gap_idx])
            
            # Try to fix by changing a prime that doesn't currently cover gap_x
            fixed = False
            for i in range(n_primes):
                p = primes[i]
                old_r = residues[i]
                new_r = gap_x % p
                
                if new_r == old_r:
                    continue
                
                # Check if this prime covers gap_x currently
                if (gap_x % p) == (old_r if old_r != 0 else 0):
                    # Wait, if old_r == 0, it covers multiples of p
                    # gap_x % p == 0 means it covers gap_x
                    if old_r == 0 and gap_x % p == 0:
                        continue
                    elif old_r != 0 and gap_x % p == old_r:
                        continue
                # Actually, let me just check
                if old_r == 0:
                    covers_old = (gap_x % p == 0)
                else:
                    covers_old = (gap_x % p == old_r)
                
                if covers_old:
                    continue  # Changing would lose coverage at gap_x
                
                # Try this change
                apply_change(coverage, i, old_r, new_r)
                residues[i] = new_r
                new_score = get_score(coverage)
                
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                    fixed = True
                    break
                else:
                    revert_change(coverage, i, old_r, new_r)
                    residues[i] = old_r
            
            if not fixed:
                # Try random perturbation
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                p = primes[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                apply_change(coverage, i, old_r, new_r)
                residues[i] = new_r
                new_score = get_score(coverage)
                
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                else:
                    revert_change(coverage, i, old_r, new_r)
                    residues[i] = old_r
    
    return best_residues.tolist()
