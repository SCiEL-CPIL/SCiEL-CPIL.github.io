"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _compute_prefix_score(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L satisfies at least 2 congruences."""
    counts = [0] * (max_L + 1)
    for p, r in zip(primes, residues):
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        for x in range(start, max_L + 1, p):
            counts[x] += 1
    
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _build_counts(residues, primes, max_L):
    """Build count array efficiently."""
    counts = [0] * (max_L + 1)
    for p, r in zip(primes, residues):
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        for x in range(start, max_L + 1, p):
            counts[x] += 1
    return counts

def _find_prefix_length(counts, max_L):
    """Find prefix length from counts array."""
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Working range for optimization
    WORK_L = 1500
    
    # Phase 1: Smart initialization
    # For small primes, spread residues to maximize coverage
    residues = [0] * n_primes
    
    # Initialize small primes with spreading strategy
    # For prime p, choose r to maximize coverage of [1, WORK_L]
    # Use a greedy approach but with better initialization
    
    # Start with all zeros, then improve
    counts = [0] * (WORK_L + 1)
    
    # Process primes in order of increasing size (small primes first)
    # But use a smarter criterion: prioritize primes that create gaps
    
    # Better initialization: for each prime, choose residue that maximizes
    # the minimum count over [1, WORK_L]
    
    # First pass: simple greedy
    order = list(range(n_primes))
    # Sort by prime value (small first)
    order.sort(key=lambda i: primes[i])
    
    for idx in order:
        p = primes[idx]
        best_r = 0
        best_metric = -1
        
        # For each candidate residue, compute a metric
        # Metric: sum of (1 if count[x] < 2 else 0) for x ≡ r (mod p)
        # Plus a bonus for reducing max deficit
        
        if p <= WORK_L:
            for r in range(p):
                gain = 0
                start_x = r if r >= 1 else r + p
                for x in range(start_x, WORK_L + 1, p):
                    if counts[x] < 2:
                        gain += 1
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
        else:
            # p > WORK_L: only one integer per residue
            best_r = 0
            best_metric = -1
            for r in range(1, min(p, WORK_L + 1)):
                gain = 1 if counts[r] < 2 else 0
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
            if best_metric <= 0:
                best_r = 0
        
        residues[idx] = best_r
        
        # Update counts
        start_x = best_r if best_r >= 1 else best_r + p
        for x in range(start_x, WORK_L + 1, p):
            counts[x] += 1
    
    # Get initial score
    current_score = _find_prefix_length(counts, WORK_L)
    
    # Phase 2: Simulated Annealing
    # Temperature schedule
    T_initial = 10.0
    T_final = 0.1
    cooling_rate = 0.999
    
    T = T_initial
    best_residues = residues[:]
    best_score = current_score
    
    # Count array for fast evaluation
    counts = _build_counts(residues, primes, WORK_L)
    current_score = _find_prefix_length(counts, WORK_L)
    
    max_iterations = 500000
    iteration = 0
    
    while time.time() < deadline - 10.0 and iteration < max_iterations:
        iteration += 1
        
        # Choose a prime to modify
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        
        # Choose new residue
        if p <= 50:
            new_r = random.randint(0, p - 1)
        else:
            # For large primes, try residues near the bottleneck
            # Find first gap
            new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Compute delta score
        # Remove old coverage, add new coverage
        delta = 0
        
        # For x ≡ old_r (mod p): count decreases by 1
        # For x ≡ new_r (mod p): count increases by 1
        # These sets are disjoint (since old_r != new_r and both < p)
        
        # Effect on prefix:
        # - If any x in old_r-set had count exactly 2, it becomes 1 (bad if in prefix)
        # - If any x in new_r-set had count exactly 1, it becomes 2 (good if in prefix)
        
        # For efficiency, only check up to current_score + some margin
        check_limit = min(WORK_L, current_score + 200)
        
        # Count losses: x in old_r-set with count == 2 and x <= current_score
        loss = 0
        start_old = old_r if old_r >= 1 else old_r + p
        for x in range(start_old, check_limit + 1, p):
            if counts[x] == 2 and x <= current_score:
                loss += 1
        
        # Count gains: x in new_r-set with count == 1 and x <= current_score
        gain = 0
        start_new = new_r if new_r >= 1 else new_r + p
        for x in range(start_new, check_limit + 1, p):
            if counts[x] == 1 and x <= current_score:
                gain += 1
        
        # Also check if this extends the prefix
        # New prefix could extend if we fill a gap at current_score + 1
        new_prefix = current_score
        if loss == 0:
            # Check if we can extend
            x = current_score + 1
            if x <= WORK_L:
                # Check if x is now covered by at least 2
                # x's count changes only if x ≡ old_r or new_r (mod p)
                old_count_x = counts[x]
                if x >= start_old and (x - start_old) % p == 0:
                    old_count_x -= 1
                if x >= start_new and (x - start_new) % p == 0:
                    old_count_x += 1
                if old_count_x >= 2:
                    # Try to extend
                    new_prefix = x
                    while new_prefix + 1 <= WORK_L:
                        x2 = new_prefix + 1
                        c = counts[x2]
                        if x2 >= start_old and (x2 - start_old) % p == 0:
                            c -= 1
                        if x2 >= start_new and (x2 - start_new) % p == 0:
                            c += 1
                        if c >= 2:
                            new_prefix = x2
                        else:
                            break
        
        new_score = new_prefix
        
        # Acceptance criterion
        if new_score > current_score:
            # Apply move
            # Update counts
            start_old = old_r if old_r >= 1 else old_r + p
            for x in range(start_old, WORK_L + 1, p):
                counts[x] -= 1
            start_new = new_r if new_r >= 1 else new_r + p
            for x in range(start_new, WORK_L + 1, p):
                counts[x] += 1
            residues[idx] = new_r
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
        else:
            # Metropolis acceptance
            delta_score = new_score - current_score
            if delta_score >= 0 or random.random() < math.exp(delta_score / T):
                # Apply move
                start_old = old_r if old_r >= 1 else old_r + p
                for x in range(start_old, WORK_L + 1, p):
                    counts[x] -= 1
                start_new = new_r if new_r >= 1 else new_r + p
                for x in range(start_new, WORK_L + 1, p):
                    counts[x] += 1
                residues[idx] = new_r
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_final:
            T = T_final
    
    # Use best found
    residues = best_residues
    
    # Phase 3: Local search refinement
    # Try to extend the prefix by targeted moves
    counts = _build_counts(residues, primes, WORK_L)
    current_score = _find_prefix_length(counts, WORK_L)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Find the first gap
        if current_score >= WORK_L:
            break
        
        gap_pos = current_score + 1
        if gap_pos > WORK_L:
            break
        
        # Find which primes cover gap_pos
        covering = []
        for i, (p, r) in enumerate(zip(primes, residues)):
            if gap_pos % p == r % p:
                covering.append(i)
        
        # We need at least 2 covering. If we have 0 or 1, try to add more
        if len(covering) < 2:
            # Try to change a non-covering prime to cover gap_pos
            for i in range(n_primes):
                if i in covering:
                    continue
                p = primes[i]
                old_r = residues[i]
                # Residue needed: gap_pos mod p
                new_r = gap_pos % p
                if new_r == old_r:
                    continue
                
                # Check if this move helps
                # Remove old, add new
                # Only check if it doesn't break the prefix
                loss = 0
                start_old = old_r if old_r >= 1 else old_r + p
                for x in range(start_old, current_score + 1, p):
                    if counts[x] == 2:
                        loss += 1
                
                if loss == 0:
                    # Apply
                    for x in range(start_old, WORK_L + 1, p):
                        counts[x] -= 1
                    start_new = new_r if new_r >= 1 else new_r + p
                    for x in range(start_new, WORK_L + 1, p):
                        counts[x] += 1
                    residues[i] = new_r
                    improved = True
                    break
        
        # Recompute score
        if improved:
            current_score = _find_prefix_length(counts, WORK_L)
    
    # Final verification with larger range if time allows
    if time.time() < deadline - 2.0:
        # Check if we can extend beyond WORK_L
        extended = _compute_prefix_score(residues, primes, min(10000, WORK_L * 3))
        # We don't update residues, just for information
    
    return residues
# EVOLVE-BLOCK-END
