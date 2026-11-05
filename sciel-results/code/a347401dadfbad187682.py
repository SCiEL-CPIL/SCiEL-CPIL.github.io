"""
Constraint-satisfaction approach for the doubly-covered prefix problem.
Uses large primes as flexible variables to satisfy coverage debt, with
backtracking and forward-checking for a globally consistent solution.
"""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _build_count_array(residues, primes, max_L):
    """Build coverage count array for positions 1..max_L."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(len(primes)):
        r = residues[i]
        p = primes[i]
        start = r if r > 0 else p
        if start <= max_L:
            counts[start::p] += 1
    return counts

def _find_L(counts, current_L, max_L):
    """Find the largest L such that all positions 1..L have count >= 2."""
    if current_L > 0:
        if np.min(counts[1:current_L + 1]) < 2:
            bad = np.where(counts[1:current_L + 1] < 2)[0]
            return int(bad[0])
        L = current_L
        if L < max_L:
            rest = counts[L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return max_L
            else:
                return L + int(bad_idx[0])
    else:
        bad_idx = np.where(counts[1:] < 2)[0]
        if len(bad_idx) == 0:
            return max_L
        else:
            return int(bad_idx[0])

def _score(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered at least twice."""
    counts = _build_count_array(residues, primes, max_L)
    return _find_L(counts, 0, max_L)

def _greedy_construct(primes, max_L_check=5000):
    """
    Forward greedy: process primes in order, for each prime choose the residue
    that maximizes coverage of currently-deficient positions.
    """
    n = len(primes)
    residues = [0] * n
    counts = np.zeros(max_L_check + 1, dtype=np.int32)
    
    for idx in range(n):
        p = primes[idx]
        best_r = 0
        best_score = -1
        
        # For small primes, try all residues; for large primes, sample
        if p <= 200:
            candidates = range(p)
        else:
            candidates = np.random.default_rng(idx).choice(p, size=min(p, 80), replace=False)
        
        for r in candidates:
            start = r if r > 0 else p
            if start > max_L_check:
                continue
            positions = np.arange(start, max_L_check + 1, p)
            deficient = np.sum(counts[positions] < 2)
            already_covered = np.sum(counts[positions] >= 2)
            score = deficient - 0.3 * already_covered
            if score > best_score:
                best_score = score
                best_r = int(r)
        
        residues[idx] = best_r
        start = best_r if best_r > 0 else p
        if start <= max_L_check:
            counts[start::p] += 1
    
    return residues

def _csp_solve(residues, primes, max_L, deadline):
    """
    Constraint satisfaction: use large primes as flexible variables to satisfy
    coverage debt. Process positions from the first gap, and for each deficient
    position, find a large prime whose residue can be changed to cover it
    without breaking the prefix.
    """
    n = len(primes)
    counts = _build_count_array(residues, primes, max_L)
    current_L = _find_L(counts, 0, max_L)
    
    # Separate small and large primes
    small_indices = [i for i in range(n) if primes[i] <= 100]
    large_indices = [i for i in range(n) if primes[i] > 100]
    
    # Sort large primes by size (largest first for more flexibility)
    large_indices.sort(key=lambda i: primes[i], reverse=True)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        
        if current_L >= max_L:
            break
        
        # Find the first deficient position
        if current_L + 1 > max_L:
            break
        remaining = counts[current_L + 1:]
        bad_idx = np.where(remaining < 2)[0]
        if len(bad_idx) == 0:
            current_L = max_L
            break
        target_pos = current_L + 1 + int(bad_idx[0])
        
        # Try to fix target_pos using large primes
        fixed = False
        for idx in large_indices:
            if time.time() > deadline - 3:
                break
            if fixed:
                break
            
            p = primes[idx]
            old_r = residues[idx]
            needed_r = target_pos % p
            
            if needed_r == old_r:
                continue
            
            # Try changing to needed_r
            old_start = old_r if old_r > 0 else p
            new_start = needed_r if needed_r > 0 else p
            
            # Check if this would break the prefix
            # Remove old coverage, add new coverage
            if old_start <= max_L:
                counts[old_start::p] -= 1
            if new_start <= max_L:
                counts[new_start::p] += 1
            
            new_L = _find_L(counts, min(current_L, target_pos - 1), max_L)
            
            if new_L > current_L:
                residues[idx] = needed_r
                current_L = new_L
                improved = True
                fixed = True
            else:
                # Revert
                if new_start <= max_L:
                    counts[new_start::p] -= 1
                if old_start <= max_L:
                    counts[old_start::p] += 1
        
        # If large primes couldn't fix it, try small primes (last resort)
        if not fixed:
            for idx in small_indices[:20]:
                if time.time() > deadline - 3:
                    break
                if fixed:
                    break
                
                p = primes[idx]
                old_r = residues[idx]
                needed_r = target_pos % p
                
                if needed_r == old_r:
                    continue
                
                old_start = old_r if old_r > 0 else p
                new_start = needed_r if needed_r > 0 else p
                
                if old_start <= max_L:
                    counts[old_start::p] -= 1
                if new_start <= max_L:
                    counts[new_start::p] += 1
                
                new_L = _find_L(counts, min(current_L, target_pos - 1), max_L)
                
                if new_L > current_L:
                    residues[idx] = needed_r
                    current_L = new_L
                    improved = True
                    fixed = True
                else:
                    if new_start <= max_L:
                        counts[new_start::p] -= 1
                    if old_start <= max_L:
                        counts[old_start::p] += 1
    
    return residues, current_L

def _local_search(residues, primes, max_L, deadline):
    """
    Local search: for each prime, try to find a better residue by evaluating
    the impact on the doubly-covered prefix length.
    """
    n = len(primes)
    counts = _build_count_array(residues, primes, max_L)
    current_L = _find_L(counts, 0, max_L)
    best_L = current_L
    best_residues = residues.copy()
    
    # Process primes in random order, multiple passes
    rng = np.random.default_rng(42)
    order = list(range(n))
    
    for pass_num in range(5):
        if time.time() > deadline - 5:
            break
        
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            p = primes[idx]
            old_r = residues[idx]
            
            # Generate candidate residues
            if p <= 100:
                candidates = list(range(p))
            else:
                # Sample candidates around the current residue and target positions
                candidates = [old_r]
                # Add some random candidates
                for _ in range(30):
                    candidates.append(int(rng.integers(p)))
                # Remove duplicates
                candidates = list(set(candidates))
            
            best_r = old_r
            best_local_L = -1
            
            # Evaluate current residue
            old_start = old_r if old_r > 0 else p
            if old_start <= max_L:
                counts[old_start::p] -= 1
            
            for new_r in candidates:
                if time.time() > deadline - 5:
                    break
                
                new_start = new_r if new_r > 0 else p
                if new_start <= max_L:
                    counts[new_start::p] += 1
                
                new_L = _find_L(counts, min(current_L, max_L), max_L)
                
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_r = new_r
                
                if new_start <= max_L:
                    counts[new_start::p] -= 1
            
            # Restore old coverage
            if old_start <= max_L:
                counts[old_start::p] += 1
            
            # Apply best
            if best_r != old_r:
                residues[idx] = best_r
                new_start = best_r if best_r > 0 else p
                if new_start <= max_L:
                    counts[new_start::p] += 1
                if old_start <= max_L:
                    counts[old_start::p] -= 1
                current_L = _find_L(counts, 0, max_L)
            
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
    
    return best_residues, best_L

def _simulated_annealing(residues, primes, counts, current_L, max_L, deadline):
    """
    Simulated annealing to escape local optima.
    """
    n = len(primes)
    best_L = current_L
    best_residues = residues.copy()
    
    T = 5.0
    cooling = 0.99995
    min_T = 0.05
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline:
        # Pick random prime
        i = int(rng.integers(n))
        p = primes[i]
        old_r = residues[i]
        new_r = int(rng.integers(p))
        
        if new_r == old_r:
            continue
        
        # Apply change
        old_start = old_r if old_r > 0 else p
        if old_start <= max_L:
            counts[old_start::p] -= 1
        
        new_start = new_r if new_r > 0 else p
        if new_start <= max_L:
            counts[new_start::p] += 1
        
        # Calculate new L
        new_L = _find_L(counts, current_L, max_L)
        
        # Acceptance
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            if delta > -200:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    residues[i] = new_r
                    current_L = new_L
                else:
                    # Revert
                    residues[i] = old_r
                    counts[new_start::p] -= 1
                    counts[old_start::p] += 1
            else:
                # Revert
                residues[i] = old_r
                counts[new_start::p] -= 1
                counts[old_start::p] += 1
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _get_primes(1000)
    n = len(primes)
    MAX_L = 100000
    
    # Phase 1: Greedy construction
    residues = _greedy_construct(primes, max_L_check=5000)
    
    # Build full count array
    counts = _build_count_array(residues, primes, MAX_L)
    current_L = _find_L(counts, 0, MAX_L)
    
    # Phase 2: CSP solving with large primes as flexible variables
    if time.time() < deadline - 30:
        residues, current_L = _csp_solve(residues, primes, MAX_L, deadline - 25)
    
    # Rebuild counts
    counts = _build_count_array(residues, primes, MAX_L)
    current_L = _find_L(counts, 0, MAX_L)
    
    # Phase 3: Local search
    if time.time() < deadline - 20:
        residues, current_L = _local_search(residues, primes, MAX_L, deadline - 15)
    
    # Rebuild counts
    counts = _build_count_array(residues, primes, MAX_L)
    current_L = _find_L(counts, 0, MAX_L)
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5:
        residues, current_L = _simulated_annealing(residues, primes, counts, current_L, MAX_L, deadline)
    
    # Final evaluation
    final_score = _score(residues, primes, MAX_L)
    
    return residues

