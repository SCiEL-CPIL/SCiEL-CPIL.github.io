import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # Precompute for each prime: for each residue r, the slice indices
    # This is just start = r if r > 0 else p, then step p

    def build_counts(residues):
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def find_L(counts, current_L):
        """Find the largest L such that counts[1..L] >= 2."""
        if current_L >= MAX_L:
            if np.min(counts[1:MAX_L+1]) >= 2:
                return MAX_L
            # Find where it breaks
            bad = np.where(counts[1:MAX_L+1] < 2)[0]
            if len(bad) == 0:
                return MAX_L
            return bad[0]
        
        if current_L > 0:
            # Check if current prefix is still valid
            if np.min(counts[1:current_L+1]) < 2:
                bad = np.where(counts[1:current_L+1] < 2)[0]
                return bad[0]
            # Try to extend
            if current_L < MAX_L:
                rest = counts[current_L+1:]
                bad_idx = np.where(rest < 2)[0]
                if len(bad_idx) == 0:
                    return MAX_L
                else:
                    return current_L + bad_idx[0]
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return bad_idx[0]

    def fast_find_L(counts, current_L):
        """Faster L evaluation that uses current_L as anchor."""
        if current_L <= 0:
            return find_L(counts, 0)
        
        # First check if current_L is still valid
        if np.min(counts[1:current_L+1]) < 2:
            bad = np.where(counts[1:current_L+1] < 2)[0]
            return bad[0]
        
        # Try to extend from current_L
        if current_L >= MAX_L:
            return MAX_L
        
        rest = counts[current_L+1:]
        bad_idx = np.where(rest < 2)[0]
        if len(bad_idx) == 0:
            return MAX_L
        return current_L + bad_idx[0]

    # Phase 1: Smart greedy initialization
    def greedy_init():
        residues = [0] * n_primes
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        current_L = 0
        
        # Process primes in order of increasing size for better coverage
        # Actually, process in a strategic order: small primes first for dense coverage
        order = list(range(n_primes))
        
        for i in order:
            p = primes[i]
            best_r = 0
            best_L = -1
            
            # For small primes, try all residues
            # For larger primes, try strategic candidates
            if p <= 40:
                candidates = list(range(p))
            else:
                # Try residues that align with the frontier
                candidates = set()
                for offset in range(-5, 60):
                    target = current_L + offset
                    if target > 0:
                        candidates.add(target % p)
                candidates.add(0)
                # Add some random
                for _ in range(15):
                    candidates.add(random.randint(0, p-1))
                candidates = list(candidates)
            
            for r in candidates:
                start = r if r > 0 else p
                if start <= MAX_L:
                    counts[start::p] += 1
                    new_L = fast_find_L(counts, current_L)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    counts[start::p] -= 1
            
            start = best_r if best_r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
            residues[i] = best_r
            current_L = max(current_L, best_L)
        
        return residues, counts, current_L

    residues, counts, current_L = greedy_init()
    best_L = current_L
    best_residues = residues.copy()
    
    # Phase 2: Local search with multiple sweeps
    def local_search_sweep(residues, counts, current_L, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n_primes):
                if time.time() > max_time:
                    break
                p = primes[i]
                old_r = residues[i]
                best_r = old_r
                best_L = current_L
                
                # Remove old contribution
                start_old = old_r if old_r > 0 else p
                if start_old <= MAX_L:
                    counts[start_old::p] -= 1
                
                # Generate candidates
                if p <= 25:
                    candidates = list(range(p))
                else:
                    candidates = set()
                    # Near frontier
                    for offset in range(-15, 40):
                        target = current_L + offset
                        if target > 0:
                            candidates.add(target % p)
                    # Also try to cover sparse regions
                    candidates.add(0)
                    for _ in range(8):
                        candidates.add(random.randint(0, p-1))
                    candidates = list(candidates)
                
                for r in candidates:
                    if r == old_r:
                        continue
                    start_new = r if r > 0 else p
                    if start_new <= MAX_L:
                        counts[start_new::p] += 1
                    new_L = fast_find_L(counts, current_L)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                
                # Apply best
                if best_r != old_r:
                    start_new = best_r if best_r > 0 else p
                    if start_new <= MAX_L:
                        counts[start_new::p] += 1
                    residues[i] = best_r
                    current_L = best_L
                    improved = True
                else:
                    # Restore old
                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
        
        return residues, counts, current_L

    residues, counts, current_L = local_search_sweep(residues, counts, current_L, deadline - 30)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues.copy()
    
    # Phase 3: Simulated annealing with adaptive temperature
    if time.time() < deadline - 10:
        T = 500.0
        min_T = 0.1
        cooling = 0.99995
        
        # Track best
        sa_residues = residues.copy()
        sa_counts = counts.copy()
        sa_L = current_L
        
        while time.time() < deadline - 3:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = sa_residues[i]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply change
            start_old = old_r if old_r > 0 else p
            if start_old <= MAX_L:
                sa_counts[start_old::p] -= 1
            
            start_new = new_r if new_r > 0 else p
            if start_new <= MAX_L:
                sa_counts[start_new::p] += 1
            
            new_L = fast_find_L(sa_counts, sa_L)
            delta = new_L - sa_L
            
            if delta >= 0:
                sa_residues[i] = new_r
                sa_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = sa_residues.copy()
            else:
                # Accept with probability exp(delta/T)
                if delta > -5000:
                    prob = np.exp(delta / T)
                    if random.random() < prob:
                        sa_residues[i] = new_r
                        sa_L = new_L
                    else:
                        # Revert
                        sa_residues[i] = old_r
                        if start_new <= MAX_L:
                            sa_counts[start_new::p] -= 1
                        if start_old <= MAX_L:
                            sa_counts[start_old::p] += 1
                else:
                    # Revert
                    sa_residues[i] = old_r
                    if start_new <= MAX_L:
                        sa_counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        sa_counts[start_old::p] += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
                # Restart with higher temperature if time permits
                if time.time() < deadline - 20:
                    T = 200.0
                    # Restart from best
                    sa_residues = best_residues.copy()
                    sa_counts = build_counts(sa_residues)
                    sa_L = best_L
    
    # Phase 4: Final polish - targeted local search on best solution
    if time.time() < deadline - 2:
        final_residues = best_residues.copy()
        final_counts = build_counts(final_residues)
        final_L = best_L
        
        # Multiple passes of targeted improvement
        for sweep in range(3):
            if time.time() > deadline - 1:
                break
            for i in range(n_primes):
                if time.time() > deadline - 1:
                    break
                p = primes[i]
                old_r = final_residues[i]
                best_r = old_r
                best_L = final_L
                
                start_old = old_r if old_r > 0 else p
                if start_old <= MAX_L:
                    final_counts[start_old::p] -= 1
                
                # Try all residues for small primes, strategic for large
                if p <= 20:
                    candidates = list(range(p))
                else:
                    candidates = set()
                    for offset in range(-10, 30):
                        target = final_L + offset
                        if target > 0:
                            candidates.add(target % p)
                    candidates.add(0)
                    candidates = list(candidates)
                
                for r in candidates:
                    if r == old_r:
                        continue
                    start_new = r if r > 0 else p
                    if start_new <= MAX_L:
                        final_counts[start_new::p] += 1
                    new_L = fast_find_L(final_counts, final_L)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    if start_new <= MAX_L:
                        final_counts[start_new::p] -= 1
                
                if best_r != old_r:
                    start_new = best_r if best_r > 0 else p
                    if start_new <= MAX_L:
                        final_counts[start_new::p] += 1
                    final_residues[i] = best_r
                    final_L = best_L
                    if final_L > best_L:
                        best_L = final_L
                        best_residues = final_residues.copy()
                else:
                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_L:
                        final_counts[start_old::p] += 1
    
    return best_residues
