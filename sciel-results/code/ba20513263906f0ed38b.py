import numpy as np
import time

def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.array([i for i in range(n + 1) if sieve[i]])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)  # 168
    
    # Search bound: we want L >= 1000, search up to 2000 for margin
    SEARCH_BOUND = 2000
    
    # Initial residues: try a few different initializations
    # Strategy: use residues that spread coverage evenly
    def initial_residues(strategy):
        residues = np.zeros(n_primes, dtype=np.int64)
        if strategy == 0:
            # Seed-like: cover ceil(i/2)
            for i, p in enumerate(primes):
                residues[i] = ((i + 2) // 2) % p
        elif strategy == 1:
            # Random
            residues = np.random.randint(0, primes, size=n_primes)
        elif strategy == 2:
            # Try to make small integers covered multiple times
            # Assign residues so that small numbers get hit
            for i, p in enumerate(primes):
                # Try residue that hits position 1, 2, etc.
                # x = r (mod p) hits x if r = x % p
                # Assign residue = (i+1) % p to hit position i+1
                residues[i] = (i + 1) % p
        elif strategy == 3:
            # Greedy: for each prime, pick residue that maximizes coverage of current vulnerable set
            residues = np.zeros(n_primes, dtype=np.int64)
            # Start with all zeros, then greedily improve
            for i in range(n_primes):
                best_r = 0
                best_score = -1
                for r in range(primes[i]):
                    # Quick estimate: how many of 1..SEARCH_BOUND does this cover that aren't already well-covered
                    # Just pick r = i % p for now
                    pass
                residues[i] = i % primes[i]
        return residues
    
    def compute_coverage(residues):
        """Compute coverage count for each integer 1..SEARCH_BOUND."""
        counts = np.zeros(SEARCH_BOUND + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            # Positions: r, r+p, r+2p, ... up to SEARCH_BOUND
            # First position >= 1: if r == 0, start at p; else start at r
            if r == 0:
                start = p
            else:
                start = r
            if start <= SEARCH_BOUND:
                counts[start::p] += 1
        return counts
    
    def compute_L(counts):
        """Find largest L such that all integers 1..L have count >= 2."""
        # Find first position where count < 2
        # counts[1..SEARCH_BOUND]
        if SEARCH_BOUND < 1:
            return 0
        # We need counts[i] >= 2 for all i in 1..L
        # Find first i where counts[i] < 2
        idx = np.where(counts[1:SEARCH_BOUND+1] < 2)[0]
        if len(idx) == 0:
            return SEARCH_BOUND
        return idx[0]  # first index (0-based in the slice) corresponds to position idx[0]+1, so L = idx[0]
    
    def evaluate(residues):
        counts = compute_coverage(residues)
        L = compute_L(counts)
        # Also track the minimum coverage in 1..L+1 for tie-breaking
        return L, counts
    
    # Try multiple initial strategies and keep the best
    best_L = 0
    best_residues = None
    best_counts = None
    
    for strategy in range(4):
        residues = initial_residues(strategy)
        L, counts = evaluate(residues)
        if L > best_L:
            best_L = L
            best_residues = residues.copy()
            best_counts = counts.copy()
    
    # Simulated annealing
    if best_L < SEARCH_BOUND and time.time() < deadline - 10:
        current_residues = best_residues.copy()
        current_counts = best_counts.copy()
        current_L = best_L
        
        # Track vulnerable set: positions with count == 1
        # We want to eliminate vulnerable positions in the prefix
        
        T = 50.0  # initial temperature
        cooling = 0.995
        min_T = 0.1
        
        rng = np.random.default_rng(42)
        
        while time.time() < deadline - 5 and T > min_T:
            # Pick a random prime index to change
            i = rng.integers(0, n_primes)
            p = primes[i]
            old_r = current_residues[i]
            
            # Pick a new random residue
            new_r = rng.integers(0, p)
            if new_r == old_r:
                continue
            
            # Compute delta in coverage
            # Old coverage: positions old_r, old_r+p, ...
            # New coverage: positions new_r, new_r+p, ...
            
            # Remove old coverage
            if old_r == 0:
                old_start = p
            else:
                old_start = old_r
            if old_start <= SEARCH_BOUND:
                old_positions = np.arange(old_start, SEARCH_BOUND + 1, p)
                current_counts[old_positions] -= 1
            
            # Add new coverage
            if new_r == 0:
                new_start = p
            else:
                new_start = new_r
            if new_start <= SEARCH_BOUND:
                new_positions = np.arange(new_start, SEARCH_BOUND + 1, p)
                current_counts[new_positions] += 1
            
            # Compute new L
            idx = np.where(current_counts[1:SEARCH_BOUND+1] < 2)[0]
            if len(idx) == 0:
                new_L = SEARCH_BOUND
            else:
                new_L = idx[0]
            
            # Acceptance criterion
            delta_L = new_L - current_L
            if delta_L > 0:
                # Always accept improvement
                current_residues[i] = new_r
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current_residues.copy()
                    best_counts = current_counts.copy()
            else:
                # Accept with probability exp(delta_L / T)
                if delta_L < 0:
                    prob = np.exp(delta_L / T)
                    if rng.random() < prob:
                        current_residues[i] = new_r
                        current_L = new_L
                # If delta_L == 0, reject (no improvement)
            
            T *= cooling
        
        # Local search on best solution
        if time.time() < deadline - 5:
            current_residues = best_residues.copy()
            current_counts = best_counts.copy()
            current_L = best_L
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # For each prime, try all residues and keep the best
                for i in range(n_primes):
                    if time.time() > deadline - 5:
                        break
                    p = primes[i]
                    old_r = current_residues[i]
                    best_r_for_i = old_r
                    best_L_for_i = current_L
                    
                    # Try a subset of residues if p is large
                    if p <= 50:
                        candidates = list(range(p))
                    else:
                        # Sample some residues
                        candidates = [old_r]
                        for _ in range(min(20, p)):
                            candidates.append(rng.integers(0, p))
                    
                    for r in candidates:
                        if r == old_r:
                            continue
                        
                        # Remove old
                        if old_r == 0:
                            old_start = p
                        else:
                            old_start = old_r
                        if old_start <= SEARCH_BOUND:
                            old_positions = np.arange(old_start, SEARCH_BOUND + 1, p)
                            current_counts[old_positions] -= 1
                        
                        # Add new
                        if r == 0:
                            new_start = p
                        else:
                            new_start = r
                        if new_start <= SEARCH_BOUND:
                            new_positions = np.arange(new_start, SEARCH_BOUND + 1, p)
                            current_counts[new_positions] += 1
                        
                        idx = np.where(current_counts[1:SEARCH_BOUND+1] < 2)[0]
                        if len(idx) == 0:
                            new_L = SEARCH_BOUND
                        else:
                            new_L = idx[0]
                        
                        if new_L > best_L_for_i:
                            best_L_for_i = new_L
                            best_r_for_i = r
                        
                        # Undo
                        if r == 0:
                            new_start = p
                        else:
                            new_start = r
                        if new_start <= SEARCH_BOUND:
                            new_positions = np.arange(new_start, SEARCH_BOUND + 1, p)
                            current_counts[new_positions] -= 1
                        
                        if old_r == 0:
                            old_start = p
                        else:
                            old_start = old_r
                        if old_start <= SEARCH_BOUND:
                            old_positions = np.arange(old_start, SEARCH_BOUND + 1, p)
                            current_counts[old_positions] += 1
                    
                    if best_r_for_i != old_r:
                        # Apply best
                        old_r = current_residues[i]
                        if old_r == 0:
                            old_start = p
                        else:
                            old_start = old_r
                        if old_start <= SEARCH_BOUND:
                            old_positions = np.arange(old_start, SEARCH_BOUND + 1, p)
                            current_counts[old_positions] -= 1
                        
                        if best_r_for_i == 0:
                            new_start = p
                        else:
                            new_start = best_r_for_i
                        if new_start <= SEARCH_BOUND:
                            new_positions = np.arange(new_start, SEARCH_BOUND + 1, p)
                            current_counts[new_positions] += 1
                        
                        current_residues[i] = best_r_for_i
                        current_L = best_L_for_i
                        improved = True
                        
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = current_residues.copy()
                            best_counts = current_counts.copy()
    
    # Final evaluation
    final_L, final_counts = evaluate(best_residues)
    return list(best_residues)

