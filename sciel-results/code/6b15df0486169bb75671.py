import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    N = 1000
    x = np.arange(1, N + 1)
    
    def compute_L(residues):
        counts = np.zeros(N, dtype=int)
        for i, p in enumerate(primes):
            counts += (x % p == residues[i])
        is_fail = counts < 2
        if not np.any(is_fail):
            return N, counts
        b_idx = np.argmax(is_fail)
        return b_idx, counts
    
    def greedy_repair(residues, counts):
        """Fast greedy repair: fix the first bottleneck repeatedly."""
        L, counts = compute_L(residues)
        best_L = L
        best_residues = residues.copy()
        
        while L < N:
            target_x = L + 1
            improved = False
            
            # Try primes in random order, but prioritize smaller primes (more frequent)
            order = list(range(num_primes))
            random.shuffle(order)
            
            for i in order:
                p = primes[i]
                current_r = residues[i]
                new_r = target_x % p
                
                if current_r == new_r:
                    continue
                
                # Check safety: no x in 1..L has count==2 and is covered by current_r
                if L > 0:
                    mask_R_prefix = (x[:L] % p == current_r)
                    if np.any(mask_R_prefix & (counts[:L] == 2)):
                        continue
                
                # Apply move
                residues[i] = new_r
                mask_R = (x % p == current_r)
                mask_A = (x % p == new_r)
                counts -= mask_R
                counts += mask_A
                
                # Check new L
                is_fail = counts < 2
                if np.any(is_fail):
                    new_L = np.argmax(is_fail)
                else:
                    new_L = N
                
                if new_L > L:
                    L = new_L
                    best_L = L
                    best_residues = residues.copy()
                    improved = True
                    break
                else:
                    # Revert
                    residues[i] = current_r
                    counts -= mask_A
                    counts += mask_R
            
            if not improved:
                break
        
        return best_L, best_residues, counts
    
    # Initial residues - try several random starts
    best_overall_L = 0
    best_overall_residues = None
    best_overall_counts = None
    
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        residues = np.array([random.randint(0, p - 1) for p in primes], dtype=int)
        
        # Greedy repair
        L, residues, counts = greedy_repair(residues, None)
        
        if L > best_overall_L:
            best_overall_L = L
            best_overall_residues = residues.copy()
            best_overall_counts = counts.copy()
    
    # Simulated annealing phase
    if best_overall_residues is not None and time.time() < deadline - 10:
        residues = best_overall_residues.copy()
        counts = best_overall_counts.copy()
        L = best_overall_L
        
        temperature = 50.0
        cooling = 0.9995
        
        while time.time() < deadline - 5:
            if temperature < 0.1:
                break
            
            # Pick random prime
            i = random.randint(0, num_primes - 1)
            p = primes[i]
            current_r = residues[i]
            
            # Pick new residue
            new_r = random.randint(0, p - 1)
            if new_r == current_r:
                continue
            
            # Compute change in L
            mask_R = (x % p == current_r)
            mask_A = (x % p == new_r)
            
            old_counts = counts.copy()
            counts -= mask_R
            counts += mask_A
            
            is_fail = counts < 2
            if np.any(is_fail):
                new_L = np.argmax(is_fail)
            else:
                new_L = N
            
            delta_L = new_L - L
            
            # Acceptance criterion
            if delta_L > 0 or random.random() < np.exp(-delta_L / temperature):
                L = new_L
                residues[i] = new_r
                if L > best_overall_L:
                    best_overall_L = L
                    best_overall_residues = residues.copy()
                    best_overall_counts = counts.copy()
            else:
                # Revert
                counts = old_counts
            
            temperature *= cooling
            
            # Periodic repair restart
            if L < best_overall_L and random.random() < 0.01:
                residues = best_overall_residues.copy()
                counts = best_overall_counts.copy()
                L = best_overall_L
    
    if best_overall_residues is None:
        # Fallback: simple initialization
        residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=int)
    
    return [int(r) for r in best_overall_residues]
