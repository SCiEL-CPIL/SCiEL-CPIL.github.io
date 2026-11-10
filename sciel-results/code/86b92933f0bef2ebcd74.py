import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    # Sieve for primes up to 1000
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000 ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    primes = [i for i in range(2, 1001) if sieve[i]]
    n_primes = len(primes)
    
    # Helper: build count array for given residues up to limit
    def build_counts(residues, limit):
        counts = np.zeros(limit + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p:limit+1:p] += 1
            else:
                counts[r:limit+1:p] += 1
        return counts
    
    # Helper: find largest L such that all 1..L have count >= 2
    def get_L(counts, limit):
        if limit < 1:
            return 0
        below = np.where(counts[1:limit+1] < 2)[0]
        if len(below) == 0:
            return limit
        return int(below[0])
    
    # Smart initialization: greedy for small primes
    residues = [0] * n_primes
    L_INIT = 300
    count_init = np.zeros(L_INIT + 1, dtype=np.int32)
    
    # For small primes, greedily pick residue that covers most singly-covered numbers
    small_prime_count = min(30, n_primes)
    for i in range(small_prime_count):
        p = primes[i]
        best_r = 0
        best_score = -1
        for r in range(p):
            if r == 0:
                positions = np.arange(p, L_INIT + 1, p)
            else:
                if r > L_INIT:
                    continue
                positions = np.arange(r, L_INIT + 1, p)
            if len(positions) == 0:
                score = 0
            else:
                score = int(np.sum(count_init[positions] == 1))
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        r = best_r
        if r == 0:
            count_init[p:L_INIT+1:p] += 1
        else:
            count_init[r:L_INIT+1:p] += 1
    
    # For remaining primes, random
    for i in range(small_prime_count, n_primes):
        p = primes[i]
        residues[i] = random.randint(0, p - 1)
    
    # Initial M and counts
    M = 5000
    max_M = 100000
    counts = build_counts(residues, M)
    current_L = get_L(counts, M)
    best_L = current_L
    best_residues = residues[:]
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # SA parameters
    T = 8.0
    cooling = 0.9995
    min_T = 0.1
    
    stagnation_count = 0
    max_stagnation = 50
    
    while time.time() < deadline - 5:
        # Expand M if close to boundary
        if current_L >= M - 500:
            new_M = min(M * 2, max_M)
            if new_M > M:
                counts = build_counts(residues, new_M)
                M = new_M
                current_L = get_L(counts, M)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                continue
        
        # Greedy repair: find first gap and try to cover it
        gap = current_L + 1
        if gap <= M and counts[gap] < 2:
            # Try to cover gap by changing one residue
            best_move = None
            min_cost = float('inf')
            
            # Try smaller primes first (they affect more numbers)
            for i in range(n_primes):
                p = primes[i]
                old_r = residues[i]
                new_r = gap % p
                
                if old_r == new_r:
                    continue
                
                # Cost: how many x in [1, current_L] with count[x]==2 that we'd break
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                
                if start_old > current_L:
                    cost = 0
                else:
                    positions = np.arange(start_old, current_L + 1, p)
                    cost = int(np.sum(counts[positions] == 2))
                
                if cost < min_cost:
                    min_cost = cost
                    best_move = (i, new_r)
                    if cost == 0:
                        break
            
            if best_move is not None:
                i, new_r = best_move
                p = primes[i]
                old_r = residues[i]
                
                # Apply
                if old_r == 0:
                    counts[p:M+1:p] -= 1
                else:
                    counts[old_r:M+1:p] -= 1
                if new_r == 0:
                    counts[p:M+1:p] += 1
                else:
                    counts[new_r:M+1:p] += 1
                residues[i] = new_r
                
                new_L = get_L(counts, M)
                
                if new_L > current_L:
                    current_L = new_L
                    stagnation_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                else:
                    # Revert
                    if new_r == 0:
                        counts[p:M+1:p] -= 1
                    else:
                        counts[new_r:M+1:p] -= 1
                    if old_r == 0:
                        counts[p:M+1:p] += 1
                    else:
                        counts[old_r:M+1:p] += 1
                    residues[i] = old_r
                    stagnation_count += 1
            else:
                stagnation_count += 1
        else:
            # No gap found or gap beyond M, do SA step
            stagnation_count += 1
        
        # SA burst when stagnant
        if stagnation_count > max_stagnation:
            sa_start = time.time()
            sa_T = max(T, 5.0)
            sa_cooling = 0.99
            sa_min_T = 0.5
            
            while time.time() < sa_start + 3.0 and sa_T > sa_min_T and time.time() < deadline - 5:
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                # Apply
                if old_r == 0:
                    counts[p:M+1:p] -= 1
                else:
                    counts[old_r:M+1:p] -= 1
                if new_r == 0:
                    counts[p:M+1:p] += 1
                else:
                    counts[new_r:M+1:p] += 1
                residues[idx] = new_r
                
                new_L = get_L(counts, M)
                delta = new_L - current_L
                
                if delta > 0:
                    current_L = new_L
                    stagnation_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                elif delta == 0:
                    pass
                else:
                    prob = np.exp(delta / sa_T)
                    if random.random() < prob:
                        current_L = new_L
                    else:
                        # Revert
                        if new_r == 0:
                            counts[p:M+1:p] -= 1
                        else:
                            counts[new_r:M+1:p] -= 1
                        if old_r == 0:
                            counts[p:M+1:p] += 1
                        else:
                            counts[old_r:M+1:p] += 1
                        residues[idx] = old_r
                
                sa_T *= sa_cooling
            
            stagnation_count = 0
            T = max(T * 0.95, min_T)
        else:
            # Regular SA step
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply
            if old_r == 0:
                counts[p:M+1:p] -= 1
            else:
                counts[old_r:M+1:p] -= 1
            if new_r == 0:
                counts[p:M+1:p] += 1
            else:
                counts[new_r:M+1:p] += 1
            residues[idx] = new_r
            
            new_L = get_L(counts, M)
            delta = new_L - current_L
            
            if delta > 0:
                current_L = new_L
                stagnation_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
            elif delta == 0:
                pass
            else:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    if new_r == 0:
                        counts[p:M+1:p] -= 1
                    else:
                        counts[new_r:M+1:p] -= 1
                    if old_r == 0:
                        counts[p:M+1:p] += 1
                    else:
                        counts[old_r:M+1:p] += 1
                    residues[idx] = old_r
            
            T = max(T * cooling, min_T)
    
    return best_residues
# EVOLVE-BLOCK-END
