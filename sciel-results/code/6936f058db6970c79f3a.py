import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin for 150s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Precompute prime indices sorted by prime value descending
    prime_indices_desc = sorted(range(n_primes), key=lambda i: primes[i], reverse=True)
    
    # We will search for L up to M.
    M = 5000
    counts = np.zeros(M + 1, dtype=np.int64)
    
    # Initialize residues with greedy construction
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Phase 1: Greedy initialization
    # Assign residues to cover as many small integers as possible
    for i, p in enumerate(primes):
        # For small primes, try to cover early integers
        # Use a heuristic: assign residue that covers the first uncovered integer
        best_r = 0
        best_score = -1
        for r in range(min(p, 50)):  # Try first 50 residues
            # Count how many integers in [1, 100] this residue covers
            # that are currently uncovered or barely covered
            cnt = 0
            for x in range(1, min(101, M + 1)):
                if x % p == r:
                    if counts[x] == 0:
                        cnt += 10
                    elif counts[x] == 1:
                        cnt += 5
                    else:
                        cnt += 1
            if cnt > best_score:
                best_score = cnt
                best_r = r
        residues[i] = best_r
        counts[best_r::p] += 1
    
    def calc_L(counts, M):
        """Returns the largest L such that counts[1..L] >= 2"""
        if M < 1 or counts[1] < 2:
            return 0
        # Find first index >= 1 where count < 2
        mask = counts[1:M+1] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx
    
    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    # Phase 2: Greedy repair to extend L
    x = 1
    while x <= M and time.time() - start_time < deadline - 10:
        if counts[x] >= 2:
            x += 1
            continue
            
        # counts[x] < 2, need to fix
        fixed = False
        # Try large primes first to minimize disruption
        for i in prime_indices_desc:
            p = primes[i]
            if residues[i] != x % p:
                r_old = residues[i]
                r_new = x % p
                
                # Check if this move breaks any y < x with count == 2
                is_safe = True
                if r_old < x:
                    k = (x - 1 - r_old) // p
                    if k >= 0:
                        y = r_old + k * p
                        while y >= 1:
                            if counts[y] == 2:
                                is_safe = False
                                break
                            y -= p
                            
                if is_safe:
                    counts[r_old::p] -= 1
                    counts[r_new::p] += 1
                    residues[i] = r_new
                    fixed = True
                    break
        
        if not fixed:
            # No safe move, just pick largest prime that doesn't cover x
            for i in prime_indices_desc:
                p = primes[i]
                if residues[i] != x % p:
                    r_old = residues[i]
                    r_new = x % p
                    counts[r_old::p] -= 1
                    counts[r_new::p] += 1
                    residues[i] = r_new
                    break
        x += 1
    
    current_L = calc_L(counts, M)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues.copy()
    
    # Phase 3: Simulated Annealing with multi-start
    T = 15.0
    T_min = 0.05
    cooling_rate = 0.9998
    best_L_global = best_L
    best_residues_global = best_residues.copy()
    
    # Multiple SA runs with different starting points
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 5:
            break
            
        # For restarts after the first, perturb the best solution
        if restart > 0:
            residues = best_residues_global.copy()
            # Randomly perturb 10-20 residues
            n_perturb = random.randint(10, 20)
            for _ in range(n_perturb):
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r != old_r:
                    counts[old_r::p] -= 1
                    counts[new_r::p] += 1
                    residues[i] = new_r
        else:
            # Use current state
            pass
        
        current_L = calc_L(counts, M)
        no_improve = 0
        
        while time.time() - start_time < deadline - 2:
            # Pick a prime to update
            # Bias toward large primes
            if random.random() < 0.6:
                i = prime_indices_desc[random.randint(0, min(30, n_primes - 1))]
            else:
                i = random.randint(0, n_primes - 1)
            
            p = primes[i]
            r_old = residues[i]
            
            # Determine new residue
            # With some probability, target the first uncovered point
            if current_L < M and random.random() < 0.5:
                x_target = current_L + 1
                r_new = x_target % p
                if r_new == r_old:
                    r_new = (r_new + random.randint(1, p - 1)) % p
            else:
                r_new = random.randint(0, p - 1)
            
            if r_new == r_old:
                continue
            
            # Apply move
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            
            new_L = calc_L(counts, M)
            
            if new_L >= current_L:
                current_L = new_L
                residues[i] = r_new
                no_improve = 0
                
                if new_L > best_L_global:
                    best_L_global = new_L
                    best_residues_global = residues.copy()
                    
                    # Extend M if close to boundary
                    if best_L_global >= M - 50:
                        new_M = min(M * 2, 20000)
                        old_counts = counts
                        counts = np.zeros(new_M + 1, dtype=np.int64)
                        # Copy old counts
                        counts[:M+1] = old_counts
                        # Add contributions for new range
                        for j, pj in enumerate(primes):
                            rj = residues[j]
                            # Find first index >= M+1 that is congruent to rj mod pj
                            if rj <= M:
                                start_idx = rj + ((M - rj) // pj + 1) * pj
                            else:
                                start_idx = rj
                            if start_idx <= new_M:
                                counts[start_idx::pj] += 1
                        M = new_M
                        current_L = calc_L(counts, M)
                        if current_L > best_L_global:
                            best_L_global = current_L
                            best_residues_global = residues.copy()
            else:
                delta = current_L - new_L
                # SA acceptance
                if T > T_min:
                    prob = np.exp(-min(delta, 50) / T)
                    if random.random() < prob:
                        current_L = new_L
                        residues[i] = r_new
                        no_improve += 1
                    else:
                        # Revert
                        counts[r_new::p] -= 1
                        counts[r_old::p] += 1
                        no_improve += 1
                else:
                    # Revert
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    no_improve += 1
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # Reheat if stuck
            if no_improve > 3000:
                T = max(T, 5.0)
                no_improve = 0
    
    # Phase 4: Final local search polish
    if time.time() - start_time < deadline - 1:
        residues = best_residues_global.copy()
        # Rebuild counts
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        
        current_L = calc_L(counts, M)
        
        # Try to improve by fixing uncovered points
        for x in range(1, min(current_L + 100, M + 1)):
            if time.time() - start_time > deadline - 0.5:
                break
            if counts[x] >= 2:
                continue
            
            # Try to fix x
            for i in prime_indices_desc:
                p = primes[i]
                if residues[i] != x % p:
                    r_old = residues[i]
                    r_new = x % p
                    
                    # Check safety
                    is_safe = True
                    if r_old < x:
                        k = (x - 1 - r_old) // p
                        if k >= 0:
                            y = r_old + k * p
                            while y >= 1:
                                if counts[y] == 2:
                                    is_safe = False
                                    break
                                y -= p
                    
                    if is_safe:
                        counts[r_old::p] -= 1
                        counts[r_new::p] += 1
                        residues[i] = r_new
                        break
        
        new_L = calc_L(counts, M)
        if new_L > best_L_global:
            best_L_global = new_L
            best_residues_global = residues.copy()
    
    return [int(r) for r in best_residues_global]
