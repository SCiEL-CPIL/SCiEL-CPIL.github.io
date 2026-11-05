import time
import numpy as np

def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _build_counts(residues, primes, max_L):
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(len(primes)):
        r = residues[i]
        p = primes[i]
        if r == 0:
            r_eff = p
        else:
            r_eff = r
        if r_eff <= max_L:
            counts[r_eff::p] += 1
    return counts

def _find_L(counts, max_L):
    deficient = np.where(counts[1:max_L+1] < 2)[0]
    if len(deficient) == 0:
        return max_L
    return deficient[0]

def _score(residues, primes, max_L=100000):
    counts = _build_counts(residues, primes, max_L)
    return _find_L(counts, max_L)

def _greedy_construct(primes, max_L_check=5000):
    n = len(primes)
    residues = [0] * n
    counts = np.zeros(max_L_check + 1, dtype=np.int32)
    
    for idx in range(n):
        p = primes[idx]
        best_r = 0
        best_gain = -1
        
        deficient_positions = []
        for pos in range(1, min(max_L_check + 1, 200)):
            if counts[pos] < 2:
                deficient_positions.append(pos)
                if len(deficient_positions) >= 50:
                    break
        
        if not deficient_positions:
             best_r = 1
             best_count = 0
             for r in range(p):
                 r_eff = p if r == 0 else r
                 if r_eff > max_L_check: continue
                 num_positions = (max_L_check - r_eff) // p + 1
                 if num_positions > best_count:
                     best_count = num_positions
                     best_r = r
        else:
             residue_counts = np.zeros(p, dtype=np.int32)
             for pos in deficient_positions:
                 r_needed = pos % p
                 weight = 2 if counts[pos] == 0 else 1
                 residue_counts[r_needed] += weight
             
             best_r = int(np.argmax(residue_counts))
        
        residues[idx] = best_r
        
        r = best_r
        r_eff = p if r == 0 else r
        if r_eff <= max_L_check:
            counts[r_eff::p] += 1
            
    return residues

def construct():
    primes = _get_primes(1000)
    n = len(primes)
    deadline = time.time() + 140
    
    best_residues = None
    best_score = 0
    
    # Multiple restarts
    max_restarts = 5
    time_per_restart = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # 1. Greedy Init
        residues = _greedy_construct(primes)
        
        # 2. Local Search
        # Use a window that grows if we hit the boundary
        max_L_opt = 2000
        current_score = _score(residues, primes, max_L_opt)
        
        while time.time() < deadline - 5:
            improved = False
            indices = list(range(n))
            np.random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                    
                p = primes[idx]
                old_r = residues[idx]
                
                counts = _build_counts(residues, primes, max_L_opt)
                
                r_eff_old = p if old_r == 0 else old_r
                if r_eff_old <= max_L_opt:
                    counts[r_eff_old::p] -= 1
                
                best_r = old_r
                best_local_score = _find_L(counts, max_L_opt)
                
                # Try all residues
                # Optimization: If p is large, sampling might be ok, but p<=997 is small enough for full search
                for new_r in range(p):
                    r_eff_new = p if new_r == 0 else new_r
                    if r_eff_new <= max_L_opt:
                        counts[r_eff_new::p] += 1
                    
                    new_score = _find_L(counts, max_L_opt)
                    
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_r = new_r
                        
                    if r_eff_new <= max_L_opt:
                        counts[r_eff_new::p] -= 1
                
                if best_r != old_r:
                    residues[idx] = best_r
                    improved = True
            
            if not improved:
                break
            
            # Check if we need to expand window
            if current_score >= max_L_opt:
                # Calculate true score on larger range
                true_score = _score(residues, primes, 10000)
                if true_score > max_L_opt:
                    max_L_opt = true_score + 1000
                    if max_L_opt > 100000:
                        max_L_opt = 100000
                else:
                    # Stuck at boundary? Maybe just break or continue
                    pass
            
            current_score = _score(residues, primes, max_L_opt)

        # Update global best
        final_score = _score(residues, primes, 100000)
        if final_score > best_score:
            best_score = final_score
            best_residues = residues[:]

    if best_residues is None:
        best_residues = _greedy_construct(primes)
        
    return best_residues
