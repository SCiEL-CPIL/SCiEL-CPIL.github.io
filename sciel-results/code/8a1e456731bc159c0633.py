import numpy as np
import random
import time
import math

def _primes_up_to(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    N = 50000
    
    # Precompute weights for biased move selection in SA
    # Smaller primes are more impactful
    weights = np.array([1.0 / p for p in primes])
    weights /= weights.sum()
    
    global_best_r = None
    global_best_L = 0
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper to rebuild cov from r
    def build_cov(r):
        cov = np.zeros(N + 1, dtype=np.int16)
        for i, p in enumerate(primes):
            rr = r[i]
            if rr == 0:
                cov[p::p] += 1
            else:
                cov[rr::p] += 1
        cov[0] = 100 # Ignore 0
        return cov

    def get_L(cov):
        bad = np.nonzero(cov[1:] < 2)[0]
        if len(bad) == 0:
            return N
        return bad[0]

    def apply_change(cov, r, i, new_r):
        p = primes[i]
        old_r = r[i]
        if old_r == new_r:
            return
        
        # Remove old
        if old_r == 0:
            cov[p::p] -= 1
        else:
            cov[old_r::p] -= 1
            
        # Add new
        r[i] = new_r
        if new_r == 0:
            cov[p::p] += 1
        else:
            cov[new_r::p] += 1
            
    def revert_change(cov, r, i, old_r):
        p = primes[i]
        new_r = r[i]
        if old_r == new_r:
            return
        
        # Remove new
        if new_r == 0:
            cov[p::p] -= 1
        else:
            cov[new_r::p] -= 1
            
        # Add old
        r[i] = old_r
        if old_r == 0:
            cov[p::p] += 1
        else:
            cov[old_r::p] += 1

    # Initializations
    inits = [
        np.zeros(n_primes, dtype=np.int32),
        np.ones(n_primes, dtype=np.int32),
        np.array([random.randint(0, p-1) for p in primes], dtype=np.int32),
        np.array([random.randint(0, p-1) for p in primes], dtype=np.int32)
    ]
    
    for init_r in inits:
        if time.time() > deadline:
            break
            
        r = init_r.copy()
        cov = build_cov(r)
        L = get_L(cov)
        
        # Greedy Phase
        while L < N and time.time() < deadline:
            x = L + 1
            candidates = []
            best_score = -1
            best_i = -1
            
            for i, p in enumerate(primes):
                target_r = x % p
                if r[i] == target_r:
                    continue
                
                old_r = r[i]
                # Check if removing old_r coverage breaks any y < x
                # We need min cov in the range to be >= 3
                
                if old_r == 0:
                    if x <= p:
                        min_cov = 100
                    else:
                        # Indices: p, 2p, ... < x
                        min_cov = np.min(cov[p:x:p])
                else:
                    if old_r >= x:
                        min_cov = 100
                    else:
                        # Indices: old_r, old_r+p, ... < x
                        # Note: if old_r == 0, handled above.
                        # If old_r > 0, first index is old_r.
                        # But we only care about indices >= 1.
                        # If old_r == 0, it's handled.
                        # If old_r > 0, indices are old_r, old_r+p...
                        # If old_r < 1, i.e., 0, handled.
                        # So old_r >= 1.
                        min_cov = np.min(cov[old_r:x:p])
                
                if min_cov >= 3:
                    # Prefer moves with higher min_cov (safer)
                    # Or smaller p (more impact)?
                    # Let's use min_cov as primary, p as secondary
                    score = min_cov * 10000 - p
                    if score > best_score:
                        best_score = score
                        best_i = i
                        
            if best_i != -1:
                apply_change(cov, r, best_i, x % primes[best_i])
                L = get_L(cov)
            else:
                break
        
        # Update Global Best
        if L > global_best_L:
            global_best_L = L
            global_best_r = r.copy()
            
        # SA Phase
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline:
            # Biased move selection
            i = random.choices(range(n_primes), weights=weights, k=1)[0]
            p = primes[i]
            old_r = r[i]
            new_r = random.randint(0, p - 1)
            
            if old_r == new_r:
                continue
                
            apply_change(cov, r, i, new_r)
            new_L = get_L(cov)
            
            delta = new_L - L
            
            if delta > 0 or random.random() < math.exp(delta / T):
                L = new_L
                if L > global_best_L:
                    global_best_L = L
                    global_best_r = r.copy()
            else:
                revert_change(cov, r, i, old_r)
                
            T *= cooling
            
            if T < 0.01:
                # Restart from current best of this run? Or global best?
                # Let's restart from the best state found in this specific run's SA
                # Actually, restarting from global_best_r is better for diversity
                # But we need to rebuild cov.
                if time.time() > deadline - 5:
                    break
                r = global_best_r.copy()
                cov = build_cov(r)
                L = get_L(cov)
                T = 1.0

    if global_best_r is None:
        return [0] * n_primes
        
    return global_best_r.tolist()
