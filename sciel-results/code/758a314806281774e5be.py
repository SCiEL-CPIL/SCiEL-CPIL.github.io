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
    
    # State
    r = np.zeros(n_primes, dtype=np.int32)
    cov = np.zeros(N + 1, dtype=np.int8)
    
    # Helper to rebuild cov from r
    def build_cov():
        cov[:] = 0
        for i, p in enumerate(primes):
            rr = r[i]
            if rr == 0:
                cov[p::p] += 1
            else:
                cov[rr::p] += 1
        cov[0] = 100 # Ignore 0
        
    # Helper to get L
    def get_L():
        # Find first index in 1..N where cov < 2
        bad = np.nonzero(cov[1:] < 2)[0]
        if len(bad) == 0:
            return N
        return bad[0]
        
    # Helper to apply change
    def apply_change(i, new_r):
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
            
    # Helper to revert
    def revert_change(i, old_r):
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

    # Greedy Phase
    build_cov()
    L = get_L()
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Greedy: try to extend L
    while L < N and time.time() < deadline:
        x = L + 1
        candidates = []
        for i, p in enumerate(primes):
            target_r = x % p
            if r[i] == target_r:
                continue
            
            old_r = r[i]
            # Check if removing old_r coverage breaks any y < x
            # y must be in [1, x-1] and y % p == old_r
            # cov[y] must be >= 3
            
            if old_r == 0:
                # Indices: p, 2p, ... < x
                # If x <= p, no indices.
                if x <= p:
                    candidates.append(i)
                    continue
                # Check min cov in p, 2p, ..., floor((x-1)/p)*p
                # We can just check the specific indices
                # But looping in Python is slow.
                # Let's use numpy slice
                # Indices are p, 2p, ...
                # cov[p:x:p]
                min_cov = np.min(cov[p:x:p]) if x > p else 100
                if min_cov >= 3:
                    candidates.append(i)
            else:
                # Indices: old_r, old_r+p, ... < x
                # First index is old_r if old_r >= 1
                # If old_r == 0, handled above.
                # If old_r > 0:
                # Indices: old_r, old_r+p, ...
                # We need indices in [1, x-1]
                # If old_r >= x, no indices.
                if old_r >= x:
                    candidates.append(i)
                    continue
                # Slice: cov[old_r:x:p]
                min_cov = np.min(cov[old_r:x:p])
                if min_cov >= 3:
                    candidates.append(i)
                    
        if candidates:
            i = random.choice(candidates)
            apply_change(i, x % primes[i])
            L = get_L()
        else:
            break
            
    best_r = r.copy()
    best_L = L
    
    # SA Phase
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = r[i]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        apply_change(i, new_r)
        new_L = get_L()
        
        delta = new_L - L
        
        if delta > 0 or random.random() < math.exp(delta / T):
            L = new_L
            if L > best_L:
                best_L = L
                best_r = r.copy()
        else:
            revert_change(i, old_r)
            
        T *= cooling
        
        if T < 0.01:
            # Restart
            r = best_r.copy()
            build_cov()
            L = get_L()
            T = 1.0

    return best_r.tolist()
