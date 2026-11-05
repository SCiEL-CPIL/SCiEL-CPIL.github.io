import time
import random
import numpy as np
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
    
    start_time = time.time()
    deadline = start_time + 110

    r = np.zeros(n_primes, dtype=np.int32)
    cov = np.zeros(N + 1, dtype=np.int32)

    def build_cov():
        cov[:] = 0
        for i, p in enumerate(primes):
            rr = r[i]
            if rr == 0:
                cov[p::p] += 1
            else:
                cov[rr::p] += 1
        cov[0] = 100

    def get_L():
        bad = np.nonzero(cov[1:] < 2)[0]
        if len(bad) == 0:
            return N
        return int(bad[0])

    def apply_change(i, new_r):
        p = primes[i]
        old_r = r[i]
        if old_r == new_r:
            return
        if old_r == 0:
            cov[p::p] -= 1
        else:
            cov[old_r::p] -= 1
        
        r[i] = new_r
        
        if new_r == 0:
            cov[p::p] += 1
        else:
            cov[new_r::p] += 1

    def revert_change(i, old_r):
        p = primes[i]
        new_r = r[i]
        if old_r == new_r:
            return
        if new_r == 0:
            cov[p::p] -= 1
        else:
            cov[new_r::p] -= 1
            
        r[i] = old_r
        
        if old_r == 0:
            cov[p::p] += 1
        else:
            cov[old_r::p] += 1

    build_cov()
    L = get_L()
    best_L = L
    best_r = r.copy()

    random.seed(42)
    no_progress = 0
    
    while time.time() < deadline:
        if L >= N:
            break
            
        x = L + 1
        
        # Try to fix x
        found_move = False
        for i in range(n_primes):
            p = primes[i]
            target_r = x % p
            if r[i] == target_r:
                continue
            
            old_r = r[i]
            
            # Check if removing old_r breaks any y in [1, L]
            # Positions affected by removal: y in [1, L] such that y % p == old_r
            # We need min cov in these positions to be >= 3 (so that after -1, it is >= 2)
            
            if old_r == 0:
                # Positions are p, 2p, ...
                if p > L:
                    # No positions in [1, L] affected
                    min_cov_check = 100
                else:
                    # Slice: p, 2p, ... <= L
                    # In 0-indexed cov, indices are p, 2p, ...
                    # cov[p:L+1:p]
                    min_cov_check = np.min(cov[p:L+1:p])
            else:
                # Positions are old_r, old_r+p, ...
                if old_r > L:
                    min_cov_check = 100
                else:
                    min_cov_check = np.min(cov[old_r:L+1:p])
            
            if min_cov_check < 3:
                continue
                
            # If safe, apply and check if x is covered
            apply_change(i, target_r)
            
            if cov[x] >= 2:
                # Success!
                new_L = get_L()
                if new_L > L:
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_r = r.copy()
                    found_move = True
                    break
                else:
                    # cov[x] >= 2 but L didn't increase?
                    # This means L+1 was already covered? No, L is the first bad.
                    # So cov[x] was < 2. Now it is >= 2.
                    # So L should increase to at least x.
                    # Unless get_L is slow or wrong?
                    # Actually, if cov[x] >= 2, then x is not bad.
                    # So the next bad position is > x.
                    # So new_L >= x > L.
                    # So this case shouldn't happen if logic is correct.
                    revert_change(i, old_r)
            else:
                # cov[x] still < 2?
                # This means target_r didn't cover x?
                # x % p == target_r. So x is covered by this prime.
                # So cov[x] should have increased by 1.
                # It was < 2. Now it is < 2 + 1.
                # If it was 0, now 1. Still < 2.
                # If it was 1, now 2. OK.
                # So if cov[x] was 0, one prime is not enough.
                # We need to add 2 primes?
                # Greedy repair assumes we can fix it with 1 prime.
                # If cov[x] was 0, we need 2 primes.
                # We can't do that in one step easily.
                # So we revert.
                revert_change(i, old_r)
        
        if found_move:
            no_progress = 0
        else:
            no_progress += 1
            if no_progress > 5:
                # Kick
                for _ in range(5):
                    i = random.randint(0, n_primes - 1)
                    p = primes[i]
                    new_r = random.randint(0, p - 1)
                    apply_change(i, new_r)
                
                L = get_L()
                if L < best_L:
                    r = best_r.copy()
                    build_cov()
                    L = best_L
                no_progress = 0

    # Final Polish with SA
    r = best_r.copy()
    build_cov()
    L = get_L()
    current_L = L
    best_L_sa = L
    best_r_sa = r.copy()
    
    T = 1.0
    while time.time() < deadline and T > 0.01:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = r[i]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        apply_change(i, new_r)
        new_L = get_L()
        
        delta = new_L - current_L
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_L = new_L
            if current_L > best_L_sa:
                best_L_sa = current_L
                best_r_sa = r.copy()
        else:
            revert_change(i, old_r)
            
        T *= 0.999

    if best_L_sa > best_L:
        best_r = best_r_sa

    return [int(x) for x in best_r.tolist()]
