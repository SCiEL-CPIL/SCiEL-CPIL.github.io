import numpy as np

def construct():
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    P = len(primes)
    L_MAX = 2000 # We aim for > 1000, 2000 is a good target to search in
    
    x_vals = np.arange(1, L_MAX + 1)
    
    # Initialization: Two-shift strategy
    # Assign 0 to first half, 1 to second half
    residues = np.zeros(P, dtype=int)
    residues[P//2:] = 1
    for i, p in enumerate(primes):
        residues[i] = residues[i] % p

    # Precompute coverage for each prime
    cov_p = np.zeros((P, L_MAX), dtype=np.uint8)
    for i, p in enumerate(primes):
        r = residues[i]
        cov_p[i] = (x_vals % p == r)
    
    cov_total = np.sum(cov_p, axis=0)

    def find_L(cov):
        if len(cov) == 0: return 0
        if cov[0] < 2: return 0
        failures = np.where(cov < 2)[0]
        if len(failures) == 0:
            return L_MAX
        return failures[0]

    current_L = find_L(cov_total)
    
    # Local Search
    np.random.seed(42)
    
    for step in range(3000):
        if current_L >= L_MAX:
            break
        
        failures = np.where(cov_total < 2)[0]
        if len(failures) == 0:
            break
            
        # Target the first failure
        target_idx = failures[0]
        target_x = target_idx + 1
        
        improved = False
        
        # Try to fix by changing one prime
        # Sample 30 random primes
        for _ in range(30):
            i = np.random.randint(P)
            p = primes[i]
            new_r = target_x % p
            
            if new_r == residues[i]:
                continue
            
            new_cov_i = (x_vals % p == new_r)
            
            # Optimization: Only check if the change improves the situation
            # We can quickly check if target_x is now covered (it is, by construction)
            # And if any other x drops below 2.
            
            cov_total_new = cov_total - cov_p[i] + new_cov_i
            new_L = find_L(cov_total_new)
            
            if new_L > current_L:
                residues[i] = new_r
                cov_p[i] = new_cov_i
                cov_total = cov_total_new
                current_L = new_L
                improved = True
                break
            elif new_L == current_L:
                # If L is same, check if the specific target is fixed?
                # If target was failing, and now it's passing, but L didn't increase,
                # it means another failure appeared earlier or the first failure was not the target?
                # No, target is the FIRST failure. If we fix it, and L doesn't increase,
                # it means there was another failure at the same index? Impossible, cov is integer.
                # Or a new failure appeared at an index < target_idx?
                # If we fix target_idx, and a new failure appears at j < target_idx, L decreases.
                # If new failure appears at j > target_idx, L stays same (since target_idx is now ok, next failure is j).
                # So if new_L == current_L, it means we fixed target_idx, but there was already a failure at target_idx?
                # No, target_idx WAS the first failure. So cov[target_idx] < 2.
                # After change, cov[target_idx] >= 2.
                # If new_L == current_L, it implies the new first failure is at current_L?
                # Wait, find_L returns the index of the FIRST failure.
                # If old first failure was at idx K, L=K.
                # If we fix K, and the next failure is at K+1, new L = K+1.
                # If we fix K, and a new failure appears at K-1, new L = K-1.
                # If we fix K, and no new failure appears before the old next failure, L increases.
                # So new_L == current_L is unlikely if we fix the first failure, unless we break something earlier.
                pass

        if not improved:
            # Stuck. Try a random move that doesn't decrease L.
            i = np.random.randint(P)
            p = primes[i]
            new_r = np.random.randint(p)
            new_cov_i = (x_vals % p == new_r)
            cov_total_new = cov_total - cov_p[i] + new_cov_i
            new_L = find_L(cov_total_new)
            
            if new_L >= current_L:
                residues[i] = new_r
                cov_p[i] = new_cov_i
                cov_total = cov_total_new
                current_L = new_L

    return [int(r) for r in residues]
