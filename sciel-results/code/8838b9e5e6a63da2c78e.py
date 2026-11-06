"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _compute_L(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by at least 2 congruences."""
    n_primes = len(primes)
    # For each prime p_i with residue r_i, it covers numbers x where x ≡ r_i (mod p_i)
    # We need to check coverage for x = 1, 2, 3, ...
    
    # Efficient approach: for each x, count how many primes cover it
    # x is covered by prime i if x % primes[i] == residues[i]
    
    # We'll track the minimum coverage as we go
    # For efficiency, we process in batches
    
    L = 0
    x = 1
    while x <= max_check:
        # Check if x is covered by at least 2 primes
        count = 0
        for i in range(n_primes):
            if x % primes[i] == residues[i]:
                count += 1
                if count >= 2:
                    break
        if count < 2:
            break
        L = x
        x += 1
    
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Initial assignment: try to cover small numbers well
    # Strategy: for each prime, choose residue to cover as many small under-covered numbers as possible
    residues = [0] * n_primes
    
    # Better initial: use the seed strategy but improved
    # For prime i (0-indexed), cover ceil((i+1)/2) 
    for i, p in enumerate(primes):
        residues[i] = ((i + 2) // 2) % p
    
    # Compute initial L
    best_L = _compute_L(residues, primes)
    best_residues = residues[:]
    
    # Local search: try to improve by changing one residue at a time
    # Focus on the bottleneck: find smallest x with coverage < 2
    def get_bottleneck(residues, primes):
        """Find the smallest x >= 1 with coverage < 2."""
        x = 1
        while x <= 100000:
            count = 0
            for i in range(len(primes)):
                if x % primes[i] == residues[i]:
                    count += 1
                    if count >= 2:
                        break
            if count < 2:
                return x
            x += 1
        return 100000
    
    def try_improve(residues, primes, bottleneck_x):
        """Try to change one residue to cover the bottleneck x."""
        # For bottleneck_x to be covered by prime i, we need residues[i] = bottleneck_x % primes[i]
        # Try each prime and see if changing its residue helps
        best_gain = 0
        best_change = None
        
        for i in range(len(primes)):
            new_r = bottleneck_x % primes[i]
            if new_r == residues[i]:
                continue  # Already covers it
            
            # Try this change and see if it improves L
            old_r = residues[i]
            residues[i] = new_r
            L = _compute_L(residues, primes)
            residues[i] = old_r
            
            if L > best_gain:
                best_gain = L
                best_change = (i, new_r)
        
        return best_gain, best_change
    
    iterations = 0
    while time.time() < deadline - 5:  # 5 second safety margin
        iterations += 1
        
        bottleneck_x = get_bottleneck(best_residues, primes)
        if bottleneck_x >= 100000:
            break
        
        # Try to improve by fixing the bottleneck
        gain, change = try_improve(best_residues, primes, bottleneck_x)
        
        if change is not None and gain > best_L:
            i, new_r = change
            best_residues[i] = new_r
            best_L = gain
        
        # Also try random perturbations
        if iterations % 10 == 0:
            # Random local search: pick a random prime and try a few random residues
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(0, n_primes)
                old_r = best_residues[i]
                # Try a few random residues
                for _ in range(3):
                    new_r = np.random.randint(0, primes[i])
                    best_residues[i] = new_r
                    L = _compute_L(best_residues, primes)
                    if L > best_L:
                        best_L = L
                    else:
                        best_residues[i] = old_r
                        break
                else:
                    best_residues[i] = old_r
    
    return best_residues
# EVOLVE-BLOCK-END
