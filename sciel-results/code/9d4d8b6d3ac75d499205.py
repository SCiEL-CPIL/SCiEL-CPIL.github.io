import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Initial residues: try to get a good starting point
    # The seed strategy: r_p = ((i + 2) // 2) % p
    # Let's start with a randomized initialization or the seed, then improve.
    # Actually, a smarter initialization:
    # We want to cover 1..L.
    # Let's use a greedy approach to build the solution.
    
    # State: residues list
    # Score: max L such that 1..L are covered >= 2 times
    
    def compute_score(residues):
        # Efficiently compute the max L
        # We need to check coverage for 1, 2, ...
        # Coverage of x by prime p[i] is 1 if (x - residues[i]) % primes[i] == 0
        # We can precompute for each prime the residues mod p.
        
        # To speed this up, we can maintain a count array for a fixed range?
        # No, L can grow.
        # But L is likely small (seed gets 84).
        # Let's just iterate x from 1 upwards until we find a gap.
        
        counts = [0] * 10000 # Cap at 10000 for speed, task says capped at 10^5 but we expect small L
        # Actually, we can just count on the fly for x=1..L
        # But we don't know L.
        
        # Better: For each prime i, it covers x if x % p_i == r_i.
        # Let's create a boolean array or count array for a sufficient range, say 5000.
        # If L > 5000, we are doing great, but unlikely.
        
        max_check = 2000
        cover_count = [0] * (max_check + 1)
        
        for i, p in enumerate(primes):
            r = residues[i]
            # x = r, r+p, r+2p, ... <= max_check
            # Note: x must be >= 1
            first_x = r
            if first_x == 0:
                first_x = p # x=p is 0 mod p, but x=0 is not in 1..L
            # Wait, x % p == r.
            # If r == 0, x can be p, 2p, ...
            # If r > 0, x can be r, r+p, ... (if r>=1)
            
            start_x = r
            if start_x == 0:
                start_x = p
            if start_x > max_check:
                continue
            # Ensure start_x >= 1
            if start_x < 1:
                # This shouldn't happen since r >= 0 and p >= 2
                # If r=0, start_x=p. If r>0, start_x=r.
                pass
                
            for x in range(start_x, max_check + 1, p):
                cover_count[x] += 1
                
        # Find max L
        L = 0
        for x in range(1, max_check + 1):
            if cover_count[x] >= 2:
                L = x
            else:
                break
        return L

    # Greedy/Local Search Strategy
    # 1. Initialize residues
    # 2. Compute score
    # 3. While time remains:
    #    - Identify the "bottleneck" integer x (smallest x where coverage < 2 or coverage is critical)
    #    - Actually, simpler: Local search.
    #    - Pick a random prime i.
    #    - Try changing residues[i] to a new value.
    #    - If score improves, accept.
    #    - If not, maybe accept with probability (Simulated Annealing) or just reject.
    
    # To make it "Materially Different", let's try a **Greedy Construction from Scratch**
    # that thinks about coverage.
    
    # Strategy:
    # Start with L=0.
    # We want to extend L.
    # The current seed is static.
    # Let's try a heuristic:
    # For each prime, choose residue to maximize coverage of the "frontier".
    
    # Let's use a hybrid: Start with seed, then perform hill climbing with a specific move:
    # "Fix the first gap".
    
    def get_first_gap(residues):
        # Returns the first x where coverage < 2
        # Reuse logic from compute_score but stop early
        max_check = 2000
        cover_count = [0] * (max_check + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            start_x = r if r != 0 else p
            if start_x < 1: continue # r=0 -> p, r>0 -> r. r is 0..p-1.
            # Actually if r=0, x=p, 2p...
            # If r>0, x=r, r+p...
            # But x must be >= 1.
            # r is in [0, p-1].
            # If r=0, first positive x is p.
            # If r>0, first positive x is r.
            
            for x in range(start_x, max_check + 1, p):
                cover_count[x] += 1
                
        for x in range(1, max_check + 1):
            if cover_count[x] < 2:
                return x
        return max_check + 1

    # Initialize
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    current_score = compute_score(residues)
    best_residues = residues[:]
    best_score = current_score
    
    # Hill Climbing
    # Move: Change one residue.
    # To be efficient, we can't recompute full score from scratch every time if it's slow.
    # But n=168, max_check=2000. 168*2000/avg_p operations.
    # Sum(2000/p) approx 2000 * log(log(1000)) ~ 2000 * 2.5 = 5000 ops per score calc.
    # Very fast.
    
    import random
    random.seed(42)
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick a random prime to modify
        i = random.randint(0, n_primes - 1)
        old_r = residues[i]
        
        # Try a random new residue
        new_r = random.randint(0, primes[i] - 1)
        if new_r == old_r:
            continue
            
        residues[i] = new_r
        new_score = compute_score(residues)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues[:]
        else:
            # Revert
            residues[i] = old_r
            
        # Occasionally try a "targeted" move:
        # Find the first gap x.
        # Try to increase coverage at x.
        # Which primes can cover x?
        # x % p == r_p.
        # For each prime p, the required residue is x % p.
        # If we set r_p = x % p, we add 1 to coverage of x.
        # But it might break coverage elsewhere.
        
        if steps % 10 == 0:
            gap_x = get_first_gap(residues)
            if gap_x <= 2000:
                # Try to fix gap_x
                # Find primes where current residue is NOT gap_x % p
                # and changing it to gap_x % p is "cheap" (doesn't hurt too much)
                # Just try one random such prime
                candidates = []
                for j, p in enumerate(primes):
                    req_r = gap_x % p
                    if residues[j] != req_r:
                        candidates.append(j)
                
                if candidates:
                    j = random.choice(candidates)
                    old_r_j = residues[j]
                    residues[j] = gap_x % primes[j]
                    test_score = compute_score(residues)
                    if test_score > current_score:
                        current_score = test_score
                        if test_score > best_score:
                            best_score = test_score
                            best_residues = residues[:]
                    else:
                        residues[j] = old_r_j
                        
    return best_residues
