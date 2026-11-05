import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    # Convert to set for O(1) lookup? No, we need residue coverage.
    # points is a list or set.
    point_set = set(points)
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in point_set:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    # We will iteratively add primes to our constraint set P
    # Start with a small set
    P = []
    best_S = set()
    best_size = 0
    
    # Precompute bitsets for all primes up to N? 
    # Memory: sum(p for p in primes) * (N/8) bytes.
    # Sum of primes up to 5000 is ~ 15 million? No, sum of primes up to x is ~ x^2 / (2 ln x).
    # 5000^2 / (2 * 8.5) = 25e6 / 17 ~ 1.5e6.
    # 1.5e6 * 625 bytes ~ 1 GB. Too much.
    
    # So we compute bitsets on the fly or only for active primes.
    # Active primes are small.
    
    # Strategy:
    # 1. Start with P = primes up to 50.
    # 2. Optimize residues for P.
    # 3. Check admissibility.
    # 4. If not admissible, find the smallest violating prime p, add to P, repeat.
    
    # To speed up, we precompute bitsets for primes up to 200.
    # If we need more, we compute on the fly.
    
    max_precompute_prime = 200
    precomputed_bits = {}
    for p in all_primes:
        if p > max_precompute_prime:
            break
        bits = [0] * p
        for r in range(p):
            mask = 0
            for x in range(r, N, p):
                mask |= (1 << x)
            bits[r] = mask
        precomputed_bits[p] = bits
    
    def get_bits(p):
        if p in precomputed_bits:
            return precomputed_bits[p]
        bits = [0] * p
        for r in range(p):
            mask = 0
            for x in range(r, N, p):
                mask |= (1 << x)
            bits[r] = mask
        precomputed_bits[p] = bits
        return bits

    # Initialize P with small primes
    P = [p for p in all_primes if p <= 50]
    residues = {p: random.randint(0, p-1) for p in P}
    
    # Local Search Optimization
    def optimize(P, residues):
        # Precompute bits for all p in P
        bits_map = {p: get_bits(p) for p in P}
        
        # Current K (killed set)
        K = 0
        for p in P:
            K |= bits_map[p][residues[p]]
        
        current_score = N - K.bit_count()
        
        # Hill Climbing
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            improved = False
            # Shuffle P for diversity
            P_shuffled = P[:]
            random.shuffle(P_shuffled)
            
            for p in P_shuffled:
                current_r = residues[p]
                base_K = K & ~bits_map[p][current_r]
                
                best_q = current_r
                best_score = current_score
                best_new_K = K
                
                # Try all other residues
                for q in range(p):
                    if q == current_r:
                        continue
                    new_K = base_K | bits_map[p][q]
                    new_score = N - new_K.bit_count()
                    if new_score > best_score:
                        best_score = new_score
                        best_q = q
                        best_new_K = new_K
                
                if best_q != current_r:
                    residues[p] = best_q
                    K = best_new_K
                    current_score = best_score
                    improved = True
            
            if not improved:
                break
        
        # Construct S
        S = set()
        for x in range(N):
            if not (K >> x) & 1:
                S.add(x)
                
        return S, residues

    # Iterative refinement
    for _ in range(20): # Max iterations to add primes
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        S, residues = optimize(P, residues)
        k = len(S)
        
        if k > best_size:
            best_S = S
            best_size = k
            
        # Check admissibility
        is_adm = True
        violating_p = None
        for p in all_primes:
            if p > k:
                break
            if p in residues:
                # We already constrained this prime, so it should be missed
                # But let's verify just in case
                pass
            else:
                # Check if S covers all residues mod p
                seen = bytearray(p)
                count = 0
                for v in S:
                    r = v % p
                    if not seen[r]:
                        seen[r] = 1
                        count += 1
                        if count == p:
                            is_adm = False
                            violating_p = p
                            break
            if not is_adm:
                break
        
        if is_adm:
            break
        else:
            # Add violating prime to P
            P.append(violating_p)
            residues[violating_p] = random.randint(0, violating_p - 1)
            # Note: We should probably re-optimize all residues, not just the new one
            # But the previous residues might be suboptimal with the new constraint.
            # The optimize function handles all P.

    return list(best_S)

# Initial call to set start time and warm up
construct()
