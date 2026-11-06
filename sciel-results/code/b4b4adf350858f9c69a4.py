import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    
    # Define candidate prime sets based on budget analysis
    # Set 1: {3, 5} + primes >= 100
    # Sum(3)+Sum(5) = 0.53. Sum(p>=100) approx 0.41. Total 0.94.
    set1 = [3, 5] + [p for p in primes if p >= 100]
    
    # Set 2: {3, 5, 7} + primes >= 150
    # Sum(3)+Sum(5)+Sum(7) = 0.67. Sum(p>=150) approx 0.33. Total 1.0.
    set2 = [3, 5, 7] + [p for p in primes if p >= 150]
    
    # Set 3: {2} + primes >= 67
    # Sum(2) = 0.5. Sum(p>=67) approx 0.5. Total 1.0.
    set3 = [2] + [p for p in primes if p >= 67]
    
    # Set 4: {2, 3} + primes >= 330
    # Sum(2)+Sum(3) = 0.83. Sum(p>=330) approx 0.17. Total 1.0.
    set4 = [2, 3] + [p for p in primes if p >= 330]
    
    candidate_sets = [set1, set2, set3, set4]
    
    # Filter sets that are actually valid (sum <= 1)
    valid_sets = []
    for P in candidate_sets:
        # Use exact rationals for safety check
        from fractions import Fraction
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(P)
    
    if not valid_sets:
        # Fallback to seed-like if no valid sets found (should not happen)
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    best_L = 0
    best_result = []
    
    # Simulated Annealing Parameters
    # We will run SA for each valid set, allocating time equally
    time_per_set = 25  # 4 sets * 25s = 100s. 18s margin.
    
    for P in valid_sets:
        if time.time() > deadline - 5:
            break
            
        set_start = time.time()
        set_deadline = set_start + time_per_set
        
        # Initialize residues randomly
        residues = {p: random.randint(0, p - 1) for p in P}
        
        # Helper to compute L
        def compute_L(residues, P):
            # Max L we care about is 10000? No, capped at 10^5 but likely < 5000.
            # Let's check up to 5000 to be safe, but stop early if gap found.
            # Actually, we can just check 1..10000.
            # Optimization: Use an array for coverage counts?
            # For SA, we need fast evaluation.
            # Let's use a boolean array for 1..20000.
            # If L > 20000, we return 20000 (capped for speed).
            # But we want to maximize L.
            
            # To make it fast, we can precompute which numbers are covered by each prime?
            # No, residues change.
            
            # Simple loop:
            # For n from 1 to L_max:
            #   Check if any p covers n.
            #   If not, return n-1.
            
            # This is O(L * |P|).
            # |P| ~ 100. L ~ 2000. 200,000 ops.
            # In Python, ~0.05s.
            # 25s / 0.05s = 500 steps. Too few.
            
            # We need a faster evaluation or fewer primes.
            # Or we can use NumPy.
            
            # Let's try NumPy for the coverage check.
            # Create an array of 1..L_max.
            # For each p, mark n where n % p == r.
            # This is O(L) per prime.
            
            # Actually, we can just do the simple loop but optimize the inner check.
            # Or, we can maintain a "covered" array and update it incrementally in SA moves.
            
            # Incremental Update Strategy:
            # We maintain `cover_count` array of size L_max+1.
            # Initially 0.
            # For each p, r in residues:
            #   for n in range(r % p, L_max+1, p):
            #       if n >= 1: cover_count[n] += 1
            #       else: # handle n=0? We care about 1..L
            #            pass
            # Note: r is in [0, p-1].
            # If r == 0, covers p, 2p, ...
            # If r > 0, covers r, r+p, ...
            
            # When we change residue of p from r_old to r_new:
            # 1. Decrement cover_count for n in r_old mod p.
            # 2. Increment cover_count for n in r_new mod p.
            # 3. Update L by scanning from current L?
            #    Or scan from 1?
            #    Scanning from 1 is O(L).
            #    If we only change one prime, the change in L might be small.
            #    But we don't know.
            
            # Let's stick to recomputation for correctness first, then optimize if needed.
            # With NumPy, we can vectorize the check.
            
            # But SA moves are local.
            # Let's implement the incremental update.
            
            pass

        # Let's implement a simple SA with recomputation for now, 
        # but limit L_max to 3000 to keep it fast.
        # If we find L > 3000, we can extend.
        
        L_MAX = 3000
        cover_count = [0] * (L_MAX + 1)
        
        def initialize_coverage():
            for n in range(1, L_MAX + 1):
                cover_count[n] = 0
            for p, r in residues.items():
                start = r % p
                if start == 0:
                    start = p # 0 is not in 1..L_MAX, next is p
                # Actually, if r=0, n=p, 2p...
                # If r>0, n=r, r+p...
                # But r can be 0.
                # Range(start, L_MAX+1, p)
                # If start < 1, we need to adjust.
                # Since r < p, start is in [0, p-1].
                # If start == 0, first positive multiple is p.
                # If start > 0, first is start.
                first = start if start > 0 else p
                if first > L_MAX:
                    continue
                for n in range(first, L_MAX + 1, p):
                    cover_count[n] += 1

        def get_L():
            # Find largest L such that cover_count[1..L] > 0
            # Scan from 1
            for n in range(1, L_MAX + 1):
                if cover_count[n] == 0:
                    return n - 1
            return L_MAX

        # Initialize
        initialize_coverage()
        current_L = get_L()
        
        # SA Loop
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        
        best_set_L = current_L
        best_set_residues = residues.copy()
        
        while time.time() < set_deadline:
            # Pick a random prime
            p = random.choice(P)
            r_old = residues[p]
            
            # Generate new residue
            # Random walk or random jump?
            # Random jump is better for SA.
            r_new = random.randint(0, p - 1)
            
            if r_new == r_old:
                continue
                
            # Update cover_count
            # Decrement old
            first_old = r_old if r_old > 0 else p
            if first_old <= L_MAX:
                for n in range(first_old, L_MAX + 1, p):
                    cover_count[n] -= 1
                    
            # Increment new
            first_new = r_new if r_new > 0 else p
            if first_new <= L_MAX:
                for n in range(first_new, L_MAX + 1, p):
                    cover_count[n] += 1
            
            # Calculate new L
            new_L = get_L()
            
            # Acceptance
            if new_L > current_L or random.random() < math.exp((new_L - current_L) / T):
                current_L = new_L
                residues[p] = r_new
                if new_L > best_set_L:
                    best_set_L = new_L
                    best_set_residues = residues.copy()
            else:
                # Revert
                # Decrement new
                if first_new <= L_MAX:
                    for n in range(first_new, L_MAX + 1, p):
                        cover_count[n] -= 1
                # Increment old
                if first_old <= L_MAX:
                    for n in range(first_old, L_MAX + 1, p):
                        cover_count[n] += 1
                
            # Cool down
            T *= cooling
            
        # End of SA for this set
        if best_set_L > best_L:
            best_L = best_set_L
            best_result = [[p, best_set_residues[p]] for p in best_set_residues]

    # Final verification and return
    if not best_result:
        # Fallback
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    # Verify best_result is valid
    # Check sum
    from fractions import Fraction
    s = sum(Fraction(1, p) for p, r in best_result)
    if s > 1:
        # This shouldn't happen, but if it does, fallback
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result
