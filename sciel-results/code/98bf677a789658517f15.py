import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    
    # Generate candidate prime sets
    # Strategy: Take smallest primes, optionally excluding 2, 3, etc.
    candidate_sets = []
    
    # Set 1: All smallest primes
    s = Fraction(0, 1)
    set1 = []
    for p in primes:
        if s + Fraction(1, p) <= 1:
            set1.append(p)
            s += Fraction(1, p)
        else:
            break
    if set1:
        candidate_sets.append(set1)

    # Set 2: Exclude 2
    s = Fraction(0, 1)
    set2 = []
    for p in primes:
        if p == 2:
            continue
        if s + Fraction(1, p) <= 1:
            set2.append(p)
            s += Fraction(1, p)
        else:
            break
    if set2:
        candidate_sets.append(set2)
        
    # Set 3: Exclude 2, 3
    s = Fraction(0, 1)
    set3 = []
    for p in primes:
        if p in (2, 3):
            continue
        if s + Fraction(1, p) <= 1:
            set3.append(p)
            s += Fraction(1, p)
        else:
            break
    if set3:
        candidate_sets.append(set3)

    # Set 4: Exclude 2, 3, 5
    s = Fraction(0, 1)
    set4 = []
    for p in primes:
        if p in (2, 3, 5):
            continue
        if s + Fraction(1, p) <= 1:
            set4.append(p)
            s += Fraction(1, p)
        else:
            break
    if set4:
        candidate_sets.append(set4)

    # Remove duplicates and empty sets
    unique_sets = []
    seen = set()
    for s_set in candidate_sets:
        key = tuple(s_set)
        if key not in seen:
            seen.add(key)
            unique_sets.append(s_set)

    best_L_global = 0
    best_config_global = []

    L_MAX = 1000
    
    # Precompute masks for all primes in the union of candidate sets
    all_primes_needed = set()
    for s_set in unique_sets:
        all_primes_needed.update(s_set)
        
    prime_masks = {}
    for p in all_primes_needed:
        masks = []
        for r in range(p):
            m = 0
            first_x = r if r > 0 else p
            if first_x <= L_MAX:
                for x in range(first_x, L_MAX + 1, p):
                    m |= (1 << (x - 1))
            masks.append(m)
        prime_masks[p] = masks

    def evaluate_L(residues, prime_set):
        total_mask = 0
        for p, r in residues:
            total_mask |= prime_masks[p][r]
        # Calculate L
        if total_mask == 0:
            return 0
        # L is the number of trailing 1s in the binary representation
        # (total_mask ^ (total_mask + 1)) has 1s in the trailing block
        # Example: 0b...0111 -> 0b...0111 ^ 0b...1000 = 0b...1111
        # bit_length of 0b1111 is 4. L = 4 - 1 = 3.
        L = (total_mask ^ (total_mask + 1)).bit_length() - 1
        return L

    def local_search(prime_set, time_limit):
        n_primes = len(prime_set)
        if n_primes == 0:
            return 0, []
        
        # Initialize with random residues
        current_residues = [random.randint(0, p - 1) for p in prime_set]
        current_L = evaluate_L(list(zip(prime_set, current_residues)), prime_set)
        
        best_L = current_L
        best_residues = current_residues[:]
        
        t_start = time.time()
        T = 1.0
        cooling = 0.995
        
        while time.time() - t_start < time_limit:
            # Pick a random prime index
            idx = random.randint(0, n_primes - 1)
            p = prime_set[idx]
            
            # Pick a new random residue
            old_r = current_residues[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
                
            current_residues[idx] = new_r
            new_L = evaluate_L(list(zip(prime_set, current_residues)), prime_set)
            
            delta = new_L - current_L
            
            if delta > 0 or random.random() < pow(2.718, delta / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current_residues[:]
            else:
                current_residues[idx] = old_r
                
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_L, best_residues

    # Run local search on each candidate set
    # Allocate time based on number of sets
    time_per_set = (deadline - time.time()) / len(unique_sets)
    
    for s_set in unique_sets:
        if time.time() > deadline - 5:
            break
            
        # Run multiple restarts if time permits
        # For simplicity, one long run per set
        L, residues = local_search(s_set, time_per_set * 0.8)
        
        if L > best_L_global:
            best_L_global = L
            # Format result
            best_config_global = [[p, r] for p, r in zip(s_set, residues)]
            
    # Final check and return
    if not best_config_global:
        # Fallback
        return [[2, 0], [3, 0], [5, 0]]
        
    return best_config_global
