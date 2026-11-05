import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 1000
PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)

def _sum_reciprocals_exact(classes):
    """Exact Fraction sum for final verification."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def _sum_reciprocals_float(classes):
    """Fast float sum of reciprocals for budget checking."""
    return sum(1.0 / p for p, r in classes)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # We will search for the best set of primes and their residues.
    # The core engine is a local search on residues for a fixed set of primes.
    # We will try multiple candidate sets of primes.
    
    # Candidate generation strategies
    candidates = []
    
    # 1. Greedy Small Primes (fill budget with smallest primes)
    s = Fraction(0)
    res = []
    for p in PRIMES:
        if s + Fraction(1, p) <= 1:
            res.append(p)
            s += Fraction(1, p)
    candidates.append(res)
    
    # 2. Greedy Large Primes (fill budget with largest primes)
    s = Fraction(0)
    res = []
    for p in reversed(PRIMES):
        if s + Fraction(1, p) <= 1:
            res.append(p)
            s += Fraction(1, p)
    candidates.append(res)
    
    # 3. Random Sets
    random.seed(42)
    for _ in range(15):
        s = Fraction(0)
        res = []
        available = list(PRIMES)
        random.shuffle(available)
        for p in available:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        candidates.append(res)
        
    # 4. "Balanced" Sets: Mix of small and medium primes
    # Try to pick primes such that sum is close to 1 but not dominated by tiny primes
    # e.g. skip 2,3? Or start from middle?
    for start_idx in [0, 2, 5, 10, 20]:
        s = Fraction(0)
        res = []
        for p in PRIMES[start_idx:]:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        candidates.append(res)
        
    # Remove duplicates (by sorted tuple)
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates
    
    best_L = 0
    best_classes = []
    
    # Optimization Engine
    # Uses NumPy for fast coverage counting.
    # MAX_L can be large, but we only care about the first gap.
    # However, to correctly find the gap, we need to check up to a reasonable limit.
    # If we cover 1..1000, the gap is likely not too far beyond.
    # But to be safe, let's check up to 5000 or 10000.
    # Actually, if we have a good covering system, the gaps might be sparse.
    # Let's use a limit of 20000 for the search to be safe, but optimize speed.
    # With ~20-30 primes, updating counts is O(N/p).
    # N=20000. p=2 -> 10000 ops. p=997 -> 20 ops.
    # Total ops per move ~ sum(20000/p).
    # If we have 30 primes, average p ~ 50? Sum ~ 30 * 400 = 12000 ops.
    # Very fast in NumPy.
    
    MAX_L = 20000
    
    def optimize(prime_set):
        if not prime_set:
            return 0, []
            
        # Check budget
        s = sum(Fraction(1, p) for p in prime_set)
        if s > 1:
            return 0, []
            
        P_sorted = sorted(prime_set)
        n_primes = len(P_sorted)
        
        # Initialize residues to 0 (or random? 0 is fine, local search will move them)
        # Actually, random initialization might help avoid local minima?
        # Let's stick to 0 for determinism and speed, or maybe n%p?
        # Let's try random residues for the first prime? No, 0 is a valid start.
        
        residues = [0] * n_primes
        # Map prime to index in P_sorted for quick lookup?
        # We iterate over P_sorted directly.
        
        # Counts array
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        # Initial application
        for i, p in enumerate(P_sorted):
            r = residues[i]
            # Indices: r, r+p, r+2p...
            # In numpy, counts[r::p] += 1
            # Note: if r=0, it covers 0, p, 2p...
            # We care about 1..MAX_L.
            # Setting index 0 is harmless.
            counts[r::p] += 1
            
        # Ensure index 0 is "covered" so it doesn't break logic if we check from 0
        # But we check from 1.
        
        def get_L_from_counts(c):
            # Find first index i >= 1 where c[i] == 0
            # We can use np.where on c[1:]
            # If no zeros, return MAX_L
            zero_indices = np.where(c[1:] == 0)[0]
            if len(zero_indices) == 0:
                return MAX_L
            return int(zero_indices[0]) + 1

        current_L = get_L_from_counts(counts)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 100  # Allow more exploration
        
        # Time limit per candidate?
        # We have ~20 candidates. Total time 140s. ~7s per candidate.
        # Each iteration is very fast. We can do thousands of iterations.
        
        while time.time() < deadline - 5.0:
            if current_L >= MAX_L:
                break
                
            n = current_L # First uncovered number
            
            best_move = None
            best_L_new = current_L
            
            # Try changing each prime's residue to cover n
            for i, p in enumerate(P_sorted):
                r_new = n % p
                if residues[i] == r_new:
                    continue
                
                r_old = residues[i]
                
                # Apply move
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                
                new_L = get_L_from_counts(counts)
                
                # Revert move
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                
                if new_L > best_L_new:
                    best_L_new = new_L
                    best_move = (i, r_old, r_new)
                    
            if best_move:
                i, r_old, r_new = best_move
                residues[i] = r_new
                counts[r_old::p] -= 1  # Wait, p is from loop above, need to get it from P_sorted[i]
                # Fix: p = P_sorted[i]
                p = P_sorted[i]
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                current_L = best_L_new
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Stuck in local minimum.
                    # Try a random perturbation?
                    # Pick a random prime and change its residue randomly.
                    # This allows escaping local minima.
                    if random.random() < 0.1: # 10% chance to perturb
                        idx = random.randint(0, n_primes - 1)
                        p = P_sorted[idx]
                        r_old = residues[idx]
                        r_new = random.randint(0, p - 1)
                        if r_new != r_old:
                            counts[r_old::p] -= 1
                            counts[r_new::p] += 1
                            residues[idx] = r_new
                            current_L = get_L_from_counts(counts)
                            no_improve_count = 0
                    else:
                        break
                        
        classes = [[P_sorted[i], residues[i]] for i in range(n_primes)]
        return current_L, classes

    for P in candidates:
        if time.time() > deadline - 10.0:
            break
        L, classes = optimize(P)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    # Final verification
    if best_classes:
        if _sum_reciprocals_exact(best_classes) > 1:
            # Fallback
            return []
            
    return best_classes

# EVOLVE-BLOCK-END
