import time
from fractions import Fraction
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Sieve
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Helper to build a set
    def build_set(skip_set, force_set=None):
        if force_set is None: force_set = []
        s = sum(Fraction(1, p) for p in force_set)
        final_set = list(force_set)
        for p in primes:
            if p in skip_set or p in force_set:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return sorted(final_set)
        
    # Generate Candidates
    candidates = []
    seen = set()
    
    # 1. Greedy
    c = build_set([])
    candidates.append(c)
    seen.add(tuple(c))
    
    # 2. Skip small primes
    for skip in [{2}, {3}, {5}, {2,3}, {2,5}, {3,5}, {2,3,5}, {2,3,7}, {3,7,11}]:
        c = build_set(skip)
        if tuple(c) not in seen:
            candidates.append(c)
            seen.add(tuple(c))
            
    # 3. Specific sets from inspiration
    # Example: [13, 17, ...]
    # We'll just rely on the skip sets which are systematic.
    
    best_L = 0
    best_classes = []
    
    max_L_check = 100000
    
    for P in candidates:
        if time.time() > deadline - 10:
            break
        if not P: continue
            
        # Initialize residues
        # Simple greedy init: cover 1, 2, 3...
        counts = [0] * (max_L_check + 1)
        residues = {}
        used_primes = set()
        
        # Greedy Init
        for n in range(1, max_L_check + 1):
            if counts[n] > 0: continue
            assigned = False
            for p in P:
                if p in used_primes: continue
                r = n % p
                residues[p] = r
                used_primes.add(p)
                start_k = r if r > 0 else p
                for k in range(start_k, max_L_check + 1, p):
                    counts[k] += 1
                assigned = True
                break
            if not assigned: break
            
        # Calculate L
        def get_L(c_arr):
            for i in range(1, max_L_check + 1):
                if c_arr[i] == 0:
                    return i - 1
            return max_L_check
            
        current_L = get_L(counts)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            gap = current_L + 1
            if gap > max_L_check: break
            
            # Try changing existing prime
            found_improvement = False
            for idx, p in enumerate(P):
                if p not in residues: continue
                old_r = residues[p]
                new_r = gap % p
                if new_r == old_r: continue
                
                # Tentative change
                # Remove old
                start_old = old_r if old_r > 0 else p
                for k in range(start_old, max_L_check + 1, p):
                    counts[k] -= 1
                # Add new
                start_new = new_r if new_r > 0 else p
                for k in range(start_new, max_L_check + 1, p):
                    counts[k] += 1
                
                new_L = get_L(counts)
                
                if new_L > current_L:
                    # Keep
                    residues[p] = new_r
                    current_L = new_L
                    improved = True
                    found_improvement = True
                    break # Restart loop with new gap
                else:
                    # Revert
                    for k in range(start_old, max_L_check + 1, p):
                        counts[k] += 1
                    for k in range(start_new, max_L_check + 1, p):
                        counts[k] -= 1
            
            if found_improvement:
                continue
                
            # Try adding new prime
            current_sum = sum(Fraction(1, p) for p in residues.keys())
            for p in P:
                if p in residues: continue
                if current_sum + Fraction(1, p) > 1: continue
                
                new_r = gap % p
                residues[p] = new_r
                start_k = new_r if new_r > 0 else p
                for k in range(start_k, max_L_check + 1, p):
                    counts[k] += 1
                
                new_L = get_L(counts)
                
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    break
                else:
                    # Revert
                    for k in range(start_k, max_L_check + 1, p):
                        counts[k] -= 1
                    del residues[p]
            
            if current_L > best_L:
                best_L = current_L
                best_classes = [[p, residues[p]] for p in residues]
                
    return best_classes
