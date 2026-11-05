import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    MAX_L = 100000
    
    def get_L_from_counts(c):
        zero_indices = np.where(c[1:] == 0)[0]
        if len(zero_indices) == 0:
            return MAX_L
        return int(zero_indices[0]) + 1

    def optimize(prime_set, time_budget=10):
        # Check budget
        s = sum(Fraction(1, p) for p in prime_set)
        if s > 1:
            return 0, []
            
        P_sorted = sorted(prime_set)
        residues = {p: 0 for p in P_sorted}
        
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        # Initial application
        for p in P_sorted:
            r = residues[p]
            counts[r::p] += 1
            
        counts[0] = 10 # Ensure 0 is covered, though irrelevant
        
        current_L = get_L_from_counts(counts)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 50
        local_deadline = start_time + time_budget
        
        while time.time() < local_deadline:
            if current_L == MAX_L:
                break
                
            n = current_L # First uncovered
            best_move = None
            best_L = current_L
            
            # Try changing each prime's residue to cover n
            for p in P_sorted:
                r_new = n % p
                if residues[p] == r_new:
                    continue
                
                r_old = residues[p]
                
                # Apply move
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                
                new_L = get_L_from_counts(counts)
                
                # Revert move
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                
                if new_L > best_L:
                    best_L = new_L
                    best_move = (p, r_old, r_new)
                    
            if best_move:
                p, r_old, r_new = best_move
                residues[p] = r_new
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                current_L = best_L
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                    
        classes = [[p, residues[p]] for p in P_sorted]
        return current_L, classes

    # Generate Candidates
    candidates = []
    
    # 1. Greedy Small Primes
    def build_greedy_small():
        s = Fraction(0)
        res = []
        for p in primes:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        return res
        
    # 2. Greedy Large Primes
    def build_greedy_large():
        s = Fraction(0)
        res = []
        for p in reversed(primes):
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        return res
        
    # 3. Random Sets
    random.seed(42)
    for _ in range(20):
        s = Fraction(0)
        res = []
        # Pick random primes, ensure distinct
        available = list(primes)
        random.shuffle(available)
        for p in available:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        candidates.append(res)
        
    candidates.append(build_greedy_small())
    candidates.append(build_greedy_large())
    
    best_L = 0
    best_classes = []
    
    # Time management: allocate time per candidate
    # We have ~120s total. 
    # Let's try to fit as many as possible.
    # Initial estimate: 2s per candidate?
    # If a candidate takes 2s and yields L=1000, we keep it.
    
    # Sort candidates? Maybe not, random sets vary.
    
    # Dynamic time allocation
    remaining = deadline - start_time
    num_candidates = len(candidates)
    time_per_candidate = max(1.0, (remaining - 5) / num_candidates) # At least 1s, leave 5s buffer
    
    for i, P in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Adjust time per candidate based on remaining time and candidates left
        remaining_time = deadline - 5 - elapsed
        candidates_left = num_candidates - i
        time_per_candidate = max(0.5, remaining_time / candidates_left)
        
        if not P:
            continue
            
        L, classes = optimize(P, time_budget=time_per_candidate)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    return best_classes
