import time
import random
import math
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Generate primes up to 1000
    primes = _primes_upto(1000)
    
    MAX_L = 100000
    # Use NumPy array for speed
    covered_count = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def update_coverage(p, r, delta):
        """Update covered_count for prime p, residue r, adding or removing delta."""
        first = r if r != 0 else p
        if first < 1:
            first += p * ((1 - first + p - 1) // p)
        # Vectorized update
        # Indices: first, first+p, ...
        # slice: [first : MAX_L+1 : p]
        covered_count[first:MAX_L+1:p] += delta

    def compute_L():
        """Compute the largest L such that covered_count[1..L] > 0."""
        # Find first index where covered_count == 0
        # We need the first zero in covered_count[1:]
        # np.nonzero(covered_count[1:] == 0)[0]
        # If empty, L = MAX_L
        zeros = np.nonzero(covered_count[1:] == 0)[0]
        if len(zeros) == 0:
            return MAX_L
        else:
            return zeros[0]

    # Initialize
    # Start with primes >= 13
    initial_primes = [p for p in primes if p >= 13]
    initial_primes.sort(reverse=True) # Remove largest first if budget exceeded
    
    current_classes = []
    current_budget = Fraction(0)
    for p in initial_primes:
        cost = Fraction(1, p)
        if current_budget + cost <= 1:
            r = random.randint(0, p - 1)
            current_classes.append([p, r])
            current_budget += cost
            update_coverage(p, r, 1)
            
    best_L = compute_L()
    best_classes = [c[:] for c in current_classes]
    best_budget = current_budget
    
    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.7:
            # Change residue
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            p, r_old = current_classes[idx]
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                continue
            
            update_coverage(p, r_old, -1)
            update_coverage(p, r_new, 1)
            current_classes[idx] = [p, r_new]
            
            new_L = compute_L()
            
            if new_L > best_L:
                best_L = new_L
                best_classes = [c[:] for c in current_classes]
                best_budget = current_budget
            elif new_L >= best_L or random.random() < math.exp((new_L - best_L) / T):
                pass
            else:
                update_coverage(p, r_new, -1)
                update_coverage(p, r_old, 1)
                current_classes[idx] = [p, r_old]
                
        elif move_type < 0.9:
            # Swap prime
            if not current_classes:
                continue
            idx_out = random.randint(0, len(current_classes) - 1)
            p_out, r_out = current_classes[idx_out]
            
            current_primes_set = {c[0] for c in current_classes}
            candidates = [p for p in primes if p not in current_primes_set]
            if not candidates:
                continue
            
            p_in = random.choice(candidates)
            cost_in = Fraction(1, p_in)
            cost_out = Fraction(1, p_out)
            
            new_budget = current_budget - cost_out + cost_in
            if new_budget > 1:
                continue
            
            update_coverage(p_out, r_out, -1)
            r_in = random.randint(0, p_in - 1)
            update_coverage(p_in, r_in, 1)
            
            current_classes[idx_out] = [p_in, r_in]
            current_budget = new_budget
            
            new_L = compute_L()
            
            if new_L > best_L:
                best_L = new_L
                best_classes = [c[:] for c in current_classes]
                best_budget = current_budget
            elif new_L >= best_L or random.random() < math.exp((new_L - best_L) / T):
                pass
            else:
                update_coverage(p_in, r_in, -1)
                update_coverage(p_out, r_out, 1)
                current_classes[idx_out] = [p_out, r_out]
                current_budget = current_budget - cost_in + cost_out
                
        else:
            # Add or Remove
            if random.random() < 0.5:
                # Add
                current_primes_set = {c[0] for c in current_classes}
                candidates = [p for p in primes if p not in current_primes_set]
                if not candidates:
                    continue
                p_in = random.choice(candidates)
                cost_in = Fraction(1, p_in)
                if current_budget + cost_in > 1:
                    continue
                
                r_in = random.randint(0, p_in - 1)
                update_coverage(p_in, r_in, 1)
                current_classes.append([p_in, r_in])
                current_budget += cost_in
                
                new_L = compute_L()
                
                if new_L > best_L:
                    best_L = new_L
                    best_classes = [c[:] for c in current_classes]
                    best_budget = current_budget
                elif new_L >= best_L or random.random() < math.exp((new_L - best_L) / T):
                    pass
                else:
                    update_coverage(p_in, r_in, -1)
                    current_classes.pop()
                    current_budget -= cost_in
            else:
                # Remove
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                p_out, r_out = current_classes[idx]
                cost_out = Fraction(1, p_out)
                
                update_coverage(p_out, r_out, -1)
                current_classes.pop(idx)
                current_budget -= cost_out
                
                new_L = compute_L()
                
                if new_L > best_L:
                    best_L = new_L
                    best_classes = [c[:] for c in current_classes]
                    best_budget = current_budget
                elif new_L >= best_L or random.random() < math.exp((new_L - best_L) / T):
                    pass
                else:
                    update_coverage(p_out, r_out, 1)
                    current_classes.insert(idx, [p_out, r_out])
                    current_budget += cost_out

        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_classes

def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

# EVOLVE-BLOCK-END
