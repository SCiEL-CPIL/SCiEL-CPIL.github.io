import random
import itertools
import numpy as np

def construct():
    best_count = 23  # baseline
    best_coeffs = [1]*12
    
    # Use arithmetic progression exponents 0..11
    # Try to find coefficients that minimize nonzero terms in square
    # Square coefficient at position k (0..22) is sum_{i+j=k} a_i*a_j
    
    def count_nonzero(coeffs):
        n = len(coeffs)
        square = [0]*(2*n-1)
        for i in range(n):
            for j in range(n):
                square[i+j] += coeffs[i]*coeffs[j]
        return sum(1 for x in square if x != 0)
    
    # Local search
    random.seed(42)
    for trial in range(50000):
        # Start with random small coefficients
        coeffs = [random.choice([-2,-1,1,2]) for _ in range(12)]
        # Ensure nonzero
        for i in range(12):
            if coeffs[i] == 0:
                coeffs[i] = 1
        
        current = count_nonzero(coeffs)
        if current < best_count:
            best_count = current
            best_coeffs = coeffs[:]
        
        # Local search: try changing one coefficient
        improved = True
        while improved and current < best_count:
            improved = False
            for i in range(12):
                for delta in [-2,-1,1,2]:
                    new_c = coeffs[i] + delta
                    if -64 <= new_c <= 64 and new_c != 0:
                        old = coeffs[i]
                        coeffs[i] = new_c
                        new_count = count_nonzero(coeffs)
                        if new_count < current:
                            current = new_count
                            improved = True
                            if current < best_count:
                                best_count = current
                                best_coeffs = coeffs[:]
                        else:
                            coeffs[i] = old
    
    # Also try some structured patterns
    # Pattern: alternating signs
    patterns = [
        [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
        [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
        [1,-2,1,-2,1,-2,1,-2,1,-2,1,-2],
        [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
    ]
    for p in patterns:
        c = count_nonzero(p)
        if c < best_count:
            best_count = c
            best_coeffs = p[:]
    
    # Final local search from best
    current = count_nonzero(best_coeffs)
    improved = True
    while improved:
        improved = False
        for i in range(12):
            for new_c in range(-64, 65):
                if new_c == 0 or new_c == best_coeffs[i]:
                    continue
                old = best_coeffs[i]
                best_coeffs[i] = new_c
                new_count = count_nonzero(best_coeffs)
                if new_count < current:
                    current = new_count
                    improved = True
                else:
                    best_coeffs[i] = old
    
    # Return as coefficient list in ascending degree
    # Exponents 0..11, so list is best_coeffs
    return best_coeffs
