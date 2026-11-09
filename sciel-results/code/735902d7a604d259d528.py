import random
import numpy as np

def construct():
    def count_nonzero_square(coeffs):
        """Count nonzero terms in the exact integer square of coeffs."""
        n = len(coeffs)
        max_deg = 2 * n - 1
        square = [0] * (max_deg + 1)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(n):
                cj = coeffs[j]
                if cj == 0:
                    continue
                square[i + j] += ci * cj
        return sum(1 for x in square if x != 0)
    
    def build_poly_from_decomposition(degs_A, coefs_A, degs_B, coefs_B):
        """Build P = A + B from sparse representations."""
        max_deg = max(max(degs_A) if degs_A else 0, max(degs_B) if degs_B else 0)
        P = [0] * (max_deg + 1)
        for d, c in zip(degs_A, coefs_A):
            P[d] += c
        for d, c in zip(degs_B, coefs_B):
            P[d] += c
        # Trim trailing zeros
        while len(P) > 0 and P[-1] == 0:
            P.pop()
        return P
    
    def get_nonzero_count(P):
        return sum(1 for c in P if c != 0)
    
    best_count = 999
    best_coeffs = [1] * 12
    
    # Strategy: P = A + B where A has 6 terms, B has 6 terms
    # Degrees for A: {0, 3, 6, 9, 12, 15}
    # Degrees for B: {1, 4, 7, 10, 13, 16}
    # This gives P 12 terms at degrees {0,1,3,4,6,7,9,10,12,13,15,16}
    degs_A = [0, 3, 6, 9, 12, 15]
    degs_B = [1, 4, 7, 10, 13, 16]
    
    # Simulated annealing on coefficients
    allowed_coefs = [-2, -1, 1, 2]
    
    def sa_borwein_erdelyi(seed, steps=30000):
        random.seed(seed)
        current_coefs_A = [random.choice(allowed_coefs) for _ in range(6)]
        current_coefs_B = [random.choice(allowed_coefs) for _ in range(6)]
        
        P = build_poly_from_decomposition(degs_A, current_coefs_A, degs_B, current_coefs_B)
        if get_nonzero_count(P) != 12:
            # Adjust to ensure 12 nonzero terms
            for i in range(6):
                if P[degs_A[i]] == 0:
                    current_coefs_A[i] = 1
                if P[degs_B[i]] == 0:
                    current_coefs_B[i] = 1
            P = build_poly_from_decomposition(degs_A, current_coefs_A, degs_B, current_coefs_B)
        
        current_score = count_nonzero_square(P)
        best_local_score = current_score
        best_local_coefs_A = current_coefs_A[:]
        best_local_coefs_B = current_coefs_B[:]
        
        for i in range(steps):
            temp = 3.0 * (0.01 / 3.0) ** (i / steps)
            
            # Pick which polynomial to modify
            if random.random() < 0.5:
                # Modify A
                idx = random.randint(0, 5)
                old_val = current_coefs_A[idx]
                new_val = random.choice(allowed_coefs)
                if new_val == old_val:
                    continue
                current_coefs_A[idx] = new_val
                P = build_poly_from_decomposition(degs_A, current_coefs_A, degs_B, current_coefs_B)
                if get_nonzero_count(P) != 12:
                    current_coefs_A[idx] = old_val
                    continue
            else:
                # Modify B
                idx = random.randint(0, 5)
                old_val = current_coefs_B[idx]
                new_val = random.choice(allowed_coefs)
                if new_val == old_val:
                    continue
                current_coefs_B[idx] = new_val
                P = build_poly_from_decomposition(degs_A, current_coefs_A, degs_B, current_coefs_B)
                if get_nonzero_count(P) != 12:
                    current_coefs_B[idx] = old_val
                    continue
            
            new_score = count_nonzero_square(P)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_coefs_A = current_coefs_A[:]
                    best_local_coefs_B = current_coefs_B[:]
            else:
                if random.random() < 0.5:
                    current_coefs_A[idx] = old_val
                else:
                    current_coefs_B[idx] = old_val
        
        P = build_poly_from_decomposition(degs_A, best_local_coefs_A, degs_B, best_local_coefs_B)
        return P, best_local_score
    
    # Run multiple SA instances
    for run in range(10):
        P, score = sa_borwein_erdelyi(seed=1000 + run, steps=20000)
        if score < best_count:
            best_count = score
            best_coeffs = P[:]
    
    # Local search refinement
    current_score = count_nonzero_square(best_coeffs)
    improved = True
    while improved:
        improved = False
        for i in range(len(best_coeffs)):
            if best_coeffs[i] == 0:
                continue
            for new_c in [-2, -1, 1, 2]:
                if new_c == 0 or new_c == best_coeffs[i]:
                    continue
                old = best_coeffs[i]
                best_coeffs[i] = new_c
                if get_nonzero_count(best_coeffs) != 12:
                    best_coeffs[i] = old
                    continue
                new_count = count_nonzero_square(best_coeffs)
                if new_count < current_score:
                    current_score = new_count
                    best_count = new_count
                    improved = True
                else:
                    best_coeffs[i] = old
    
    # Ensure exactly 12 nonzero terms and valid bounds
    if get_nonzero_count(best_coeffs) != 12:
        # Fallback: try a known good structure
        best_coeffs = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
        while get_nonzero_count(best_coeffs) != 12:
            # This shouldn't happen, but just in case
            best_coeffs = [1] * 12
            break
        if get_nonzero_count(best_coeffs) != 12:
            best_coeffs = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    
    # Ensure all coefficients are in [-64, 64]
    for i in range(len(best_coeffs)):
        if best_coeffs[i] > 64:
            best_coeffs[i] = 64
        elif best_coeffs[i] < -64:
            best_coeffs[i] = -64
    
    # Ensure nonzero final coefficient
    if best_coeffs[-1] == 0:
        best_coeffs[-1] = 1
    
    return best_coeffs
