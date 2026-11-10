"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_coeffs = None
    best_nonzero = None
    
    def count_nonzero_square(coeffs):
        """Count nonzero terms in the exact square of a polynomial given as list."""
        n = len(coeffs)
        # Square: result has degree up to 2*(n-1)
        result = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            ci = coeffs[i]
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                cj = coeffs[j]
                deg = i + j
                val = ci * cj
                if i == j:
                    result[deg] += val
                else:
                    result[deg] += 2 * val
        # Count nonzero
        count = 0
        for v in result:
            if v != 0:
                count += 1
        return count
    
    def is_valid(coeffs):
        """Check all constraints."""
        if len(coeffs) > 41:
            return False
        if not coeffs:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        return True
    
    def generate_candidate():
        """Generate a random 12-term polynomial with degree <= 40."""
        # Choose 12 distinct degrees from [0, 40]
        degrees = random.sample(range(41), 12)
        degrees.sort()
        coeffs = [0] * (max(degrees) + 1)
        for d in degrees:
            # Coefficients: mostly ±1, occasionally small integers
            if random.random() < 0.8:
                coeffs[d] = random.choice([-1, 1])
            else:
                coeffs[d] = random.choice([-3, -2, -1, 1, 2, 3])
        return coeffs
    
    def local_improve(coeffs, max_iter=500):
        """Try to reduce nonzero terms in square by flipping signs or adjusting coefficients."""
        current = count_nonzero_square(coeffs)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping sign of one term
            nonzero_indices = [i for i, c in enumerate(coeffs) if c != 0]
            idx = random.choice(nonzero_indices)
            old_val = coeffs[idx]
            # Try sign flip
            coeffs[idx] = -old_val
            new_count = count_nonzero_square(coeffs)
            if new_count < current:
                current = new_count
                improved = True
            else:
                coeffs[idx] = old_val
            
            # Try changing coefficient to another small value
            if not improved:
                new_val = random.choice([-3, -2, -1, 1, 2, 3])
                if new_val != old_val:
                    coeffs[idx] = new_val
                    new_count = count_nonzero_square(coeffs)
                    if new_count < current:
                        current = new_count
                        improved = True
                    else:
                        coeffs[idx] = old_val
            
            if not improved:
                # Try moving a term to a different degree
                idx2 = random.choice(nonzero_indices)
                old_deg = idx2
                old_val = coeffs[old_deg]
                # Find a new degree
                new_deg = random.randint(0, 40)
                if new_deg != old_deg and coeffs[new_deg] == 0:
                    coeffs[old_deg] = 0
                    # Extend if needed
                    if new_deg >= len(coeffs):
                        coeffs.extend([0] * (new_deg - len(coeffs) + 1))
                    coeffs[new_deg] = old_val
                    new_count = count_nonzero_square(coeffs)
                    if new_count < current:
                        current = new_count
                        improved = True
                    else:
                        coeffs[new_deg] = 0
                        coeffs[old_deg] = old_val
            
            if time.time() > deadline:
                break
        
        return coeffs, current
    
    # Phase 1: Random restarts with local improvement
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        coeffs = generate_candidate()
        if not is_valid(coeffs):
            continue
        
        coeffs, nz = local_improve(coeffs, max_iter=200)
        
        if best_nonzero is None or nz < best_nonzero:
            best_nonzero = nz
            best_coeffs = coeffs[:]
        
        # If we find a very good solution, do more polishing
        if best_nonzero <= 15:
            coeffs, nz = local_improve(coeffs, max_iter=2000)
            if nz < best_nonzero:
                best_nonzero = nz
                best_coeffs = coeffs[:]
        
        # Safety check
        if time.time() > deadline - 5.0:
            break
    
    # Ensure we have a valid result
    if best_coeffs is None or not is_valid(best_coeffs):
        # Fallback: a known valid 12-term polynomial
        # (1+x)^11 has 12 terms, square (1+x)^22 has 23 terms
        best_coeffs = [1] * 12
        best_nonzero = count_nonzero_square(best_coeffs)
    
    return best_coeffs
# EVOLVE-BLOCK-END
