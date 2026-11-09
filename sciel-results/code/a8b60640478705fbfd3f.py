"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """
    Returns a list of coefficients for a 12-term integer polynomial 
    (length <= 41, nonzero final coeff, entries in [-64, 64]) 
    that minimizes the number of nonzero terms in its exact integer square.
    """
    deadline = time.time() + 85.0  # 90s kill, keep 5s safety margin
    
    # Helper to compute square coefficients and count nonzeros
    def square_nonzero_count(poly):
        n = len(poly)
        if n == 0:
            return 0, []
        sq_len = 2 * n - 1
        sq = [0] * sq_len
        # Only iterate over nonzero terms for efficiency
        non_zero_indices = [(i, v) for i, v in enumerate(poly) if v != 0]
        for i, ci in non_zero_indices:
            for j, cj in non_zero_indices:
                sq[i + j] += ci * cj
        count = sum(1 for v in sq if v != 0)
        return count, sq

    # Helper to validate polynomial constraints
    def is_valid(poly):
        if len(poly) > 41:
            return False
        if len(poly) == 0:
            return False
        if poly[-1] == 0:
            return False
        if any(abs(v) > 64 for v in poly):
            return False
        # Exactly 12 nonzero terms
        if sum(1 for v in poly if v != 0) != 12:
            return False
        return True

    # Initial candidate: 12 terms, all 1s, degrees 0..11
    # Square has 23 nonzeros.
    best_poly = [1] * 12
    best_count, _ = square_nonzero_count(best_poly)
    
    # Randomized Local Search
    # State representation: list of (degree, coeff)
    # We represent the polynomial as a dense list for easy square calculation,
    # but operate on the sparse structure for mutations.
    
    def poly_to_dense(terms):
        if not terms:
            return []
        max_deg = max(d for d, c in terms if c != 0)
        dense = [0] * (max_deg + 1)
        for d, c in terms:
            if c != 0:
                dense[d] = c
        return dense

    # Start with a random sparse polynomial with 12 terms
    # Exponents in [0, 40]
    random.seed(42)
    
    def random_poly():
        degs = random.sample(range(41), 12)
        degs.sort()
        coeffs = [random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8,
                                  1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8]) for _ in range(12)]
        # Ensure last coeff is non-zero (it is by construction)
        terms = [(d, c) for d, c in zip(degs, coeffs)]
        # Ensure exactly 12 nonzero: if any coeff is 0, replace
        for i in range(12):
            if terms[i][1] == 0:
                terms[i] = (terms[i][0], 1)
        return terms

    current_terms = random_poly()
    current_dense = poly_to_dense(current_terms)
    current_count, _ = square_nonzero_count(current_dense)
    
    if current_count < best_count:
        best_count = current_count
        best_poly = current_dense

    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Mutation strategies:
        # 1. Change a coefficient
        # 2. Change an exponent
        # 3. Swap exponents
        # 4. Random restart if stuck? No, just keep mutating.
        
        new_terms = list(current_terms)
        mutation_type = random.randint(0, 2)
        
        if mutation_type == 0:
            # Change coefficient of a random term
            idx = random.randint(0, 11)
            old_c = new_terms[idx][1]
            new_c = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8])
            while new_c == old_c:
                new_c = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8])
            new_terms[idx] = (new_terms[idx][0], new_c)
            
        elif mutation_type == 1:
            # Change exponent of a random term
            idx = random.randint(0, 11)
            old_d = new_terms[idx][0]
            new_d = random.randint(0, 40)
            # Avoid collision with other exponents
            other_degs = {new_terms[i][0] for i in range(12) if i != idx}
            if new_d in other_degs:
                # Try a few times to find a unique degree
                for _ in range(10):
                    new_d = random.randint(0, 40)
                    if new_d not in other_degs:
                        break
                else:
                    # If still collision, skip mutation or force unique
                    # For simplicity, if we can't find unique, revert
                    continue
            new_terms[idx] = (new_d, new_terms[idx][1])
            
        else:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            di, ci = new_terms[i]
            dj, cj = new_terms[j]
            new_terms[i] = (dj, ci)
            new_terms[j] = (di, cj)

        # Sort by degree for consistent dense representation
        new_terms.sort(key=lambda x: x[0])
        
        # Check validity (exactly 12 terms, non-zero coeffs)
        if len(set(d for d, c in new_terms)) != 12:
            continue
        if any(c == 0 for d, c in new_terms):
            continue
            
        new_dense = poly_to_dense(new_terms)
        
        # Check length constraint
        if len(new_dense) > 41:
            continue
            
        new_count, _ = square_nonzero_count(new_dense)
        
        # Accept if better or equal (to allow plateau traversal) with some probability
        if new_count < current_count:
            current_terms = new_terms
            current_dense = new_dense
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_poly = current_dense
        elif new_count == current_count:
            # Accept with 20% probability to escape local minima
            if random.random() < 0.2:
                current_terms = new_terms
                current_dense = new_dense
                # current_count remains same
        else:
            # Reject worse moves
            pass
            
        # Simple annealing-like restart? 
        # If stuck for too long, maybe randomize?
        # For now, rely on the 85s budget.
        
    # Final validation of best_poly
    if not is_valid(best_poly):
        # Fallback to seed
        best_poly = [1] * 12
        
    return best_poly
# EVOLVE-BLOCK-END
