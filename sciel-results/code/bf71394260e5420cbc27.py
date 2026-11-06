import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_count = 24  # seed: [1]*12 squares to 24 nonzero terms
    best_poly = [1] * 12
    
    # Helper: compute square and count nonzeros
    def square_nonzero_count(coeffs):
        # coeffs is list of ints, length <= 41
        n = len(coeffs)
        sq = np.convolve(np.array(coeffs, dtype=np.int64), np.array(coeffs, dtype=np.int64))
        return int(np.count_nonzero(sq))
    
    # Helper: validate and trim
    def validate_and_trim(coeffs):
        # Ensure length <= 41, nonzero final, entries in [-64,64], exactly 12 nonzero
        if len(coeffs) > 41:
            return None
        if len(coeffs) == 0 or coeffs[-1] == 0:
            return None
        for c in coeffs:
            if not isinstance(c, int) or c < -64 or c > 64:
                return None
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return None
        return coeffs
    
    # Initial validation of seed
    seed = [1] * 12
    assert validate_and_trim(seed) is not None
    best_count = square_nonzero_count(seed)
    best_poly = seed[:]
    
    # Random search: generate random 12-term polynomials
    # Strategy: random degree, random positions for 12 nonzero terms, random coefficients
    max_degree = 40  # length 41
    
    # Try different coefficient distributions
    # 1) Coefficients from {-1, 1}
    # 2) Coefficients from small range
    # 3) Coefficients from wider range
    
    attempts = 0
    
    while time.time() < deadline:
        batch_start = time.time()
        
        # Generate a batch of candidates
        batch_size = 5000
        for _ in range(batch_size):
            if time.time() > deadline:
                break
            
            # Random degree between 12 and 40
            deg = random.randint(12, max_degree)
            n = deg + 1
            
            # Choose 12 distinct positions for nonzero terms
            positions = random.sample(range(n), 12)
            positions.sort()
            
            # Ensure highest degree term is nonzero (it will be since we pick 12 positions)
            # Assign coefficients
            # Mix of strategies:
            r = random.random()
            if r < 0.3:
                # Binary: +/- 1
                coeffs = [0] * n
                for p in positions:
                    coeffs[p] = random.choice([-1, 1])
            elif r < 0.6:
                # Small integers: -3 to 3
                coeffs = [0] * n
                for p in positions:
                    coeffs[p] = random.randint(-3, 3)
                    if coeffs[p] == 0:
                        coeffs[p] = 1
            elif r < 0.8:
                # Medium: -8 to 8
                coeffs = [0] * n
                for p in positions:
                    coeffs[p] = random.randint(-8, 8)
                    if coeffs[p] == 0:
                        coeffs[p] = 1
            else:
                # Wider: -64 to 64
                coeffs = [0] * n
                for p in positions:
                    coeffs[p] = random.randint(-64, 64)
                    if coeffs[p] == 0:
                        coeffs[p] = 1
            
            # Trim trailing zeros (shouldn't happen since we set exactly 12 nonzero)
            # But ensure final coefficient is nonzero
            if coeffs[-1] == 0:
                continue
            
            # Validate
            if validate_and_trim(coeffs) is None:
                continue
            
            # Compute square nonzero count
            cnt = square_nonzero_count(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = coeffs[:]
        
        attempts += batch_size
    
    return best_poly
# EVOLVE-BLOCK-END
