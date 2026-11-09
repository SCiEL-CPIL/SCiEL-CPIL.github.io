import random
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Finds 30 integer coefficients from {-1, 0, 1} such that the signed sum
    sum(c[i-1]/i) is non-zero and its absolute value is minimized.
    
    Strategy:
    1. Generate a random valid starting configuration.
    2. Use hill climbing (local search) to flip coefficients and reduce |sum|.
    3. If the sum becomes 0, revert or find a valid neighbor.
    4. Run multiple restarts to escape local minima if time permits.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s limit
    
    n = 30
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(1, i) for i in range(1, n + 1)]
    
    best_score = None
    best_coeffs = None
    
    def get_score(coeffs):
        """Calculate the absolute value of the signed sum."""
        s = Fraction(0, 1)
        for c, r in zip(coeffs, reciprocals):
            if c != 0:
                s += c * r
        if s == 0:
            return None # Invalid
        return abs(s)
    
    def random_valid_config():
        """Generate a random configuration that sums to non-zero."""
        while True:
            # Randomly assign -1, 0, 1 with some bias towards non-zero to ensure non-zero sum initially
            coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            # Ensure at least one non-zero
            if all(c == 0 for c in coeffs):
                coeffs[0] = 1
            score = get_score(coeffs)
            if score is not None:
                return coeffs, score

    # Number of restarts
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        coeffs, current_score = random_valid_config()
        
        # Hill Climbing
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                # Try flipping coeffs[i] to each value in {-1, 0, 1} excluding current
                current_val = coeffs[i]
                candidates = [v for v in [-1, 0, 1] if v != current_val]
                
                best_candidate_val = None
                best_candidate_score = None
                
                for v in candidates:
                    # Simulate change
                    old_val = coeffs[i]
                    coeffs[i] = v
                    new_score = get_score(coeffs)
                    
                    if new_score is not None:
                        if best_candidate_score is None or new_score < best_candidate_score:
                            best_candidate_score = new_score
                            best_candidate_val = v
                            
                    coeffs[i] = old_val
                
                # Apply best candidate if it improves
                if best_candidate_val is not None and best_candidate_score < current_score:
                    coeffs[i] = best_candidate_val
                    current_score = best_candidate_score
                    improved = True
                    # Break early if we found a very small value? 
                    # For now, just continue hill climbing.
                    
        # Update global best
        if best_score is None or current_score < best_score:
            best_score = current_score
            best_coeffs = coeffs[:]
            
        # Early termination if we find a "very good" solution?
        # The minimum non-zero value for harmonic sums of length 30 is likely quite small.
        # 1/30! is extremely small. But we are limited to coefficients -1,0,1.
        # The set of possible sums is finite.
        # We can't easily know the global minimum, so we rely on time.
        
    if best_coeffs is None:
        # Fallback to trivial solution if something went wrong
        return [0] * 28 + [1, -1]
        
    return best_coeffs
# EVOLVE-BLOCK-END
