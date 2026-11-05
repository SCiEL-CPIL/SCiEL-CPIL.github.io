import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    random.seed(42)
    
    def count_nonzero_square(coeffs):
        """Count nonzero terms in the exact square of the polynomial."""
        n = len(coeffs)
        # Square has terms from degree 0 to 2*(n-1)
        square = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                if i == j:
                    square[2 * i] += 2 * val
                else:
                    square[i + j] += 2 * val
        # Count nonzero entries
        count = 0
        for v in square:
            if v != 0:
                count += 1
        return count
    
    def is_valid(coeffs):
        """Check if coeffs form a valid candidate."""
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        # Nonzero final coefficient
        if coeffs[-1] == 0:
            return False
        # Entries in [-64, 64]
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        # Exactly 12 nonzero terms
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return False
        return True
    
    # Start with a random valid candidate
    def random_candidate():
        # Pick 12 distinct positions for nonzero coefficients
        positions = random.sample(range(41), 12)
        positions.sort()
        # Ensure last position is within range and coefficient nonzero
        coeffs = [0] * 41
        for pos in positions:
            coeffs[pos] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8])
        # Ensure final coefficient is nonzero
        if coeffs[-1] == 0:
            # Move one nonzero to the end
            nonzero_positions = [i for i in range(41) if coeffs[i] != 0]
            # Replace the last nonzero position with 40
            last_nz = nonzero_positions[-1]
            coeffs[last_nz] = 0
            coeffs[40] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8])
        # Trim trailing zeros to minimize length, but keep at least 1
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    # Generate initial candidate
    best_coeffs = None
    best_score = float('inf')
    
    # Try several random starts
    for _ in range(50):
        c = random_candidate()
        if is_valid(c):
            s = count_nonzero_square(c)
            if s < best_score:
                best_score = s
                best_coeffs = c[:]
    
    # Simulated annealing
    current_coeffs = best_coeffs[:]
    current_score = best_score
    temperature = 10.0
    cooling = 0.9995
    
    # Track best found
    global_best_coeffs = best_coeffs[:]
    global_best_score = best_score
    
    while time.time() < deadline:
        # Pick a random position to modify
        pos = random.randint(0, 40)
        old_val = current_coeffs[pos]
        
        # Choose new value
        if old_val == 0:
            new_val = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, 0])
        else:
            new_val = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, 0])
        
        if new_val == old_val:
            continue
        
        current_coeffs[pos] = new_val
        
        # Check validity
        if not is_valid(current_coeffs):
            current_coeffs[pos] = old_val
            continue
        
        new_score = count_nonzero_square(current_coeffs)
        
        # Accept or reject
        if new_score < current_score:
            current_score = new_score
            if new_score < global_best_score:
                global_best_score = new_score
                global_best_coeffs = current_coeffs[:]
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if random.random() < (1.0 / (1.0 + delta)) * temperature:
                current_score = new_score
            else:
                current_coeffs[pos] = old_val
        
        temperature *= cooling
        if temperature < 0.1:
            # Restart with current best
            current_coeffs = global_best_coeffs[:]
            current_score = global_best_score
            temperature = 5.0
    
    # Return the best found
    result = global_best_coeffs
    # Ensure it's valid
    if not is_valid(result):
        # Fallback to a known valid candidate
        result = [1] * 12
        while len(result) < 41:
            result.append(0)
        # Trim
        while len(result) > 1 and result[-1] == 0:
            result.pop()
    
    return result
