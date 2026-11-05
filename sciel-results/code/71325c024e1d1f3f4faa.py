import random
import time
import copy

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    def count_nonzero_square(coeffs):
        """Given a list of (pos, coeff) pairs, count nonzero terms in the square."""
        square = {}
        n = len(coeffs)
        for i in range(n):
            pi, ci = coeffs[i]
            for j in range(i, n):
                pj, cj = coeffs[j]
                pos = pi + pj
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[pos] = square.get(pos, 0) + val
        # Count nonzero entries
        return sum(1 for v in square.values() if v != 0)
    
    def square_nonzero_count(positions, coefficients):
        """Count nonzero terms in square given positions and coefficients."""
        square = {}
        n = len(positions)
        for i in range(n):
            for j in range(i, n):
                pos = positions[i] + positions[j]
                if i == j:
                    val = coefficients[i] * coefficients[i]
                else:
                    val = 2 * coefficients[i] * coefficients[j]
                square[pos] = square.get(pos, 0) + val
        return sum(1 for v in square.values() if v != 0)
    
    best_count = 23  # baseline: 12 consecutive terms, all ones
    best_positions = list(range(12))
    best_coefficients = [1] * 12
    
    # Try some structured initial guesses
    initial_guesses = [
        (list(range(12)), [1] * 12),
        (list(range(0, 24, 2)), [1] * 12),
        ([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16], [1] * 12),
        ([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [1] * 12),
        (list(range(12)), [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]),
    ]
    
    for pos, coeff in initial_guesses:
        cnt = square_nonzero_count(pos, coeff)
        if cnt < best_count:
            best_count = cnt
            best_positions = pos[:]
            best_coefficients = coeff[:]
    
    # Local search
    random.seed(42)
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Current state
        positions = best_positions[:]
        coefficients = best_coefficients[:]
        current_count = best_count
        
        # Generate a mutation
        mutation_type = random.random()
        
        if mutation_type < 0.3:
            # Change a coefficient
            idx = random.randrange(12)
            old_coeff = coefficients[idx]
            # Try a different small coefficient
            new_coeff = random.choice([-3, -2, -1, 1, 2, 3])
            if new_coeff == old_coeff:
                continue
            coefficients[idx] = new_coeff
        elif mutation_type < 0.6:
            # Flip sign of a coefficient
            idx = random.randrange(12)
            coefficients[idx] = -coefficients[idx]
        elif mutation_type < 0.8:
            # Move a term to a new position
            idx = random.randrange(12)
            old_pos = positions[idx]
            # Choose a new position not already used
            used = set(positions)
            used.discard(old_pos)
            # Try positions in a reasonable range
            candidates = [p for p in range(0, 30) if p not in used]
            if not candidates:
                continue
            new_pos = random.choice(candidates)
            positions[idx] = new_pos
            # Sort positions
            order = sorted(range(12), key=lambda i: positions[i])
            positions = [positions[i] for i in order]
            coefficients = [coefficients[i] for i in order]
        else:
            # Swap two positions
            i, j = random.sample(range(12), 2)
            positions[i], positions[j] = positions[j], positions[i]
            # Re-sort
            order = sorted(range(12), key=lambda i: positions[i])
            positions = [positions[i] for i in order]
            coefficients = [coefficients[i] for i in order]
        
        new_count = square_nonzero_count(positions, coefficients)
        
        if new_count < current_count:
            best_count = new_count
            best_positions = positions[:]
            best_coefficients = coefficients[:]
            # Keep improving from here
            continue
        elif new_count == current_count and random.random() < 0.3:
            # Accept neutral moves sometimes to escape local optima
            best_positions = positions[:]
            best_coefficients = coefficients[:]
    
    # Build the final coefficient list
    max_pos = max(best_positions)
    result = [0] * (max_pos + 1)
    for p, c in zip(best_positions, best_coefficients):
        result[p] = c
    
    # Ensure length <= 41
    if len(result) > 41:
        # Truncate - shouldn't happen with our constraints
        result = result[:41]
    
    # Ensure nonzero final coefficient
    if result[-1] == 0:
        # Find last nonzero
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                result = result[:i+1]
                break
        if not result or result[-1] == 0:
            result = [1] * 12
    
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for x in result if x != 0)
    if nonzero_count != 12:
        # Fallback to simple construction
        result = [0] * 12
        for i in range(12):
            result[i] = 1
    
    return result
