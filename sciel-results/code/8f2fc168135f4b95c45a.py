"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def _count_nonzero(coeffs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coeffs if c != 0)

def _get_sparsity(pairs):
    """Compute the number of nonzero terms in the square of the polynomial defined by pairs.
    pairs: list of (exponent, coefficient) tuples."""
    if not pairs:
        return 0
    max_deg = max(e for e, c in pairs if c != 0)
    arr = np.zeros(2 * max_deg + 1, dtype=int)
    for e, c in pairs:
        if c != 0:
            arr[e] += c
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def construct():
    best_sparsity = float('inf')
    best_result = None
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer for safety
    
    def random_start():
        exponents = random.sample(range(41), 12)
        exponents.sort()
        signs = [random.choice([-1, 1]) for _ in range(12)]
        pairs = list(zip(exponents, signs))
        return pairs

    def neighbor(pairs):
        """Generate a random neighbor of the given polynomial (list of tuples)."""
        new_pairs = pairs[:]
        move = random.choice([0, 1, 2, 3])
        
        if move == 0:
            # Flip sign of a random term
            i = random.randint(0, 11)
            e, c = new_pairs[i]
            new_pairs[i] = (e, -c)
            
        elif move == 1:
            # Change exponent of a random term to a random unused exponent
            i = random.randint(0, 11)
            e, c = new_pairs[i]
            used = set(p[0] for p in pairs)
            available = [x for x in range(41) if x not in used]
            if available:
                new_e = random.choice(available)
                new_pairs[i] = (new_e, c)
                new_pairs.sort(key=lambda x: x[0])
                
        elif move == 2:
            # Swap exponents of two random terms
            i, j = random.sample(range(12), 2)
            e1, c1 = new_pairs[i]
            e2, c2 = new_pairs[j]
            new_pairs[i] = (e2, c1)
            new_pairs[j] = (e1, c2)
            new_pairs.sort(key=lambda x: x[0])
            
        else:
            # Change coefficient value to a small random integer
            i = random.randint(0, 11)
            e, c = new_pairs[i]
            new_c = random.choice([-2, -1, 1, 2])
            new_pairs[i] = (e, new_c)
            new_pairs.sort(key=lambda x: x[0])

        return new_pairs

    # Run randomized local search with restarts
    for _ in range(100): # Max 100 restarts
        if time.time() - start_time > time_limit:
            break
            
        current = random_start()
        current_sparsity = _get_sparsity(current)
        
        if current_sparsity < best_sparsity:
            best_sparsity = current_sparsity
            best_result = current[:]
            
        # Local search
        for _ in range(1000):
            if time.time() - start_time > time_limit:
                break
            next_p = neighbor(current)
            next_sparsity = _get_sparsity(next_p)
            
            # Accept if better or equal
            if next_sparsity <= current_sparsity:
                current = next_p
                current_sparsity = next_sparsity
                if current_sparsity < best_sparsity:
                    best_sparsity = current_sparsity
                    best_result = current[:]
                    # If we reach a very low sparsity, we might want to stop early
                    # but let's keep searching to ensure global optimum within time
                    if best_sparsity <= 12:
                        # Theoretical minimum is 1 (perfect square), but for 12 terms it's likely higher
                        # 12 is a very good target
                        pass

    if best_result is None:
        # Fallback: simple polynomial
        exponents = list(range(12))
        signs = [1] * 12
        best_result = list(zip(exponents, signs))

    # Convert best_result (list of tuples) to coefficient list
    max_deg = max(e for e, c in best_result)
    result = [0] * (max_deg + 1)
    for e, c in best_result:
        result[e] = c
        
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
        
    # Ensure nonzero final coefficient
    # If the highest degree term was trimmed (which shouldn't happen as we sort and max_deg is from best_result),
    # or if it was 0 (which shouldn't happen), we need to fix it.
    if result[-1] == 0:
        # Find the last nonzero index
        last_nonzero = -1
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                last_nonzero = i
                break
        if last_nonzero != -1:
            result = result[:last_nonzero + 1]
        if result[-1] == 0:
            result[-1] = 1
            
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for c in result if c != 0)
    if nonzero_count != 12:
        # This is a safety check. If our logic failed, fallback.
        result = [1] * 12
        
    return result

# EVOLVE-BLOCK-END
