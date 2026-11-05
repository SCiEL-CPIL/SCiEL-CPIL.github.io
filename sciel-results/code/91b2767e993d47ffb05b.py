import numpy as np
import random
import time

def _square_nonzero_count(coeffs):
    """Compute number of nonzero terms in Q^2 given dense coefficient list of Q."""
    if len(coeffs) == 0:
        return 0
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def construct():
    best_count = float('inf')
    best_coeffs = None
    
    def try_candidate(coeffs):
        nonlocal best_count, best_coeffs
        # Check constraints
        if len(coeffs) != 12:
            return
        if any(c == 0 for c in coeffs):
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        count = _square_nonzero_count(coeffs)
        if count < best_count:
            best_count = count
            best_coeffs = list(coeffs)

    random.seed(42)
    
    # Local search that allows changing exponents and coefficients
    # State: list of (exponent, coeff) pairs, sorted by exponent
    # But for fast convolution, we use dense array.
    # To search exponents, we need to move terms.
    
    def local_search_dense(c, max_iter=200):
        # c is a list of 12 coeffs, but we need to know their positions?
        # The current program assumes fixed positions 0..11.
        # The task says "length<=41", "ascending degree".
        # The seed program returns a list of 12 coeffs, implying positions 0..11?
        # "Return coefficient list in ascending degree... exactly 12 nonzero terms."
        # This usually means the list *is* the coefficients for degrees 0..11, or the list represents the sparse terms?
        # The seed program returns `result` which is a list of 12 ints.
        # If I return `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]`, it has 2 nonzero terms, not 12.
        # The seed program enforces `if any(c == 0 for c in coeffs): return`.
        # This implies the returned list must have **no zeros**.
        # Therefore, the "12 nonzero terms" are at degrees 0, 1, ..., 11.
        # The degree of the polynomial is 11.
        # Wait, "length<=41". If I return a list of length 12, it's valid.
        # If I wanted to put terms at 0 and 40, I would return a list of length 41 with zeros in between?
        # But `try_candidate` rejects zeros.
        # So the seed program is restricted to polynomials of degree 11 with all coefficients nonzero.
        # Is it possible to get a better score by using a longer list with zeros?
        # The contract says: "Return coefficient list in ascending degree, length<=41, nonzero final coefficient, entries integers in [-64,64], exactly 12 nonzero terms."
        # It does NOT say "no zeros in the list". It says "exactly 12 nonzero terms".
        # The seed program's `try_candidate` is too strict! It rejects any list with a zero.
        # This restricts the search to degree 11 polynomials with full support.
        # We can improve by allowing zeros in the list, as long as there are exactly 12 nonzeros.
        # This allows us to space out the terms to create cancellations or avoid overlaps.
        
        pass

    # Revised Strategy:
    # 1. Allow the polynomial to have degree up to 40.
    # 2. Represent the polynomial as a dense list of length L (12 <= L <= 41).
    # 3. Ensure exactly 12 nonzeros.
    # 4. Search over positions and values.
    
    def evaluate_poly(dense_coeffs):
        # dense_coeffs is a list of ints
        non_zero_count = sum(1 for c in dense_coeffs if c != 0)
        if non_zero_count != 12:
            return float('inf')
        if any(abs(c) > 64 for c in dense_coeffs):
            return float('inf')
        if dense_coeffs[-1] == 0:
            return float('inf')
        return _square_nonzero_count(dense_coeffs)

    def local_search_sparse(positions, coeffs, max_iter=500):
        # positions: sorted list of 12 exponents
        # coeffs: list of 12 values corresponding to positions
        # Current best count for this state
        current_count = float('inf')
        
        def get_dense():
            L = max(positions) + 1
            d = [0] * L
            for p, c in zip(positions, coeffs):
                d[p] = c
            return d
        
        # Initial eval
        d = get_dense()
        current_count = evaluate_poly(d)
        
        for _ in range(max_iter):
            improved = False
            
            # 1. Try changing a coefficient value
            for i in range(12):
                old_c = coeffs[i]
                # Try small changes and random jumps
                candidates = [old_c - 1, old_c + 1, -old_c, 1, -1, 2, -2]
                candidates += [random.randint(-10, 10) for _ in range(3)]
                candidates = list(set(candidates))
                
                for new_c in candidates:
                    if new_c == 0 or new_c == old_c:
                        continue
                    if abs(new_c) > 64:
                        continue
                    new_coeffs = list(coeffs)
                    new_coeffs[i] = new_c
                    d = get_dense()
                    # Update d manually for speed? No, just rebuild for simplicity if L is small
                    # Actually, rebuilding is O(L). L <= 41. Convolution is O(L^2).
                    # Let's just rebuild.
                    d_new = [0] * (max(positions) + 1)
                    for p, c in zip(positions, new_coeffs):
                        d_new[p] = c
                    new_count = evaluate_poly(d_new)
                    if new_count < current_count:
                        coeffs = new_coeffs
                        current_count = new_count
                        improved = True
                        break
                if improved:
                    break
            
            # 2. Try moving a position
            if not improved:
                i = random.randint(0, 11)
                old_pos = positions[i]
                # Try moving to a nearby empty slot or swapping
                # Find all empty slots in [0, 40]
                used = set(positions)
                # Candidate new positions
                new_pos_candidates = []
                # Nearby
                for delta in [-2, -1, 1, 2, 3]:
                    np_ = old_pos + delta
                    if 0 <= np_ <= 40 and np_ not in used:
                        new_pos_candidates.append(np_)
                # Random
                for _ in range(3):
                    np_ = random.randint(0, 40)
                    if np_ not in used:
                        new_pos_candidates.append(np_)
                
                for np_ in new_pos_candidates:
                    new_positions = list(positions)
                    new_positions[i] = np_
                    new_positions.sort()
                    # Check if valid (unique)
                    if len(set(new_positions)) != 12:
                        continue
                    d_new = [0] * (max(new_positions) + 1)
                    for p, c in zip(new_positions, coeffs):
                        d_new[p] = c
                    new_count = evaluate_poly(d_new)
                    if new_count < current_count:
                        positions = new_positions
                        current_count = new_count
                        improved = True
                        break
                if improved:
                    continue
            
            # 3. Try swapping two positions
            if not improved:
                i, j = random.sample(range(12), 2)
                if positions[i] != positions[j]:
                    positions[i], positions[j] = positions[j], positions[i]
                    positions.sort()
                    d_new = [0] * (max(positions) + 1)
                    for p, c in zip(positions, coeffs):
                        d_new[p] = c
                    new_count = evaluate_poly(d_new)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                    else:
                        # Revert
                        positions[i], positions[j] = positions[j], positions[i]
                        positions.sort()
            
            if not improved:
                # Stagnation, maybe break or do a random jump
                if random.random() < 0.1:
                    # Random jump: change 3 random coeffs
                    for k in range(3):
                        idx = random.randint(0, 11)
                        coeffs[idx] = random.randint(-5, 5)
                        if coeffs[idx] == 0: coeffs[idx] = 1
                else:
                    break
        
        # Finalize
        d_final = [0] * (max(positions) + 1)
        for p, c in zip(positions, coeffs):
            d_final[p] = c
        return d_final, current_count

    # Run multiple restarts
    for restart in range(200):
        # Initialize positions
        # Strategy: Start with a random set of 12 positions in [0, 40]
        # Or start with an arithmetic progression
        if restart % 3 == 0:
            # AP
            step = random.randint(1, 3)
            start = random.randint(0, 40 - 11*step)
            positions = [start + i*step for i in range(12)]
        else:
            # Random distinct positions
            positions = random.sample(range(0, 41), 12)
            positions.sort()
        
        # Initialize coeffs
        # Bias towards small integers
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        d_best, count_best = local_search_sparse(positions, coeffs, max_iter=100)
        
        if count_best < best_count:
            best_count = count_best
            best_coeffs = d_best

    # If we found a good one, return it
    if best_coeffs is not None:
        return best_coeffs

    # Fallback: Return a simple valid candidate
    # 12 nonzeros, degree <= 40
    # Simple: 1, -1, 1, -1... at positions 0..11
    return [1 if i%2==0 else -1 for i in range(12)]
