"""Structural change: direct cancellation-targeting construction + refined local search."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    DEADLINE = start + 85.0  # safety margin before 90s kill

    def coeffs_from_positions(positions, magnitudes, signs):
        """positions: sorted list of 12 ints, magnitudes/signs: lists of 12."""
        max_deg = max(positions)
        coeffs = [0] * (max_deg + 1)
        for i in range(12):
            coeffs[positions[i]] = signs[i] * magnitudes[i]
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def is_valid(coeffs):
        if not (1 <= len(coeffs) <= 41):
            return False
        if coeffs[-1] == 0:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def square_nonzero_count(coeffs):
        sq = np.convolve(np.array(coeffs, dtype=np.int64), np.array(coeffs, dtype=np.int64))
        return int(np.count_nonzero(sq))

    def square_nonzero_count_from_state(positions, magnitudes, signs):
        coeffs = coeffs_from_positions(positions, magnitudes, signs)
        return square_nonzero_count(coeffs)

    # ---- Phase 1: Systematic pair-cancellation construction ----
    # Strategy: Build P with terms arranged so cross-terms cancel.
    # If P = sum a_i x^{p_i}, then P^2 has term x^k with coeff sum_{i+j=k} a_i a_j.
    # We want many of these sums to be zero.
    
    def try_construction(positions, magnitudes, signs):
        """Return nonzero count of square, or inf if invalid."""
        coeffs = coeffs_from_positions(positions, magnitudes, signs)
        if not is_valid(coeffs):
            return float('inf')
        return square_nonzero_count(coeffs)

    best_score = float('inf')
    best_coeffs = None

    # Method A: Consecutive positions with carefully chosen signs/magnitudes
    # Try all sign patterns on small consecutive blocks
    for base in range(0, 30):
        if time.time() > DEADLINE - 5:
            break
        positions = list(range(base, base + 12))
        # Try symmetric sign patterns
        for pattern_name in ['sym', 'anti', 'quarter']:
            if pattern_name == 'sym':
                signs = [1 if i < 6 else 1 for i in range(12)]
                mags = [1] * 12
            elif pattern_name == 'anti':
                signs = [1 if i < 6 else -1 for i in range(12)]
                mags = [1] * 12
            else:
                signs = [1 if i < 3 else (-1 if i < 6 else (1 if i < 9 else -1)) for i in range(12)]
                mags = [1] * 12
            
            sc = try_construction(positions, mags, signs)
            if sc < best_score:
                best_score = sc
                best_coeffs = coeffs_from_positions(positions, mags, signs)

    # Method B: Sparse positions with cancellation
    # Use positions that create overlap in cross-terms
    random.seed(42)
    for trial in range(2000):
        if time.time() > DEADLINE - 10:
            break
        # Choose 12 positions with some clustering to create cancellations
        strategy = random.randint(1, 4)
        if strategy == 1:
            # Two clusters of 6
            c1 = random.randint(0, 15)
            c2 = random.randint(20, 35)
            positions = sorted(list(range(c1, c1+6)) + list(range(c2, c2+6)))
        elif strategy == 2:
            # Arithmetic progression
            a = random.randint(0, 30)
            d = random.choice([1, 2, 3, 4, 5])
            positions = [a + i*d for i in range(12)]
            if positions[-1] >= 41:
                continue
        elif strategy == 3:
            # Geometric-ish
            positions = sorted(random.sample(range(41), 12))
        else:
            # Mixed
            positions = sorted([random.randint(0, 40) for _ in range(12)])
        
        if len(set(positions)) != 12 or positions[-1] >= 41:
            continue
        
        # Try various sign/magnitude patterns
        for _ in range(5):
            signs = [random.choice([-1, 1]) for _ in range(12)]
            mags = [random.choice([1, 1, 2, 2, 3]) for _ in range(12)]
            
            sc = try_construction(positions, mags, signs)
            if sc < best_score:
                best_score = sc
                best_coeffs = coeffs_from_positions(positions, mags, signs)

    # Method C: Direct coefficient assignment targeting specific cancellations
    # Start from a base pattern and greedily adjust to zero out square coefficients
    for trial in range(1000):
        if time.time() > DEADLINE - 10:
            break
        
        # Random initial state
        positions = sorted(random.sample(range(41), 12))
        signs = [random.choice([-1, 1]) for _ in range(12)]
        mags = [random.choice([1, 1, 1, 2]) for _ in range(12)]
        
        # Greedy local search: try to reduce nonzero count
        cur_score = try_construction(positions, mags, signs)
        if cur_score == float('inf'):
            continue
            
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            # Try flipping each sign
            for i in range(12):
                new_signs = signs[:]
                new_signs[i] = -new_signs[i]
                sc = try_construction(positions, mags, new_signs)
                if sc < cur_score:
                    cur_score = sc
                    signs = new_signs
                    improved = True
                    break
            if improved:
                continue
            # Try changing each magnitude
            for i in range(12):
                for new_mag in [1, 2, 3, 4, 5]:
                    if new_mag == mags[i]:
                        continue
                    new_mags = mags[:]
                    new_mags[i] = new_mag
                    sc = try_construction(positions, new_mags, signs)
                    if sc < cur_score:
                        cur_score = sc
                        mags = new_mags
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try moving one position
            for i in range(12):
                for new_pos in random.sample(range(41), min(10, 41)):
                    if new_pos in positions:
                        continue
                    new_positions = positions[:]
                    new_positions[i] = new_pos
                    new_positions = sorted(new_positions)
                    sc = try_construction(new_positions, mags, signs)
                    if sc < cur_score:
                        cur_score = sc
                        positions = new_positions
                        improved = True
                        break
                if improved:
                    break
        
        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = coeffs_from_positions(positions, mags, signs)

    # Method D: Simulated annealing from best found
    if best_coeffs is not None:
        # Extract state from best_coeffs
        nonzero_indices = [i for i, c in enumerate(best_coeffs) if c != 0]
        positions = nonzero_indices
        signs = [1 if best_coeffs[i] > 0 else -1 for i in nonzero_indices]
        mags = [abs(best_coeffs[i]) for i in nonzero_indices]
    else:
        positions = list(range(12))
        signs = [1] * 12
        mags = [1] * 12

    cur_score = try_construction(positions, mags, signs)
    if cur_score == float('inf'):
        cur_score = best_score

    # SA refinement
    T0 = 5.0
    T_end = 0.01
    total_steps = 50000

    for step in range(total_steps):
        if time.time() > DEADLINE:
            break
        progress = step / total_steps
        T = T0 * (T_end / T0) ** progress

        move = random.randint(1, 5)
        new_positions = positions[:]
        new_mags = mags[:]
        new_signs = signs[:]

        if move == 1:  # move one position
            i = random.randrange(12)
            new_pos = random.randrange(41)
            if new_pos not in new_positions:
                new_positions[i] = new_pos
                new_positions = sorted(new_positions)
        elif move == 2:  # flip one sign
            i = random.randrange(12)
            new_signs[i] = -new_signs[i]
        elif move == 3:  # change one magnitude
            i = random.randrange(12)
            new_mags[i] = random.choice([1, 1, 2, 2, 3, 4, 5, 8, 16])
        elif move == 4:  # move two positions
            i, j = random.sample(range(12), 2)
            p1, p2 = random.sample(range(41), 2)
            if p1 not in new_positions and p2 not in new_positions:
                new_positions[i] = p1
                new_positions[j] = p2
                new_positions = sorted(new_positions)
        else:  # swap two positions
            i, j = random.sample(range(12), 2)
            new_positions[i], new_positions[j] = new_positions[j], new_positions[i]
            new_positions = sorted(new_positions)

        if len(set(new_positions)) != 12 or new_positions[-1] >= 41:
            continue

        new_score = try_construction(new_positions, new_mags, new_signs)
        if new_score == float('inf'):
            continue

        delta = new_score - cur_score
        if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-9)):
            positions = new_positions
            mags = new_mags
            signs = new_signs
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_coeffs = coeffs_from_positions(positions, mags, signs)

    # Finalize
    if best_coeffs is not None and is_valid(best_coeffs):
        return best_coeffs
    # Fallback
    return [1] * 12
