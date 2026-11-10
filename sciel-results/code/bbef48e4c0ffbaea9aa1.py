"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _square(coeffs):
    """Compute exact square of polynomial given as coefficient list (ascending degree)."""
    n = len(coeffs)
    result = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0:
                continue
            result[i + j] += ci * cj
    # Trim trailing zeros but keep at least 1
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def _nonzero_count(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _is_valid_candidate(coeffs):
    """Check all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    if _nonzero_count(coeffs) != 12:
        return False
    return True

def _score_candidate(coeffs):
    """Return (nonzero_count_of_square, coeffs) - lower is better."""
    sq = _square(coeffs)
    return _nonzero_count(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_coeffs = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    # Pad/trim to exactly 12 nonzero terms - let me construct properly
    # Actually let me just start with a simple valid candidate
    best_coeffs = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    # That's 8 ones. Need exactly 12 nonzero. Let me fix.
    best_coeffs = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # Count: positions 0,6,12,18,24,30,36,40 = 8 terms. Need 12.
    # Let me just make a proper 12-term polynomial
    best_coeffs = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # Still 8. Let me be more careful.
    
    # Simple valid 12-term: put 1s at 12 spread-out positions
    positions = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
    best_coeffs = [0] * 34
    for p in positions:
        best_coeffs[p] = 1
    best_coeffs = best_coeffs  # length 34, 12 nonzero, last is 1
    
    best_score = _score_candidate(best_coeffs)
    
    random.seed(42)
    
    # Try structured families first
    def make_palindromic(n_terms=12, max_deg=40):
        """Create a palindromic polynomial with exactly n_terms nonzero terms."""
        # Choose positions symmetrically
        # For palindromic: coeff[i] = coeff[max_deg - i]
        # We need exactly 12 nonzero terms
        # If max_deg is even, middle term counts once
        max_deg = 40
        coeffs = [0] * (max_deg + 1)
        # Choose 6 pairs (or 5 pairs + 1 middle if max_deg even)
        if max_deg % 2 == 0:
            # 6 pairs gives 12 terms, no middle
            pairs = random.sample(range(0, max_deg // 2), 6)
            for p in pairs:
                v = random.choice([-1, 1, 2, -2, 3, -3])
                coeffs[p] = v
                coeffs[max_deg - p] = v
        else:
            # 5 pairs + 1 middle = 11, need 1 more... use 6 pairs
            pairs = random.sample(range(0, max_deg // 2 + 1), 6)
            for p in pairs:
                v = random.choice([-1, 1, 2, -2, 3, -3])
                coeffs[p] = v
                coeffs[max_deg - p] = v
        # Trim trailing zeros
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    def make_sparse_random():
        """Random 12-term polynomial."""
        max_deg = 40
        positions = random.sample(range(0, max_deg + 1), 12)
        positions.sort()
        coeffs = [0] * (max_deg + 1)
        for p in positions:
            coeffs[p] = random.choice([-1, 1, 2, -2, 3, -3, 4, -4])
        # Ensure last coefficient nonzero
        if coeffs[-1] == 0:
            coeffs[-1] = 1
        # Ensure exactly 12 nonzero
        nz = [i for i in range(len(coeffs)) if coeffs[i] != 0]
        if len(nz) != 12:
            # Adjust
            return make_sparse_random()
        return coeffs
    
    # Try a batch of structured candidates
    for _ in range(500):
        if time.time() > deadline:
            break
        c = make_palindromic()
        if _is_valid_candidate(c):
            s = _score_candidate(c)
            if s < best_score:
                best_score = s
                best_coeffs = c
    
    # Random search with local improvement
    for _ in range(2000):
        if time.time() > deadline:
            break
        c = make_sparse_random()
        if not _is_valid_candidate(c):
            continue
        s = _score_candidate(c)
        if s < best_score:
            best_score = s
            best_coeffs = c
        
        # Local search: try flipping signs or changing values
        for trial in range(50):
            if time.time() > deadline:
                break
            c2 = c[:]
            # Pick a random nonzero position and change its value
            nz_pos = [i for i in range(len(c2)) if c2[i] != 0]
            if not nz_pos:
                break
            idx = random.choice(nz_pos)
            old_val = c2[idx]
            new_val = random.choice([-1, 1, 2, -2, 3, -3, 4, -4, 5, -5])
            if new_val == old_val:
                continue
            c2[idx] = new_val
            if not _is_valid_candidate(c2):
                continue
            s2 = _score_candidate(c2)
            if s2 < s:
                s = s2
                c = c2
                if s < best_score:
                    best_score = s
                    best_coeffs = c
            # Also try moving a term
            if random.random() < 0.3:
                c3 = c[:]
                nz_pos3 = [i for i in range(len(c3)) if c3[i] != 0]
                if len(nz_pos3) >= 2:
                    src = random.choice(nz_pos3)
                    # Move to a new position
                    empty_pos = [i for i in range(len(c3)) if c3[i] == 0]
                    if empty_pos:
                        dst = random.choice(empty_pos)
                        c3[dst] = c3[src]
                        c3[src] = 0
                        # Trim
                        while len(c3) > 1 and c3[-1] == 0:
                            c3.pop()
                        if _is_valid_candidate(c3):
                            s3 = _score_candidate(c3)
                            if s3 < s:
                                s = s3
                                c = c3
                                if s < best_score:
                                    best_score = s
                                    best_coeffs = c
    
    # Final: try targeted search for very sparse squares
    # A square with k nonzero terms means the convolution has many zeros
    # Try polynomials of form sum a_i x^{e_i} where e_i are in arithmetic progression
    for start in range(0, 5):
        for step in range(1, 8):
            if time.time() > deadline:
                break
            positions = [start + i * step for i in range(12)]
            if positions[-1] > 40:
                continue
            coeffs = [0] * (positions[-1] + 1)
            for p in positions:
                coeffs[p] = 1
            s = _score_candidate(coeffs)
            if s < best_score:
                best_score = s
                best_coeffs = coeffs
            
            # Try sign variations
            for signs_trial in range(100):
                if time.time() > deadline:
                    break
                coeffs2 = [0] * (positions[-1] + 1)
                for i, p in enumerate(positions):
                    coeffs2[p] = random.choice([-1, 1])
                if coeffs2[-1] == 0:
                    coeffs2[-1] = 1
                if _is_valid_candidate(coeffs2):
                    s2 = _score_candidate(coeffs2)
                    if s2 < best_score:
                        best_score = s2
                        best_coeffs = coeffs2
    
    return best_coeffs
# EVOLVE-BLOCK-END
