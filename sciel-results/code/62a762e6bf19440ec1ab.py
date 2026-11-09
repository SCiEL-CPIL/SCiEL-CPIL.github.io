import time
import random
import numpy as np
from itertools import combinations

def count_nonzero(coeffs):
    """Count nonzero terms in a coefficient list."""
    if isinstance(coeffs, list):
        return sum(1 for c in coeffs if c != 0)
    return np.count_nonzero(coeffs)

def compute_square(coeffs):
    """Compute the exact square of a polynomial using numpy convolution."""
    arr = np.array(coeffs, dtype=np.int64)
    return np.convolve(arr, arr)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline (5s safety margin)
    
    # Initial candidate: P(x) = 1 + x + x^2 + ... + x^11
    # Square has 23 nonzero terms
    best_coeffs = [1] * 12
    best_square = compute_square(best_coeffs)
    best_nonzero = count_nonzero(best_square)
    
    random.seed(42)
    
    # Borwein-Erdelyi style: P = A + B, search for sparse A, B with small coeffs
    # such that 2AB cancels many terms in A^2 + B^2
    # Use coefficients from {-2, -1, 1, 2}
    small_coeffs = [-2, -1, 1, 2]
    
    # Strategy 1: Structured search using the decomposition P = A + B
    # where A and B are sparse polynomials
    # We'll try various splits of the 12 terms between A and B
    
    # Precompute some known good structures
    # Try P(x) = x^a + x^b + ... with specific patterns that lead to cancellation
    
    # Key insight: if P(x) = Q(x)^2 / something, but we need integer coeffs.
    # Better: search for P where P^2 has minimal support.
    
    # Try a more targeted search: 
    # For each degree L, try all combinations of 12 positions from {-2,-1,1,2}
    # but that's too large. Instead, use local search.
    
    # Phase 1: Try structured patterns
    # Pattern: symmetric polynomials often have sparse squares
    
    # Try P(x) = x^0 + x^1 + x^3 + x^4 + x^6 + x^7 + x^9 + x^10 + x^12 + x^13 + x^15 + x^16
    # This has gaps that might cause cancellations
    
    # More systematic: use the fact that if P has terms at positions that form
    # a "difference set" structure, cancellations occur.
    
    # Local search approach: start from random, then flip individual coefficients
    # or swap positions to minimize nonzero count in square
    
    def evaluate(coeffs):
        square = compute_square(coeffs)
        return count_nonzero(square), square
    
    # Try several structured initial candidates
    structured_candidates = []
    
    # Candidate 1: All ones (baseline)
    structured_candidates.append([1]*12)
    
    # Candidate 2: Alternating signs
    structured_candidates.append([1 if i%2==0 else -1 for i in range(12)])
    
    # Candidate 3: Grouped
    structured_candidates.append([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    
    # Candidate 4: Sparse with gaps
    for gap in [2, 3, 4, 5]:
        positions = list(range(0, 12*gap, gap))[:12]
        if len(positions) == 12:
            L = positions[-1] + 1
            coeffs = [0]*L
            for p in positions:
                coeffs[p] = 1
            structured_candidates.append(coeffs)
    
    # Evaluate all structured candidates
    for cand in structured_candidates:
        nz, sq = evaluate(cand)
        if nz < best_nonzero:
            best_nonzero = nz
            best_coeffs = cand[:]
            best_square = sq
    
    # Phase 2: Local search from best candidate
    # Use simulated annealing style: try mutations and accept if better or with prob
    
    current_coeffs = best_coeffs[:]
    current_nz = best_nonzero
    current_square = best_square
    
    # Small coefficient values for mutation
    mutation_values = [-2, -1, 1, 2, -3, -4, 3, 4]
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Choose mutation type
        mut_type = random.random()
        
        if mut_type < 0.4:
            # Change value of one nonzero coefficient
            L = len(current_coeffs)
            nonzero_idx = [i for i in range(L) if current_coeffs[i] != 0]
            if not nonzero_idx:
                continue
            idx = random.choice(nonzero_idx)
            old_val = current_coeffs[idx]
            new_val = random.choice([v for v in mutation_values if v != old_val and -64 <= v <= 64])
            current_coeffs[idx] = new_val
            nz, sq = evaluate(current_coeffs)
            if nz < current_nz:
                current_nz = nz
                current_square = sq
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_coeffs = current_coeffs[:]
                    best_square = sq
            else:
                # Revert with high probability (greedy with slight exploration)
                if random.random() > 0.05:
                    current_coeffs[idx] = old_val
        elif mut_type < 0.7:
            # Swap two nonzero positions
            L = len(current_coeffs)
            nonzero_idx = [i for i in range(L) if current_coeffs[i] != 0]
            if len(nonzero_idx) < 2:
                continue
            i, j = random.sample(nonzero_idx, 2)
            current_coeffs[i], current_coeffs[j] = current_coeffs[j], current_coeffs[i]
            nz, sq = evaluate(current_coeffs)
            if nz < current_nz:
                current_nz = nz
                current_square = sq
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_coeffs = current_coeffs[:]
                    best_square = sq
            else:
                if random.random() > 0.05:
                    current_coeffs[i], current_coeffs[j] = current_coeffs[j], current_coeffs[i]
        elif mut_type < 0.85:
            # Move a nonzero term to a new position
            L = len(current_coeffs)
            nonzero_idx = [i for i in range(L) if current_coeffs[i] != 0]
            if not nonzero_idx:
                continue
            idx = random.choice(nonzero_idx)
            val = current_coeffs[idx]
            # Choose new position (must keep exactly 12 nonzeros, final coeff nonzero)
            # Try extending or shifting
            new_L = random.choice([L, L+1, L+2]) if L < 41 else L
            if new_L > L:
                current_coeffs.extend([0]*(new_L - L))
            # Pick a zero position that is not the last (or is last if we want to move there)
            zero_positions = [i for i in range(new_L) if current_coeffs[i] == 0]
            if not zero_positions:
                current_coeffs[idx] = val
                continue
            # Ensure final coefficient is nonzero: if moving from last, new pos must not be last or we set another
            if idx == L - 1 and idx == new_L - 1:
                # Don't move the last term to a different last position easily
                pass
            new_idx = random.choice(zero_positions)
            # If new_idx is the last position, that's fine. If we're moving FROM last, 
            # we need another nonzero at last
            current_coeffs[idx] = 0
            current_coeffs[new_idx] = val
            # Check: exactly 12 nonzeros?
            nz_count = sum(1 for c in current_coeffs if c != 0)
            if nz_count != 12:
                current_coeffs[new_idx] = 0
                current_coeffs[idx] = val
                continue
            # Check: final coefficient nonzero
            if current_coeffs[-1] == 0:
                current_coeffs[new_idx] = 0
                current_coeffs[idx] = val
                continue
            nz, sq = evaluate(current_coeffs)
            if nz < current_nz:
                current_nz = nz
                current_square = sq
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_coeffs = current_coeffs[:]
                    best_square = sq
            else:
                if random.random() > 0.05:
                    current_coeffs[new_idx] = 0
                    current_coeffs[idx] = val
        else:
            # Random restart: new random polynomial
            L = random.randint(12, 41)
            last_idx = L - 1
            other_indices = random.sample(range(L - 1), 11)
            non_zero_indices = set(other_indices) | {last_idx}
            coeffs = [0] * L
            for idx in non_zero_indices:
                coeffs[idx] = random.choice(mutation_values)
            nz, sq = evaluate(coeffs)
            if nz < current_nz:
                current_nz = nz
                current_square = sq
                current_coeffs = coeffs[:]
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_coeffs = coeffs[:]
                    best_square = sq
    
    return best_coeffs
