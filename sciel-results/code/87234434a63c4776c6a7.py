"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _fft_square(c):
    """
    Compute the square of a polynomial represented by coefficient list c.
    Uses FFT for efficiency. Returns list of coefficients (integers).
    """
    n = len(c)
    # Size of FFT must be at least 2*n - 1
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    
    # Pad with zeros
    c_pad = np.zeros(fft_size, dtype=np.complex128)
    c_pad[:n] = c
    
    # FFT, square, IFFT
    c_fft = np.fft.fft(c_pad)
    c_sq_fft = c_fft * c_fft
    c_sq = np.fft.ifft(c_sq_fft)
    
    # Round to nearest integer to handle floating point errors
    c_sq_int = np.round(c_sq.real).astype(np.int64)
    
    # Trim to required length
    result = c_sq_int[:2*n-1]
    
    # Trim trailing zeros
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    """Count non-zero elements."""
    return sum(1 for x in c if x != 0)

def _is_valid(poly):
    """Check if polynomial meets constraints."""
    if len(poly) == 0:
        return False
    if poly[-1] == 0:
        return False
    if len(poly) > 41:
        return False
    if sum(1 for x in poly if x != 0) != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _local_search(poly, time_limit, current_best):
    """
    Perform local search (hill climbing) starting from poly.
    Returns best poly found and its square nnz.
    """
    best_poly = poly[:]
    best_sq = _fft_square(best_poly)
    best_nnz = _nnz(best_sq)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        improved = False
        # Try perturbing each coefficient by -2, -1, 1, 2
        for i in range(len(best_poly)):
            if time.time() - start_time > time_limit:
                break
            original_val = best_poly[i]
            for delta in [-2, -1, 1, 2]:
                new_val = original_val + delta
                if new_val == original_val:
                    continue
                if abs(new_val) > 64:
                    continue
                
                new_poly = best_poly[:]
                new_poly[i] = new_val
                
                # Check validity
                if not _is_valid(new_poly):
                    continue
                
                new_sq = _fft_square(new_poly)
                new_nnz = _nnz(new_sq)
                
                if new_nnz < best_nnz:
                    best_nnz = new_nnz
                    best_poly = new_poly
                    best_sq = new_sq
                    improved = True
                    no_improve_count = 0
                    if best_nnz < current_best[0]:
                        current_best[0] = best_nnz
                        current_best[1] = best_poly[:]
                    break # Move to next coefficient if improved
        
        if not improved:
            no_improve_count += 1
            # If stuck, try a random jump to escape local minima
            if no_improve_count % 50 == 0:
                # Randomly change 2-3 coefficients
                for _ in range(random.randint(2, 3)):
                    i = random.randint(0, len(best_poly)-1)
                    best_poly[i] = random.randint(-64, 64)
                # Ensure validity after jump
                if not _is_valid(best_poly):
                    # Revert to last valid
                    best_poly = current_best[1][:]
                    best_nnz = current_best[0]
                    best_sq = _fft_square(best_poly)
                    continue
        
    return best_poly, best_nnz

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    best_sq_nnz = float('inf')
    best_poly = [1] * 12
    
    # 1. Generate initial candidates
    candidates = []
    
    # Palindromic degree 11 (12 coeffs)
    # c0..c5 free, c6..c11 = c5..c0
    # All 12 must be non-zero.
    # Try small integer values
    vals = [-2, -1, 1, 2]
    for v0 in vals:
        for v1 in vals:
            for v2 in vals:
                for v3 in vals:
                    for v4 in vals:
                        for v5 in vals:
                            poly = [v0, v1, v2, v3, v4, v5, v5, v4, v3, v2, v1, v0]
                            if _is_valid(poly):
                                candidates.append(poly)
    
    # Add some random candidates
    random.seed(42)
    for _ in range(1000):
        poly = [random.randint(-10, 10) for _ in range(12)]
        # Ensure non-zero
        for i in range(12):
            if poly[i] == 0:
                poly[i] = 1
        if _is_valid(poly):
            candidates.append(poly)
    
    # 2. Evaluate candidates
    for c in candidates:
        if time.time() > deadline:
            break
        sq = _fft_square(c)
        nnz = _nnz(sq)
        if nnz < best_sq_nnz:
            best_sq_nnz = nnz
            best_poly = c[:]
    
    # 3. Local search from best candidate
    current_best = [best_sq_nnz, best_poly]
    best_poly, best_sq_nnz = _local_search(best_poly, 30.0, current_best)
    
    # 4. If time remains, try local search from other good candidates
    if time.time() < deadline - 10.0:
        # Find top 5 candidates from initial eval
        scored_candidates = []
        for c in candidates[:500]: # Limit to first 500 for speed
            sq = _fft_square(c)
            nnz = _nnz(sq)
            scored_candidates.append((nnz, c))
        
        scored_candidates.sort(key=lambda x: x[0])
        
        for nnz, c in scored_candidates[:5]:
            if time.time() > deadline - 5.0:
                break
            if nnz >= best_sq_nnz:
                continue
            # Run local search with remaining time
            remaining = deadline - time.time() - 2.0
            if remaining > 5.0:
                local_best, local_nnz = _local_search(c, remaining, [best_sq_nnz, best_poly])
                if local_nnz < best_sq_nnz:
                    best_sq_nnz = local_nnz
                    best_poly = local_best
    
    # Final verification and cleanup
    if not _is_valid(best_poly):
        best_poly = [1] * 12
        best_sq_nnz = _nnz(_fft_square(best_poly))
        
    return best_poly

# EVOLVE-BLOCK-END
