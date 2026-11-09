"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START
import numpy as np

def _fft_square(c):
    """Use FFT for faster convolution when polynomial is long."""
    n = len(c)
    if n <= 20:
        c_arr = np.array(c, dtype=np.int64)
        result = np.convolve(c_arr, c_arr)
        nz = np.nonzero(result)[0]
        if len(nz) == 0:
            return [0]
        last = nz[-1]
        return result[:last+1].tolist()
    
    # Use FFT for longer polynomials
    size = 1
    while size < 2 * n - 1:
        size *= 2
    size *= 2  # extra padding for numerical stability
    
    c_arr = np.array(c, dtype=np.float64)
    padded = np.zeros(size, dtype=np.complex128)
    padded[:n] = c_arr
    fft_c = np.fft.fft(padded)
    fft_sq = fft_c * fft_c
    result = np.fft.ifft(fft_sq).real
    
    # Round to nearest integer
    result = np.round(result).astype(np.int64)
    result = result[:2*n-1]
    
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    return _fft_square(c)

def _nnz(c):
    return sum(1 for x in c if x != 0)

def _is_valid(poly):
    """Check if polynomial satisfies all constraints."""
    if len(poly) > 41:
        return False
    if len(poly) == 0 or poly[-1] == 0:
        return False
    if sum(1 for x in poly if x != 0) != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _local_search(poly, max_steps=500, time_limit=5.0):
    """Hill-climb: try flipping signs or changing individual coefficients."""
    start = time.time()
    current = poly[:]
    current_sq_nnz = _nnz(_square_coeffs(current))
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        improved = False
        # Try changing one coefficient at a time
        for i in range(len(current)):
            old_val = current[i]
            for delta in [-4, -3, -2, -1, 1, 2, 3, 4]:
                new_val = old_val + delta
                if new_val == 0 or abs(new_val) > 64:
                    continue
                candidate = current[:]
                candidate[i] = new_val
                if not _is_valid(candidate):
                    continue
                sq = _square_coeffs(candidate)
                nnz_sq = _nnz(sq)
                if nnz_sq < current_sq_nnz:
                    current = candidate
                    current_sq_nnz = nnz_sq
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try sign flips
            improved = False
            for i in range(len(current)):
                if current[i] != 0:
                    candidate = current[:]
                    candidate[i] = -candidate[i]
                    if not _is_valid(candidate):
                        continue
                    sq = _square_coeffs(candidate)
                    nnz_sq = _nnz(sq)
                    if nnz_sq < current_sq_nnz:
                        current = candidate
                        current_sq_nnz = nnz_sq
                        improved = True
                        break
            if not improved:
                break
    
    return current, current_sq_nnz

def _try_product_candidates(deadline, best_sq_nnz, best_poly):
    """Systematically try products of sparse polynomials."""
    # Try products of form (sum of k terms) * (sum of m terms) where k*m >= 12
    # and the product has exactly 12 nonzero terms
    
    for k in [2, 3, 4]:
        for m in range(k, 12//k + 2):
            if k * m < 12:
                continue
            # Try different degree ranges
            for d1 in range(k, 10):
                for d2 in range(m, 10):
                    if time.time() > deadline:
                        return best_sq_nnz, best_poly
                    
                    # Generate random sparse polynomials
                    for trial in range(5):
                        f1 = [0]*(d1+1)
                        pos1 = random.sample(range(d1+1), min(k, d1+1))
                        for p in pos1:
                            f1[p] = random.choice([-1, 1])
                        
                        f2 = [0]*(d2+1)
                        pos2 = random.sample(range(d2+1), min(m, d2+1))
                        for p in pos2:
                            f2[p] = random.choice([-1, 1])
                        
                        prod = np.convolve(f1, f2)
                        prod_list = prod.tolist()
                        while len(prod_list) > 1 and prod_list[-1] == 0:
                            prod_list.pop()
                        
                        if not _is_valid(prod_list):
                            continue
                        
                        sq = _square_coeffs(prod_list)
                        nnz_sq = _nnz(sq)
                        
                        if nnz_sq < best_sq_nnz:
                            best_sq_nnz = nnz_sq
                            best_poly = prod_list[:]
    
    return best_sq_nnz, best_poly

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    random.seed(42)
    
    # Phase 1: Evaluate structured candidates
    candidates = []
    
    # All ones
    candidates.append([1]*12)
    
    # Alternating signs palindromic
    candidates.append([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1])
    
    # Various palindromic sign patterns
    for signs in [(1,1,1,1,1,1), (1,1,1,-1,-1,-1), (1,-1,1,-1,1,-1), 
                  (1,1,-1,-1,1,1), (1,-1,-1,-1,-1,1), (-1,1,1,1,1,-1)]:
        pal = list(signs) + [signs[4], signs[3], signs[2], signs[1], signs[0]]
        if all(x != 0 for x in pal) and len(pal) == 12:
            candidates.append(pal)
    
    # Try P(x) = (x^6 + x^3 + 1) * (x^5 + x^4 + x + 1)
    f1 = [0]*7
    f1[0] = 1; f1[3] = 1; f1[6] = 1
    f2 = [0]*6
    f2[0] = 1; f2[1] = 1; f2[3] = 1; f2[5] = 1
    prod = np.convolve(f1, f2)
    prod_list = prod.tolist()
    while len(prod_list) > 1 and prod_list[-1] == 0:
        prod_list.pop()
    if _is_valid(prod_list):
        candidates.append(prod_list)
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() > deadline:
            break
        if not _is_valid(c):
            continue
        sq = _square_coeffs(c)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = c[:]
    
    # Phase 2: Product-based search
    if time.time() < deadline - 10:
        best_sq_nnz, best_poly = _try_product_candidates(deadline - 5, best_sq_nnz, best_poly)
    
    # Phase 3: Random search with local optimization
    non_zero_vals = list(range(-8, 9))
    non_zero_vals.remove(0)
    
    attempts = 0
    max_attempts = 100000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 10000 == 0 and time.time() > deadline:
            break
        
        # Choose structure randomly
        r = random.random()
        
        if r < 0.3:
            # Palindromic degree 11
            free = [random.choice(non_zero_vals) for _ in range(6)]
            poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        elif r < 0.5:
            # Anti-palindromic
            free = [random.choice(non_zero_vals) for _ in range(7)]
            poly = free[:5] + [free[5], free[6]] + [-free[4], -free[3], -free[2], -free[1], -free[0]]
        elif r < 0.7:
            # Random 12-term polynomial of random degree
            degree = random.randint(11, 20)
            poly = [0]*(degree+1)
            positions = random.sample(range(degree+1), 12)
            for p in positions:
                poly[p] = random.choice(non_zero_vals)
            if poly[-1] == 0:
                poly[-1] = random.choice(non_zero_vals)
        else:
            # Product of two sparse polynomials
            n1 = random.choice([2, 3, 4])
            n2 = 12 // n1
            if n2 < 2:
                n2 = 2
            f1_len = random.randint(n1, 10)
            f1 = [0]*f1_len
            pos1 = random.sample(range(f1_len), min(n1, f1_len))
            for p in pos1:
                f1[p] = random.choice([-1, 1])
            f2_len = random.randint(n2, 10)
            f2 = [0]*f2_len
            pos2 = random.sample(range(f2_len), min(n2, f2_len))
            for p in pos2:
                f2[p] = random.choice([-1, 1])
            prod = np.convolve(f1, f2)
            poly = prod.tolist()
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        
        if not _is_valid(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
            
            # Do local search from this candidate
            if time.time() < deadline - 10:
                improved_poly, improved_nnz = _local_search(poly, max_steps=300, time_limit=4.0)
                if improved_nnz < best_sq_nnz:
                    best_sq_nnz = improved_nnz
                    best_poly = improved_poly[:]
    
    # Phase 4: More intensive local search on best found
    if time.time() < deadline - 5:
        improved_poly, improved_nnz = _local_search(best_poly, max_steps=2000, time_limit=20.0)
        if improved_nnz < best_sq_nnz:
            best_sq_nnz = improved_nnz
            best_poly = improved_poly[:]
    
    # Final polish: try more aggressive local search
    if time.time() < deadline - 2:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            improved_poly, improved_nnz = _local_search(best_poly, max_steps=1000, time_limit=5.0)
            if improved_nnz < best_sq_nnz:
                best_sq_nnz = improved_nnz
                best_poly = improved_poly[:]
            else:
                break
    
    # Ensure output is valid
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    return best_poly
# EVOLVE-BLOCK-END
