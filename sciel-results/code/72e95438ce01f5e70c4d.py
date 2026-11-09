"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START
import numpy as np

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    n = len(c)
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

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

def _local_search(poly, max_steps=200, time_limit=3.0):
    """Hill-climb: try flipping signs or changing individual coefficients."""
    start = time.time()
    current = poly[:]
    current_sq_nnz = _nnz(_square_coeffs(current))
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(len(current)):
            old_val = current[i]
            for delta in [-1, 1, -2, 2, -3, 3]:
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
            # Try sign flips on nonzero terms
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

def _enumerate_products(k, m, max_degree, deadline, best_sq_nnz, best_poly):
    """Systematically enumerate products of k-term and m-term ±1 polynomials."""
    # Generate all k-term ±1 polynomials up to max_degree
    def gen_sparse(n_terms, max_deg):
        """Generate all polynomials with exactly n_terms nonzero ±1 coeffs in [0, max_deg]."""
        positions_list = []
        for positions in itertools.combinations(range(max_deg + 1), n_terms):
            for signs in itertools.product([-1, 1], repeat=n_terms):
                poly = [0] * (max_deg + 1)
                for p, s in zip(positions, signs):
                    poly[p] = s
                positions_list.append(poly)
        return positions_list
    
    polys_k = gen_sparse(k, max_degree)
    polys_m = gen_sparse(m, max_degree)
    
    for pk in polys_k:
        if time.time() > deadline:
            return best_sq_nnz, best_poly
        for pm in polys_m:
            if time.time() > deadline:
                return best_sq_nnz, best_poly
            prod = np.convolve(np.array(pk), np.array(pm))
            prod_list = prod.tolist()
            # Trim trailing zeros
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
    deadline = start_time + 82.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1] * 12
    random.seed(42)
    
    # Phase 1: Evaluate structured candidates
    candidates = []
    candidates.append([1] * 12)
    candidates.append([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1])
    
    # Palindromic sign patterns
    for signs in [(1,1,1,1,1,1), (1,1,1,-1,-1,-1), (1,-1,1,-1,1,-1),
                  (1,1,-1,-1,1,1), (1,-1,-1,-1,-1,1), (-1,1,1,1,1,-1),
                  (1,-1,1,1,-1,-1), (1,1,1,1,-1,-1), (-1,-1,1,1,1,1)]:
        pal = list(signs) + [signs[4], signs[3], signs[2], signs[1], signs[0]]
        if all(x != 0 for x in pal) and len(pal) == 12:
            candidates.append(pal)
    
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
    
    # Phase 2: Systematic product enumeration (3-term × 4-term, 2-term × 6-term, etc.)
    # 3×4: 12 terms in product
    if time.time() < deadline - 15:
        best_sq_nnz, best_poly = _enumerate_products(3, 4, 8, deadline - 10, best_sq_nnz, best_poly)
    
    # 2×6: 12 terms
    if time.time() < deadline - 10:
        best_sq_nnz, best_poly = _enumerate_products(2, 6, 8, deadline - 8, best_sq_nnz, best_poly)
    
    # 4×3: same as 3×4 but different structure
    if time.time() < deadline - 8:
        best_sq_nnz, best_poly = _enumerate_products(4, 3, 8, deadline - 6, best_sq_nnz, best_poly)
    
    # Phase 3: Random search with product structure
    non_zero_vals = list(range(-6, 7))
    non_zero_vals.remove(0)
    
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        r = random.random()
        
        if r < 0.4:
            # Product of 3-term and 4-term ±1 polynomials
            d1 = random.randint(3, 10)
            d2 = random.randint(4, 10)
            f1 = [0] * (d1 + 1)
            pos1 = random.sample(range(d1 + 1), 3)
            for p in pos1:
                f1[p] = random.choice([-1, 1])
            f2 = [0] * (d2 + 1)
            pos2 = random.sample(range(d2 + 1), 4)
            for p in pos2:
                f2[p] = random.choice([-1, 1])
            prod = np.convolve(f1, f2)
            poly = prod.tolist()
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        elif r < 0.6:
            # Product of 2-term and 6-term ±1 polynomials
            d1 = random.randint(2, 10)
            d2 = random.randint(6, 10)
            f1 = [0] * (d1 + 1)
            pos1 = random.sample(range(d1 + 1), 2)
            for p in pos1:
                f1[p] = random.choice([-1, 1])
            f2 = [0] * (d2 + 1)
            pos2 = random.sample(range(d2 + 1), 6)
            for p in pos2:
                f2[p] = random.choice([-1, 1])
            prod = np.convolve(f1, f2)
            poly = prod.tolist()
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        elif r < 0.8:
            # Palindromic degree 11 with random small coefficients
            free = [random.choice(non_zero_vals) for _ in range(6)]
            poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        else:
            # Random 12-term polynomial
            degree = random.randint(11, 18)
            poly = [0] * (degree + 1)
            positions = random.sample(range(degree + 1), 12)
            for p in positions:
                poly[p] = random.choice(non_zero_vals)
            if poly[-1] == 0:
                poly[-1] = random.choice(non_zero_vals)
        
        if not _is_valid(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
            # Local search from this candidate
            if time.time() < deadline - 8:
                improved_poly, improved_nnz = _local_search(poly, max_steps=100, time_limit=2.0)
                if improved_nnz < best_sq_nnz:
                    best_sq_nnz = improved_nnz
                    best_poly = improved_poly[:]
    
    # Phase 4: Intensive local search on best found
    if time.time() < deadline - 3:
        improved_poly, improved_nnz = _local_search(best_poly, max_steps=500, time_limit=10.0)
        if improved_nnz < best_sq_nnz:
            best_sq_nnz = improved_nnz
            best_poly = improved_poly[:]
    
    # Final polish
    if time.time() < deadline - 1:
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            improved_poly, improved_nnz = _local_search(best_poly, max_steps=300, time_limit=3.0)
            if improved_nnz < best_sq_nnz:
                best_sq_nnz = improved_nnz
                best_poly = improved_poly[:]
            else:
                break
    
    # Ensure output is valid
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1] * 12
    
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1] * 12
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    return best_poly
# EVOLVE-BLOCK-END
