"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import combinations

def _get_prime_exponents(n):
    """Returns a dict of prime exponents for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _is_independent(vec, basis):
    """
    Checks if vec is linearly independent of the basis list.
    vec is a dict {prime: exponent}.
    basis is a list of dicts.
    """
    # We perform Gaussian elimination on the augmented matrix [basis | vec]
    # To avoid creating large matrices, we can just check if vec can be formed by basis.
    # Since dimensions are small, we can convert to a standard list of floats/Fractions 
    # aligned with all primes up to 40.
    
    # All primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Convert to vector of Fractions
    def to_vec(d):
        return [Fraction(d.get(p, 0), 1) for p in primes]
    
    v = to_vec(vec)
    b = [to_vec(bi) for bi in basis]
    
    # Gaussian elimination to see if v is in span(b)
    # We create a matrix of basis vectors and try to reduce v.
    # Actually, simpler: add v to basis, check rank.
    # Or, simpler: use a standard rank check.
    
    # Let's just do a simple check:
    # If basis is empty, independent.
    if not basis:
        return True
        
    # Create a list of vectors including v
    # We want to know if v is a linear combination of basis.
    # We can solve B * x = v.
    # Since n is small, we can just check if the determinant of the matrix formed by 
    # basis vectors and v is non-zero? No, that's for square matrices.
    
    # Better: Perform row reduction on the matrix formed by basis vectors.
    # Then check if v reduces to zero.
    
    # Copy basis to avoid mutation
    work_basis = [row[:] for row in b]
    
    # Perform Gaussian elimination on work_basis
    # We want to bring it to row echelon form
    rank = 0
    for col in range(len(primes)):
        # Find pivot
        pivot = None
        for row_idx in range(rank, len(work_basis)):
            if work_basis[row_idx][col] != 0:
                pivot = row_idx
                break
        if pivot is None:
            continue
            
        # Swap rows
        work_basis[rank], work_basis[pivot] = work_basis[pivot], work_basis[rank]
        
        # Normalize pivot row
        pivot_val = work_basis[rank][col]
        for k in range(col, len(primes)):
            work_basis[rank][k] /= pivot_val
            
        # Eliminate other rows
        for row_idx in range(len(work_basis)):
            if row_idx == rank:
                continue
            factor = work_basis[row_idx][col]
            if factor != 0:
                for k in range(col, len(primes)):
                    work_basis[row_idx][k] -= factor * work_basis[rank][k]
                    
        rank += 1
        
    # Now reduce v using the reduced basis
    # The reduced basis has rank 'rank'.
    # We can just check if v is in the span.
    # Since we have the reduced rows, we can express v in terms of them?
    # Actually, the standard way is to augment the matrix with v and see if the rank increases.
    
    # Let's restart the logic slightly for clarity and correctness:
    # Check if rank(basis) == rank(basis + [v])
    
    def calc_rank(vectors):
        if not vectors:
            return 0
        # Make a copy
        mat = [row[:] for row in vectors]
        r = 0
        for col in range(len(primes)):
            pivot = None
            for row_idx in range(r, len(mat)):
                if mat[row_idx][col] != 0:
                    pivot = row_idx
                    break
            if pivot is None:
                continue
            mat[r], mat[pivot] = mat[pivot], mat[r]
            pivot_val = mat[r][col]
            for k in range(col, len(primes)):
                mat[r][k] /= pivot_val
            for row_idx in range(len(mat)):
                if row_idx == r:
                    continue
                factor = mat[row_idx][col]
                if factor != 0:
                    for k in range(col, len(primes)):
                        mat[row_idx][k] -= factor * mat[r][k]
            r += 1
        return r

    rank_before = calc_rank(b)
    rank_after = calc_rank(b + [v])
    return rank_before != rank_after

def construct():
    n_max = 40
    selected = []
    selected_vecs = []
    
    # We iterate through numbers. 
    # To get a "different" set than just primes, we could prioritize higher powers?
    # Or just stick to the greedy order 2..40.
    # The greedy order 2..40 will pick 2, 3, 4? 
    # 2: vec (1,0..). Rank 1. Select.
    # 3: vec (0,1..). Rank 2. Select.
    # 4: vec (2,0..). Dependent on 2. Skip.
    # 5: vec (0,0,1..). Rank 3. Select.
    # ...
    # This will pick all primes.
    
    # To ensure we return a valid list of ints
    for i in range(2, n_max + 1):
        vec = _get_prime_exponents(i)
        if _is_independent(vec, selected_vecs):
            selected.append(i)
            selected_vecs.append(vec)
            
    return selected
# EVOLVE-BLOCK-END
