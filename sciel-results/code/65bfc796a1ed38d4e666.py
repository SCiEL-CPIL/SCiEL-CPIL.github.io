"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Map each number 2..40 to its prime exponent vector
    def get_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    candidates = list(range(2, 41))
    vectors = [get_vector(n) for n in candidates]
    
    # Heuristic: Start with the set of primes, which is known to be independent.
    # Size is 12. This is the theoretical maximum for dimension 12.
    best_set = [p for p in candidates if get_vector(p).count(1) == 1 and all(v == 0 or v == 1 for v in get_vector(p))]
    # Simpler: just the primes
    best_set = primes.copy()
    
    # If we are confident 12 is the max, we can just return best_set.
    # But to follow the directive, we run a B&B to "verify" or "find".
    # Since 12 is the max, B&B will find no improvement.
    
    # To make the B&B fast, we can just return the primes.
    # But I will implement the B&B structure as requested.
    
    def check_independent(vecs):
        # Check if vectors are linearly independent
        if not vecs: return True
        n = len(vecs)
        d = len(vecs[0])
        # Use Gaussian elimination
        # We work with a copy
        mat = [list(v) for v in vecs]
        rank = 0
        for col in range(d):
            # Find pivot
            pivot = -1
            for r in range(rank, n):
                if mat[r][col] != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            # Eliminate
            for r in range(n):
                if r != rank and mat[r][col] != 0:
                    # Factor
                    # To keep integers, we can't just divide.
                    # We can use the property that if rank < n, they are dependent.
                    # For small n, we can just check if the vector can be reduced to 0.
                    # Let's use a simpler check for the B&B:
                    # Add v to basis. If basis size increases, it's independent.
                    # We can maintain a basis in reduced row echelon form (RREF) over Q.
                    # But implementing RREF with integers is tricky.
                    
                    # Alternative: 
                    # Since dim is small (12), we can just check if the determinant of any 
                    # n x n submatrix is non-zero? No.
                    
                    # Let's stick to the integer elimination but be careful.
                    # Actually, for n=12, d=12, we can just check if the volume (det) is non-zero?
                    # No, for rectangular, it's rank.
                    
                    # Simplest robust check:
                    # Use sympy? Not allowed? "Standard library, NumPy, SciPy".
                    # NumPy has `linalg.matrix_rank`. 
                    # We can convert to float and check rank. 
                    # Risk of precision? Exponents are small. 12x12 matrix of small ints.
                    # Float precision is 53 bits. Determinants might be large?
                    # Max exponent is 5 (32=2^5). 
                    # Determinants of 12x12 matrices with entries < 10 are not huge.
                    # NumPy rank is safe here.
                    
                    pass
            rank += 1
            if rank == n:
                return True
        return rank == n

    # Using NumPy for rank check is much simpler and faster to code.
    import numpy as np
    
    def is_independent_numpy(vecs):
        if not vecs: return True
        mat = np.array(vecs, dtype=float)
        return np.linalg.matrix_rank(mat) == len(vecs)

    best_size = len(best_set)
    
    def solve(idx, current_vecs, current_indices):
        nonlocal best_set, best_size
        # Bound
        if len(current_vecs) + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            if len(current_vecs) > best_size:
                best_set = current_indices.copy()
                best_size = len(current_vecs)
            return
        
        # Exclude
        solve(idx + 1, current_vecs, current_indices)
        
        # Include
        if is_independent_numpy(current_vecs + [vectors[idx]]):
            current_indices.append(candidates[idx])
            solve(idx + 1, current_vecs + [vectors[idx]], current_indices)
            current_indices.pop()

    solve(0, [], [])
    
    return best_set

# EVOLVE-BLOCK-END
