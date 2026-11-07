"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Function to get the exponent vector of a number with respect to the primes
    def get_vector(n):
        vec = []
        for p in primes:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            vec.append(count)
        return vec

    # Check if a list of vectors is linearly independent
    # We can do this by checking if the rank of the matrix formed by the vectors equals the number of vectors
    # Since the dimension is 12 and we want max size, we can just try to add vectors one by one
    # and check if the new vector is in the span of the previous ones.
    
    def is_independent(vecs):
        if not vecs:
            return True
        # Simple Gaussian elimination on the list of vectors
        # We treat the vectors as rows in a matrix
        mat = [list(v) for v in vecs]
        rank = 0
        for col in range(len(mat[0])):
            # Find a pivot
            pivot_row = -1
            for r in range(rank, len(mat)):
                if mat[r][col] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue
            # Swap rows
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            # Eliminate
            for r in range(rank + 1, len(mat)):
                factor = mat[r][col]
                if factor != 0:
                    for c in range(col, len(mat[0])):
                        mat[r][c] -= factor * mat[rank][c]
            rank += 1
        return rank == len(vecs)

    # Candidate numbers from 2 to 40
    candidates = list(range(2, 41))
    
    # Sort candidates. To maximize the chance of finding a large independent set,
    # it's often good to pick "simple" numbers first, but primes are the simplest.
    # Actually, any order works if we check independence.
    # Let's just iterate through 2..40.
    
    selected = []
    selected_vecs = []
    
    for num in candidates:
        v = get_vector(num)
        # Try adding this vector to the current set
        new_vecs = selected_vecs + [v]
        if is_independent(new_vecs):
            selected.append(num)
            selected_vecs.append(v)
            
    return selected
# EVOLVE-BLOCK-END
