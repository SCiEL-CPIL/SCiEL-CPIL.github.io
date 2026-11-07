import time
import math

def construct():
    """
    Finds the largest subset of distinct integers in [2, 40] with distinct subset products.
    This is equivalent to finding the largest multiplicatively independent set.
    Multiplicative independence is equivalent to linear independence of prime exponent vectors.
    The primes <= 40 are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    There are 12 such primes, so the maximum possible size is 12.
    We use a greedy approach to select a basis of these vectors.
    """
    
    # Primes in range 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Function to get prime exponent vector for a number
    def get_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Get vectors for all numbers 2..40
    candidates = []
    for n in range(2, 41):
        candidates.append((n, get_vector(n)))
    
    # Greedy selection of linearly independent vectors
    # We want to maximize the number of selected vectors.
    # A greedy approach that picks vectors which increase the rank of the current set will find a basis.
    # To be efficient and ensure we get the maximum size (12), we can just pick the primes first,
    # as they form a standard basis. However, a generic linear independence check is safer 
    # and aligns with the "stronger algorithm" directive.
    
    selected_vectors = []
    selected_numbers = []
    
    # We iterate through candidates. To ensure we find a large independent set,
    # we can sort candidates or just iterate. Since the primes are in the set,
    # and they form a basis, any order that includes them will work.
    # A simple greedy: add vector if it is linearly independent of current selected.
    
    def is_independent(vec, current_basis):
        # Check if vec is in the span of current_basis
        # Since dimensions are small (12), we can use Gaussian elimination on the matrix
        # formed by current_basis + [vec] and check if rank increases.
        # Or simpler: solve linear system.
        # For performance in a tight loop, we can maintain a row-reduced form.
        # But for N=39, D=12, even O(N * D^3) is tiny.
        
        # Create matrix with current basis vectors and the new vector
        # Actually, we can just check if the new vector is linearly dependent.
        # We can use a simple rank check by appending and reducing.
        
        # Efficient check:
        # Maintain the basis in row-echelon form.
        # If we can reduce the new vector to zero using the basis, it's dependent.
        
        # Let's implement a helper to check independence using rational arithmetic (fractions)
        # or just integer arithmetic since vectors are small integers.
        
        # We'll use a simple Gaussian elimination on the list of vectors including the new one.
        # If the rank of {basis + new} > rank of {basis}, it's independent.
        
        # To avoid recalculating rank from scratch, we can maintain the basis in reduced row echelon form (RREF).
        # But for simplicity and given the small size, we can just do a full check.
        
        # Let's use a simpler check:
        # The vectors are in Z^12.
        # We can compute the determinant of the matrix formed by the basis vectors and the new vector?
        # No, the matrix is not square if size != 12.
        
        # Let's just use the property that if we have a basis, we can express any vector as a linear combination.
        # If the coefficients are integers, it's dependent in the integer lattice?
        # No, linear independence over Q is what matters for multiplicative independence.
        # If a vector is in the span of the basis over Q, it's dependent.
        
        # We will maintain the basis as a list of tuples.
        # To check independence, we try to reduce the new vector by the basis vectors.
        
        # Optimization: Pre-compute the basis in RREF.
        # Since we are selecting greedily, we can update the RREF incrementally.
        
        pass # Placeholder for logic below

    # Let's implement the RREF update logic properly.
    # Basis will be a list of lists (floats or fractions) in RREF.
    # Actually, using `fractions.Fraction` is safe and exact.
    from fractions import Fraction
    
    basis = [] # List of RREF rows
    selected_numbers = []
    
    for n, vec in candidates:
        v = [Fraction(x) for x in vec]
        
        # Try to reduce v using the current basis
        # For each row in basis, if the pivot of the row matches a non-zero element in v, subtract
        # We need to know the pivot positions.
        
        # Let's store (pivot_index, row)
        # basis is a list of (pivot_idx, row)
        
        is_dep = False
        temp_v = v[:]
        
        for pivot_idx, row in basis:
            if temp_v[pivot_idx] != 0:
                factor = temp_v[pivot_idx] / row[pivot_idx]
                for j in range(len(primes)):
                    temp_v[j] -= factor * row[j]
        
        # If temp_v is all zeros, it's dependent
        if all(x == 0 for x in temp_v):
            continue # Dependent, skip
            
        # It is independent. Add it to the basis.
        # Find the pivot for this new row.
        # The first non-zero element in temp_v is the pivot.
        pivot_idx = -1
        for i in range(len(primes)):
            if temp_v[i] != 0:
                pivot_idx = i
                break
        
        # Normalize the row so the pivot is 1
        factor = temp_v[pivot_idx]
        for i in range(len(primes)):
            temp_v[i] /= factor
        temp_v[pivot_idx] = Fraction(1)
        
        # Eliminate this pivot from existing basis rows
        new_basis = []
        for p_idx, row in basis:
            if row[pivot_idx] != 0:
                factor_to_eliminate = row[pivot_idx]
                new_row = row[:]
                for j in range(len(primes)):
                    new_row[j] -= factor_to_eliminate * temp_v[j]
                new_basis.append((p_idx, new_row))
            else:
                new_basis.append((p_idx, row))
        
        new_basis.append((pivot_idx, temp_v))
        
        # Sort basis by pivot index to maintain order (optional but good for consistency)
        new_basis.sort(key=lambda x: x[0])
        basis = new_basis
        selected_numbers.append(n)
        
        # If we have selected 12 vectors, we have a full basis. We can stop.
        if len(selected_numbers) == len(primes):
            break

    return selected_numbers
