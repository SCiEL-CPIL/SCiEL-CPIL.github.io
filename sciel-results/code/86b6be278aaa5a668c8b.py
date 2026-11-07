"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    The condition that all subset products are distinct is equivalent to the set of integers
    being multiplicatively independent. This is equivalent to the vectors of their prime
    factorizations being linearly independent over the rationals.
    
    We map each integer n in 2..40 to a vector of exponents of its prime factorization.
    The primes <= 40 are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (12 primes).
    Thus, each number maps to a vector in Q^12.
    
    We want the largest subset of these vectors that is linearly independent.
    Since the dimension is 12, the maximum possible size is 12.
    
    We perform Gaussian elimination (or row reduction) on the set of all available vectors
    to find a maximal linearly independent subset.
    
    To ensure we find the *maximum* size (which will be the rank of the matrix formed by
    all vectors), we can simply compute the rank. However, we need the actual indices.
    
    A greedy approach with Gaussian elimination:
    1. Sort candidates.
    2. Maintain a basis of independent vectors.
    3. For each candidate, check if it is independent of the current basis.
    4. If yes, add it to the basis and the result set.
    
    This greedy approach for linear independence always finds a basis of the span of the
    processed vectors. If we process all vectors, we find a basis of the total span.
    The size of this basis is the rank of the set of all vectors.
    Is it possible that a different ordering yields a larger independent set?
    No, the maximum size of a linearly independent subset is unique (it's the rank).
    Any maximal linearly independent subset has size equal to the rank.
    
    So, we just need to find *any* maximal linearly independent subset.
    The greedy algorithm does exactly this.
    
    We must ensure we return Python ints.
    """
    
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Map each number to its prime factorization vector
    # Vector is a list of length num_primes, with exponents.
    # We use fractions to be safe for linear independence checks over Q,
    # but since exponents are integers, we can use integer arithmetic for elimination
    # if we are careful, or just use floats if we are careful with precision.
    # Given the small numbers (exponents are small), exact integer arithmetic is best.
    
    def factorize(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        # n should be 1 if n <= 40 and we have all primes
        return vec
    
    candidates = list(range(2, 41))
    vectors = []
    for n in candidates:
        vectors.append((n, factorize(n)))
    
    # Gaussian Elimination to find a basis
    # basis is a list of (vector, original_number)
    basis = []
    selected_numbers = []
    
    # We iterate through candidates.
    # To be deterministic and potentially find "simpler" numbers first (though size is fixed by rank),
    # we just iterate 2..40.
    
    # For each vector, we try to add it to the basis.
    # We maintain the basis in row-echelon form.
    
    for n, vec in vectors:
        v = vec[:]  # copy
        # Try to eliminate leading terms using existing basis vectors
        # We assume basis vectors are sorted by their pivot column index.
        
        # If basis is empty, pivot is 0
        if not basis:
            # Find first non-zero
            for i in range(num_primes):
                if v[i] != 0:
                    # This is the pivot
                    basis.append((v, n, i))
                    selected_numbers.append(n)
                    break
            continue
            
        # Sort basis by pivot index to make elimination easier?
        # Actually, we can just iterate through basis and eliminate.
        # Standard Gaussian elimination:
        # For each pivot column in basis, if v has a non-zero entry there,
        # subtract a multiple of the basis vector to zero it out.
        
        # Let's maintain basis such that each vector has a unique pivot column
        # and is normalized (pivot coefficient is 1) for easier reasoning,
        # or just keep integer coefficients.
        
        # Let's use a dictionary for pivot column -> basis vector
        # This makes it O(1) to find the basis vector for a pivot.
        
        # Re-implementing with pivot map for efficiency and clarity
        # We'll restart the logic slightly for correctness within the loop structure.
        
    # Let's restart the selection process with a cleaner implementation using a pivot map.
    
    pivot_to_basis_vec = {}
    pivot_to_num = {}
    selected = []
    
    for n, vec in vectors:
        v = vec[:]
        
        # Eliminate using existing pivots
        # We need to know which columns are pivots.
        # The keys of pivot_to_basis_vec are the pivot columns.
        
        # To eliminate, we look at the current vector v.
        # For each pivot column p in pivot_to_basis_vec:
        #   if v[p] != 0:
        #       v = v - (v[p] / pivot_coeff) * basis_vec
        #       But we want to keep integers.
        #       Let basis_vec have pivot coefficient c_p.
        #       We can do: v = c_p * v - v[p] * basis_vec
        #       This keeps integers but scales v.
        #       Alternatively, since we only care about zero/non-zero for independence,
        #       we can use Fraction or just float if we are careful.
        #       Given small numbers, let's use Fraction for robustness, or just int with scaling.
        #       Actually, simplest: use list of Fractions.
        
        # Let's convert to Fractions for the elimination step to avoid scaling issues.
        from fractions import Fraction
        v_frac = [Fraction(x) for x in v]
        
        for p, b_vec in pivot_to_basis_vec.items():
            if v_frac[p] != 0:
                # b_vec is stored as Fractions too? 
                # Let's store basis vectors as Fractions.
                # b_vec[p] is the pivot coefficient.
                factor = v_frac[p] / b_vec[p]
                for i in range(num_primes):
                    v_frac[i] -= factor * b_vec[i]
        
        # Check if v_frac is zero vector
        if all(x == 0 for x in v_frac):
            # Dependent
            continue
        else:
            # Independent. Find the pivot column.
            # The pivot should be the first non-zero column in the reduced vector?
            # Not necessarily, but for a basis in row-echelon form, yes.
            # However, since we eliminated against *all* existing pivots,
            # the resulting vector has 0 in all existing pivot columns.
            # So the new pivot is the first non-zero column in v_frac.
            new_pivot = -1
            for i in range(num_primes):
                if v_frac[i] != 0:
                    new_pivot = i
                    break
            
            if new_pivot != -1:
                # Add to basis
                pivot_to_basis_vec[new_pivot] = v_frac
                pivot_to_num[new_pivot] = n
                selected.append(n)
                
    # The number of selected elements is the rank of the set of all vectors.
    # This is the maximum possible size of a multiplicatively independent set
    # drawn from 2..40.
    
    # Return as list of ints
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
