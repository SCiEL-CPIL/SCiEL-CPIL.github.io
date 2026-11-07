"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import itertools

P = {'n': 40, 'maximum_size': 18}

def _is_valid_subset(nums):
    """
    Check if all 2^k subset products are distinct.
    This is equivalent to checking if the numbers are multiplicatively independent.
    """
    k = len(nums)
    if k == 0:
        return True
    
    # Factorize each number into prime exponents
    # Primes up to 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def factorize(n):
        exponents = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                exponents[i] += 1
                n //= p
        return tuple(exponents)
    
    factors = [factorize(x) for x in nums]
    
    # Use DP to compute exponent vectors for all subsets
    # vector[mask] = sum of factors[i] for i in mask
    # We check if any two masks have the same vector.
    # Since k <= 18, 2^k <= 262144.
    # We can store vectors in a set.
    
    seen = set()
    # Empty subset
    seen.add(tuple([0]*len(primes)))
    
    # We can compute vector for mask from mask without lowest set bit
    # But we need to store all vectors to check for duplicates.
    # Storing 2^k tuples of length 12 is feasible.
    
    # Optimization: If k is large, we might run out of time?
    # 2^18 * 12 operations is ~3 million ops. Very fast in Python if optimized.
    
    # Let's use a list to store vectors for DP
    # Actually, we can just generate them on the fly and check against a set.
    # But we need the previous vector to compute the next.
    # So we need an array of vectors.
    
    num_subsets = 1 << k
    vectors = [None] * num_subsets
    vectors[0] = tuple([0]*len(primes))
    seen.add(vectors[0])
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        prev_vec = vectors[prev_mask]
        curr_factors = factors[idx]
        
        # Sum vectors
        new_vec = tuple(p + c for p, c in zip(prev_vec, curr_factors))
        
        if new_vec in seen:
            return False
        seen.add(new_vec)
        vectors[mask] = new_vec
        
    return True

def _build_conflict_graph(nums):
    """
    Build a conflict graph for the given list of candidate numbers.
    Nodes are indices 0..len(nums)-1.
    An edge (i, j) exists if {nums[i], nums[j]} is NOT a valid set (i.e., they are dependent).
    Wait, the condition is that ALL subset products are distinct.
    This is a global property. It is not pairwise.
    Two numbers a and b are "compatible" if {a, b} is valid.
    But a set of 3 numbers {a, b, c} is valid iff {a,b}, {a,c}, {b,c} are valid AND {a,b,c} is valid.
    Actually, multiplicative independence is a linear algebra property over the rationals (or integers with prime exponents).
    The vectors of prime exponents must be linearly independent.
    Therefore, a set is valid if and only if the corresponding exponent vectors are linearly independent.
    
    This means the "conflict" is not just pairwise.
    However, if a set of vectors is linearly dependent, there exists a minimal dependent subset.
    But for the Independent Set problem, we usually define conflicts pairwise.
    Does pairwise independence imply set independence?
    NO. {2, 3, 6} -> 2*3 = 6. Pairwise: {2,3} ok, {2,6} ok (2!=6, 1!=2, 1!=3... wait.
    Products for {2,6}: 1, 2, 6, 12. Distinct.
    Products for {2,3,6}: 1, 2, 3, 6, 6, 12, 18, 36. Duplicate 6.
    So {2,3} is valid, {2,6} is valid, {3,6} is valid, but {2,3,6} is invalid.
    
    So a simple pairwise conflict graph (where edge means "cannot be in same set") is NOT sufficient to model this problem as a standard MIS.
    The property is "Linear Independence" of the exponent vectors.
    
    This is the problem of finding the largest linearly independent subset of vectors.
    This is equivalent to finding the Rank of the matrix formed by these vectors.
    Wait, we are selecting a subset of the given vectors.
    The maximum size of a linearly independent subset of a set of vectors is exactly the Rank of the matrix formed by all vectors.
    
    Is that true?
    Yes. The maximum cardinality of a linearly independent subset of a set of vectors is the dimension of the span of that set, which is the rank.
    
    So, the problem reduces to:
    1. Convert each number 2..40 to its prime exponent vector.
    2. Form a matrix M where rows are these vectors.
    3. Find the Rank of M.
    4. The maximum number of integers we can select is the Rank.
    5. We need to return the specific integers corresponding to a basis of this matrix.
    
    This is a deterministic, polynomial-time problem! No search needed.
    
    Let's verify this logic.
    Condition: All 2^k subset products are distinct.
    This is equivalent to: No non-trivial subset product equals 1.
    Which is equivalent to: No non-trivial integer linear combination of the exponent vectors equals the zero vector.
    Which is equivalent to: The exponent vectors are linearly independent over the integers (and thus over the reals).
    
    So yes, we just need to find a basis for the vector space spanned by the prime exponent vectors of numbers 2..40.
    The size of this basis is the answer.
    And we can extract the basis vectors (and thus the numbers) using Gaussian Elimination.
    """
    pass

def construct():
    """
    Construct a set of distinct integers in 2..40 with distinct subset products.
    Maximize the number of selected integers.
    """
    # The problem is equivalent to finding the maximum linearly independent subset of prime exponent vectors.
    # This is the rank of the matrix of these vectors.
    
    # 1. Generate prime exponent vectors for all numbers 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return vec
    
    # Map: number -> vector
    vectors = {}
    for n in range(2, 41):
        vectors[n] = get_vector(n)
        
    # 2. Perform Gaussian Elimination to find a basis
    # We want to select the largest set of linearly independent vectors.
    # Standard Gaussian elimination finds a basis.
    # The size of the basis is the rank.
    # We need to track which original numbers correspond to the basis vectors.
    
    # Let's use a list of (vector, number) pairs
    items = [(vectors[n], n) for n in range(2, 41)]
    
    # We will process them and maintain a basis.
    # Basis is a list of vectors.
    # We want to maximize the number of selected numbers.
    # Any basis will have the same size (the rank).
    # So we can just pick the first independent vectors we find.
    
    basis = []
    selected_numbers = []
    
    # To ensure we get a "nice" basis, we can sort numbers or just process 2..40.
    # The rank is invariant.
    
    # Let's use a standard row-echelon form approach.
    # We maintain a basis in row-echelon form.
    # For each new vector, we try to reduce it using the existing basis.
    # If it reduces to zero, it's dependent.
    # If not, it's independent, and we add it to the basis (and update basis to maintain echelon form).
    
    # Actually, simpler:
    # Just keep a list of independent vectors found so far.
    # For each new vector v:
    #   Check if v is in the span of current_basis.
    #   If not, add v to current_basis.
    
    # How to check if v is in span?
    # Perform Gaussian elimination on [current_basis; v].
    # Or, maintain current_basis in row-echelon form.
    
    basis_re = []  # Row Echelon Form basis
    
    for n in range(2, 41):
        v = vectors[n]
        # Try to reduce v using basis_re
        # basis_re is a list of vectors.
        # Each vector in basis_re has a leading 1 (or non-zero pivot) at a specific column.
        # Let's store (pivot_col, vector) for each basis element.
        
        # Actually, let's just store the vectors and do elimination on the fly.
        # Since dimension is small (12), we can just do full elimination.
        
        is_independent = True
        temp_v = v[:]
        
        # Reduce temp_v using existing basis
        for b_vec in basis_re:
            # Find pivot of b_vec
            # We want b_vec to be in row echelon form.
            # Let's maintain basis_re such that each vector has a unique pivot column.
            
            # Find first non-zero in b_vec
            pivot_col = -1
            for j in range(num_primes):
                if b_vec[j] != 0:
                    pivot_col = j
                    break
            
            if pivot_col == -1:
                continue
                
            # If temp_v has non-zero at pivot_col, eliminate it
            if temp_v[pivot_col] != 0:
                # Factor
                factor = temp_v[pivot_col] / b_vec[pivot_col]
                # Since we are dealing with integers, and we are checking independence over reals/rationals,
                # we can use fractions or just float?
                # The vectors are integer. The span is over Q.
                # We can use Fraction from fractions module for exactness.
                # Or, since numbers are small, we can just use the property that if integer combination exists, rational does.
                # Let's use float for speed, but be careful.
                # Actually, let's use the property: v is dependent iff det of matrix with v is 0?
                # No, simpler: use Gaussian elimination with Fractions.
                
                pass # Placeholder for logic
                
        # Let's restart the basis construction with a cleaner implementation using Fractions.
        
    # Re-implementing basis construction correctly
    
    from fractions import Fraction
    
    basis_re = [] # List of (pivot_col, vector_of_Fractions)
    selected = []
    
    for n in range(2, 41):
        v = [Fraction(x) for x in vectors[n]]
        
        # Reduce v
        for pivot_col, b_vec in basis_re:
            if v[pivot_col] != 0:
                factor = v[pivot_col] / b_vec[pivot_col]
                for j in range(pivot_col, num_primes):
                    v[j] -= factor * b_vec[j]
        
        # Check if v is zero
        is_zero = all(x == 0 for x in v)
        
        if not is_zero:
            # Find new pivot
            new_pivot = -1
            for j in range(num_primes):
                if v[j] != 0:
                    new_pivot = j
                    break
            
            # Add to basis
            # To maintain echelon form, we might need to reorder or back-substitute?
            # For just finding independence, we just need to ensure pivots are unique and increasing.
            # If new_pivot is already in basis, we should have eliminated it.
            # Since we iterate basis_re in order of pivot columns, and we eliminated all existing pivots,
            # new_pivot must be a column not yet used as a pivot.
            
            # However, to maintain the property that "v[pivot_col] is non-zero and is the first non-zero",
            # we should store the vector as is.
            
            basis_re.append((new_pivot, v))
            selected.append(n)
            
    # The list 'selected' contains the numbers that form a basis.
    # The size is the rank.
    # We should verify that this set is indeed valid (distinct subset products).
    
    if not selected:
        return [2]
        
    if not _is_valid_subset(selected):
        # Fallback if something went wrong (should not happen)
        # Try to find a valid set manually?
        # For now, return a known small valid set
        return [2, 3]
        
    return selected

# EVOLVE-BLOCK-END
