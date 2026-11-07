"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. Model each integer 2..40 as a vector of prime exponents in the basis of primes <= 40.
    2. The condition that all subset products are distinct is equivalent to the linear independence
       of these vectors over the rationals.
    3. The maximum number of such integers is the rank of the matrix formed by these vectors.
    4. We use Gaussian elimination to find a maximal linearly independent subset (a basis).
    5. This approach is O(N * D^2) where N=39 and D=12, making it nearly instantaneous.
    """
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = len(primes)
    
    # Map each number to its exponent vector
    # We use tuples for hashing and ease of manipulation in a list
    vectors = {}
    for x in range(2, 41):
        v = [0] * d
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        # temp should be 1 since we included all primes up to 40
        vectors[x] = tuple(v)
        
    # Gaussian elimination to find a maximal independent subset
    # We maintain a list of basis vectors (as lists to allow modification)
    # and a list of corresponding original numbers
    basis = []
    basis_nums = []
    
    # Iterate through numbers. Order doesn't strictly matter for finding *a* basis,
    # but picking primes first might be intuitive. However, any maximal independent set works.
    # We'll just iterate 2..40.
    for x in range(2, 41):
        v = list(vectors[x])
        # Try to reduce v using the current basis
        # If v becomes zero, it is dependent.
        # We perform row reduction: v = v - c * basis_vec
        # Since we are over Q, we can use fractions or just check if it's a multiple.
        # Actually, simpler: add to a matrix and compute rank?
        # Or just do elimination.
        
        dependent = False
        # To avoid floating point issues, we can use fractions or integer arithmetic with cross-multiplication.
        # But since we just need independence, we can use a simpler check:
        # Is v in the span of basis?
        # Given small D, we can just try to eliminate.
        
        # Standard Gaussian elimination on the vector v against basis vectors
        # Note: basis vectors are not necessarily orthogonal, but they are independent.
        # We can reduce v by subtracting appropriate multiples of basis vectors.
        # To keep it simple and exact, we use Fractions.
        from fractions import Fraction
        
        # Convert v to Fractions
        v_frac = [Fraction(x, 1) for x in v]
        
        for b_idx, b_vec in enumerate(basis):
            # b_vec is a tuple of ints. Convert to Fraction for calculation?
            # Or just do: if v_frac[i] != 0, we need to eliminate it.
            # Find the first non-zero component in v_frac.
            pivot_idx = -1
            for i in range(d):
                if v_frac[i] != 0:
                    pivot_idx = i
                    break
            
            if pivot_idx == -1:
                # v is zero, so it is dependent on the current basis
                dependent = True
                break
            
            # Find a basis vector that has a non-zero component at pivot_idx
            # Actually, standard elimination assumes basis is in row-echelon form.
            # Let's maintain basis in row-echelon form.
            # If we add a new vector, we should reduce it against existing basis.
            # But existing basis might not be in echelon form relative to each other if we just appended.
            # Let's maintain a list of basis vectors.
            # To check if v is independent, we can just add it to a matrix and check rank?
            # Or simpler: just keep a list of independent vectors found so far.
            # To test if v is in the span, we can solve the linear system.
            # But D is small (12). N is 39.
            # We can just collect all vectors, form a matrix, and find the rank and a basis.
            
            # Let's restart the strategy slightly:
            # 1. Collect all vectors.
            # 2. Perform Gaussian elimination on the entire set to find a pivot set.
            # This is cleaner.
            
            pass # Break out of this inner loop logic, we'll do it outside
            
        if dependent:
            continue
            
        # If we get here, the logic above was messy. Let's just do it properly outside.
        pass

    # Proper Gaussian Elimination on the set of all vectors
    # We want to find the maximum number of linearly independent vectors.
    # We will process vectors one by one and maintain a basis in row-echelon form.
    
    basis = []  # List of lists (vectors)
    basis_nums = [] # Corresponding numbers
    
    for x in range(2, 41):
        v = list(vectors[x])
        # Reduce v against current basis
        # Basis is in row-echelon form: pivot of basis[i] is at column i (or some increasing col)
        # Let's maintain basis such that basis[i] has its first non-zero element at column i.
        # This is standard Gaussian elimination.
        
        # We need to find the pivot column for v.
        # But v might need to be reduced by multiple basis vectors.
        
        # Let's use a different representation:
        # We maintain a list of basis vectors.
        # To check if v is independent, we check if it can be reduced to zero.
        
        # Since D is small, we can just use a linear algebra library or simple elimination.
        # Let's implement simple elimination with Fractions.
        
        from fractions import Fraction
        v_frac = [Fraction(x, 1) for x in v]
        
        is_independent = True
        
        # Reduce v_frac using basis
        # We iterate through basis vectors. If basis[i] has a pivot at col c,
        # and v_frac[c] is non-zero, we subtract (v_frac[c] / basis[i][c]) * basis[i] from v_frac.
        
        # To make this efficient, we should store the pivot column for each basis vector.
        # Let's restructure:
        # basis is a list of (pivot_col, vector_tuple)
        
        # Actually, let's just build the full matrix and use a standard rank algorithm.
        # Given N=39, D=12, it's tiny.
        
        pass

    # Let's do it cleanly:
    # 1. Create list of all vectors.
    # 2. Perform Gaussian elimination to find a basis.
    
    all_vectors = [(x, list(vectors[x])) for x in range(2, 41)]
    
    basis = []
    basis_nums = []
    
    # We will process vectors and add to basis if independent.
    # We maintain basis in row-echelon form.
    # basis[i] = (pivot_col, vector_list)
    
    for x, v in all_vectors:
        v_work = [Fraction(val, 1) for val in v]
        
        # Reduce v_work
        for b_idx, (p_col, b_vec) in enumerate(basis):
            # b_vec is a list of Fractions? Or we store as list of Fractions in basis?
            # Let's store basis as list of lists of Fractions, and pivot cols.
            pass
            
    # This is getting complicated to write inline. Let's use a simpler check.
    # Since D=12, we can just keep a list of independent vectors.
    # To check if a new vector v is independent from list L:
    # We can form a matrix with rows L + [v] and check if rank increased.
    # Or simpler: try to reduce v to zero using L.
    
    # Let's use the reduction method with Fractions.
    
    basis = [] # List of lists of Fractions
    basis_pivots = [] # List of pivot columns
    basis_nums = []
    
    for x in range(2, 41):
        v = [Fraction(val, 1) for val in vectors[x]]
        
        # Reduce v
        for i in range(len(basis)):
            p_col = basis_pivots[i]
            if v[p_col] != 0:
                # Eliminate v[p_col] using basis[i]
                factor = v[p_col] / basis[i][p_col]
                for j in range(d):
                    v[j] -= factor * basis[i][j]
        
        # Check if v is zero
        if all(val == 0 for val in v):
            continue # Dependent
        
        # Find pivot column for v
        pivot_col = -1
        for j in range(d):
            if v[j] != 0:
                pivot_col = j
                break
        
        if pivot_col == -1:
            continue # Should not happen if not zero
            
        # Add to basis
        basis.append(v)
        basis_pivots.append(pivot_col)
        basis_nums.append(x)
        
        # Optional: Re-order basis to keep pivots sorted?
        # For correctness of reduction, it's better if pivots are in increasing order.
        # If we just append, the pivots might not be sorted.
        # If pivot of new vector is smaller than existing pivots, we should insert earlier.
        # But since we reduce against existing basis, the new pivot should be "new".
        # Actually, if we don't sort, the reduction might fail if a later basis vector
        # has a pivot in a column that an earlier vector could have eliminated.
        # Standard Gaussian elimination processes columns left to right.
        # So we should ensure that when we add a vector, its pivot is to the right of all existing pivots?
        # No, we just need to be able to reduce.
        # If we process vectors in any order, and reduce against the current basis,
        # we need the basis to be in a form that allows reduction.
        # Row-echelon form requires pivots to move to the right.
        # So we should insert the new vector into the basis at the position corresponding to its pivot,
        # and potentially re-reduce?
        # Actually, simpler: Just collect all vectors, form a matrix, and run standard Gaussian elimination
        # to find the rank and a basis.
        
        pass
        
    # Let's restart the implementation inside the function to be clean.
    
    from fractions import Fraction
    
    # 1. Prepare vectors
    vecs = []
    for x in range(2, 41):
        v = [0] * d
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs.append((x, v))
        
    # 2. Gaussian Elimination to find Max Independent Set
    # We maintain a list of basis vectors (as lists of Fractions)
    # and their corresponding numbers.
    # We process vectors one by one.
    
    basis = []
    basis_nums = []
    
    for x, v in vecs:
        v_frac = [Fraction(val, 1) for val in v]
        
        # Reduce v_frac against current basis
        # We assume basis is in row-echelon form (pivots increasing)
        # To maintain this, we should insert new vectors appropriately.
        # However, for a small set, we can just check independence by
        # trying to reduce to zero.
        
        # If we just append, the basis might not be in echelon form.
        # Let's ensure we find the pivot and insert in sorted order of pivots.
        
        # First, reduce against existing basis
        # We iterate through basis in order of pivot columns.
        # Since we don't have sorted pivots yet, let's just do a full reduction.
        # Actually, if we always reduce against the *current* basis, and the basis is independent,
        # the result is unique up to the representation.
        # If v reduces to zero, it's dependent.
        # If not, it's independent.
        
        # To make reduction work correctly, the basis vectors should be "orthogonal" in the sense
        # that their pivots are distinct and we eliminate from left to right.
        # Let's maintain basis sorted by pivot column.
        
        # 1. Reduce v against all basis vectors
        # We need to know the pivot of each basis vector.
        # Let's store basis as list of (pivot, vec)
        
        pass
        
    # Okay, I will write the full clean logic now.
    
    # Re-initialize
    basis = [] # List of (pivot_col, vec_list_of_Fractions)
    basis_nums = []
    
    for x in range(2, 41):
        v = [Fraction(val, 1) for val in vectors[x]]
        
        # Reduce v
        # We iterate through basis. To be safe, we should iterate in order of pivot columns.
        # Since we insert in order, basis is sorted by pivot.
        for b_idx in range(len(basis)):
            p_col = basis[b_idx][0]
            b_vec = basis[b_idx][1]
            if v[p_col] != 0:
                factor = v[p_col] / b_vec[p_col]
                for j in range(d):
                    v[j] -= factor * b_vec[j]
                    
        # Check if zero
        if all(val == 0 for val in v):
            continue
            
        # Find pivot
        pivot_col = -1
        for j in range(d):
            if v[j] != 0:
                pivot_col = j
                break
                
        if pivot_col == -1:
            continue
            
        # Insert into basis in sorted order of pivot_col
        # Find insertion point
        ins_idx = 0
        while ins_idx < len(basis) and basis[ins_idx][0] < pivot_col:
            ins_idx += 1
            
        basis.insert(ins_idx, (pivot_col, v))
        basis_nums.insert(ins_idx, x)
        
    # The size of basis is the maximum number of independent integers
    # Return the numbers
    return [int(x) for x in basis_nums]

# EVOLVE-BLOCK-END
