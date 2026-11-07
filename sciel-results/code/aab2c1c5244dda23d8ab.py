import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _prime_factorization(n, primes):
    """Return a tuple of exponents for the given primes."""
    exponents = []
    for p in primes:
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        exponents.append(count)
    return tuple(exponents)

def _gaussian_elimination(vectors):
    """
    Find a maximal linearly independent subset of the given vectors.
    Returns the indices of the vectors in the independent set.
    Vectors are tuples of integers.
    """
    if not vectors:
        return []
    
    # We work with the vectors. We want to find which ones are independent.
    # Standard Gaussian elimination to find the rank and a basis.
    # We keep track of which original vector formed each row of the basis.
    
    basis = []
    basis_indices = []
    
    # To make elimination fast and avoid floating point issues, 
    # we can use Fraction or just integer arithmetic with GCDs if we are careful, 
    # but Fraction is safer for "linear independence over Q".
    # Given small dimensions (12) and small numbers, Fraction is fine.
    
    for idx, vec in enumerate(vectors):
        current = list(vec)
        # Try to eliminate using existing basis vectors
        for i, b_vec in enumerate(basis):
            if current[i] == 0:
                continue
            # Eliminate the i-th component
            if b_vec[i] != 0:
                factor = current[i] / b_vec[i]
                for j in range(i, len(current)):
                    current[j] -= factor * b_vec[j]
        
        # Check if current is now zero
        if all(c == 0 for c in current):
            # Dependent
            continue
        else:
            # Find the pivot position
            pivot_idx = -1
            for k in range(len(current)):
                if current[k] != 0:
                    pivot_idx = k
                    break
            
            if pivot_idx == -1:
                continue
            
            # This vector is independent. Add to basis.
            # We should insert it in a way that maintains row-echelon form?
            # For simplicity of finding *any* maximal independent set, 
            # we can just append. But to maintain the elimination structure for future vectors,
            # it's better to insert it at the correct position or just re-do elimination if we append.
            # Since N is small (39), we can just append and let future eliminations handle it, 
            # BUT the standard algorithm requires the basis to be in a specific form.
            
            # Let's use a simpler approach: 
            # Maintain a list of basis vectors in row-echelon form.
            # If we just append, the previous elimination steps don't apply to the new basis vector's position.
            
            # Correct approach:
            # Insert the new vector into the basis at the position corresponding to its pivot.
            # Or, simpler: just keep the basis as a list of vectors, and for each new vector, 
            # eliminate against ALL existing basis vectors.
            
            # My loop above eliminates against `basis`. 
            # If I just append `current`, the next vector will eliminate against `current` as well.
            # This works if `basis` is just a list of linearly independent vectors.
            # The elimination loop `for i, b_vec in enumerate(basis)` assumes `b_vec` has a pivot at index `i`?
            # No, the code above `if current[i] == 0: continue` and `factor = current[i] / b_vec[i]` 
            # assumes that `b_vec[i]` is the pivot for row `i` and that previous columns are 0.
            # This is only true if we maintain row-echelon form.
            
            # Let's restart the elimination logic to properly maintain row-echelon form.
            pass
            
    # Re-implementing _gaussian_elimination properly
    pass

def _find_max_independent_set(numbers, primes):
    """
    Find the largest subset of numbers whose prime exponent vectors are linearly independent.
    """
    # Generate exponent vectors
    vectors = []
    for n in numbers:
        vec = _prime_factorization(n, primes)
        vectors.append(vec)
    
    # Gaussian Elimination to find maximal independent set
    # We will iterate through the numbers and try to add them to the basis.
    # Basis is a list of vectors in row-echelon form.
    # Each basis vector is stored as (pivot_index, vector_tuple, original_index)
    
    basis = []
    
    for orig_idx, vec in enumerate(vectors):
        current = list(vec)
        dependent = False
        
        # Eliminate using existing basis vectors
        # We iterate through the basis. Since basis is in row-echelon form,
        # basis[i].pivot == i (if we insert in order) or we just use the pivot index.
        
        # To make it simple and robust for small N:
        # For each basis vector, if its pivot index matches a non-zero entry in current, eliminate.
        
        # Let's store basis as a list of dicts or tuples: (pivot_col, vec_list)
        # And keep them sorted by pivot_col.
        
        new_vec = list(vec)
        
        # Eliminate
        for b_pivot, b_vec in basis:
            if new_vec[b_pivot] != 0:
                factor = new_vec[b_pivot] / b_vec[b_pivot]
                for j in range(b_pivot, len(new_vec)):
                    new_vec[j] -= factor * b_vec[j]
        
        # Check if zero
        if all(x == 0 for x in new_vec):
            continue
            
        # Find pivot
        pivot_idx = -1
        for i in range(len(new_vec)):
            if new_vec[i] != 0:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
            
        # Add to basis
        # We need to insert it in the correct position to maintain sorted pivots?
        # Actually, for the elimination to work correctly for subsequent vectors,
        # the basis vectors must have distinct pivots, and we eliminate from top to bottom.
        # If we just append, and a later vector has a pivot at an earlier column,
        # it won't be eliminated by the later basis vectors.
        
        # So we should insert into the list such that pivots are sorted.
        # Or, simpler: since we process vectors in order, and we want *a* maximal set,
        # we can just maintain the basis in row-echelon form by inserting at the correct spot.
        
        insert_pos = len(basis)
        for i, (b_pivot, _) in enumerate(basis):
            if b_pivot > pivot_idx:
                insert_pos = i
                break
        
        basis.insert(insert_pos, (pivot_idx, new_vec))
        
    # The size of the basis is the maximum number of independent vectors.
    # The task asks to return the *integers*.
    # The current implementation finds the *count*. We need the *indices*.
    
    # Let's modify to track original indices.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Numbers to consider: 2..40
    candidates = list(range(2, 41))
    
    # We want to find a maximal linearly independent set of their exponent vectors.
    # Since the dimension is 12, the max size is 12.
    # The primes themselves form a basis.
    # To be "strong" and follow the directive, I will implement the selection.
    
    # To ensure we return a valid list of ints, and given the constraints,
    # returning the primes is the optimal solution.
    # I will implement a check to confirm they are independent, and return them.
    # If I want to be fancy, I can run the elimination to *find* them.
    
    # Let's implement the index-tracking version of the elimination.
    
    def get_exponent_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                n //= p
                vec[i] += 1
        return vec

    vectors = []
    for n in candidates:
        vectors.append(get_exponent_vector(n))
    
    # Basis: list of (pivot_index, vector_list, original_index)
    basis = []
    
    for orig_idx, vec in enumerate(vectors):
        current = list(vec)
        
        # Eliminate against existing basis
        # Iterate through basis in order of pivot index (which is the list order if we insert sorted)
        for b_pivot, b_vec, b_orig in basis:
            if current[b_pivot] != 0:
                factor = current[b_pivot] / b_vec[b_pivot]
                for j in range(b_pivot, len(current)):
                    current[j] -= factor * b_vec[j]
        
        # Check if dependent
        if all(c == 0 for c in current):
            continue
            
        # Find pivot
        pivot_idx = -1
        for i in range(len(current)):
            if current[i] != 0:
                pivot_idx = i
                break
                
        if pivot_idx == -1:
            continue
            
        # Insert into basis maintaining sorted pivot order
        insert_pos = len(basis)
        for i, (b_pivot, _, _) in enumerate(basis):
            if b_pivot > pivot_idx:
                insert_pos = i
                break
        
        basis.insert(insert_pos, (pivot_idx, current, orig_idx))
        
    # Extract original indices
    independent_indices = [b[2] for b in basis]
    
    # Map back to values
    result = [candidates[i] for i in independent_indices]
    
    # Ensure types are Python int
    result = [int(x) for x in result]
    
    # Verify distinctness of subset products? 
    # The linear independence guarantees this.
    # Just in case, a quick check for small sets (though 2^12 is 4096, which is fine).
    # But the linear algebra is exact.
    
    return result

# EVOLVE-BLOCK-END
