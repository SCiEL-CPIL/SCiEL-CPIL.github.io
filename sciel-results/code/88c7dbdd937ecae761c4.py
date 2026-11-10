import time

_DEADLINE = 85

def _prime_factor_vector(n):
    """Return the prime exponent vector for n, indexed by primes <= 40."""
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    vec = [0] * len(primes)
    for i, p in enumerate(primes):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return vec

def _is_independent(vec, basis):
    """
    Check if vec is linearly independent from basis.
    basis is a list of vectors (lists of ints).
    Returns the reduced vector if independent, None if dependent.
    We perform Gaussian elimination.
    """
    v = list(vec)
    for b in basis:
        # Find pivot in b
        pivot_idx = -1
        for i in range(len(v)):
            if b[i] != 0:
                pivot_idx = i
                break
        if pivot_idx == -1:
            continue
            
        if v[pivot_idx] != 0:
            # Eliminate v[pivot_idx] using b
            factor = v[pivot_idx] / b[pivot_idx]
            # To keep integers, we can scale, but for independence over Q, 
            # we just need to check if it can be zeroed out.
            # Actually, simpler: maintain basis in row-echelon form.
            # If we maintain basis in row-echelon form, we can just check if 
            # the new vector has a non-zero component in a column where 
            # no basis vector has a pivot.
            
            # Let's stick to a simpler check for small D:
            # If we have D=12, we can just solve the linear system?
            # Or, simpler: 
            # If the vector reduces to 0 when subtracting multiples of basis vectors, it's dependent.
            
            # Let's implement a simple reduction.
            # For each basis vector, if it has a pivot at col j, 
            # subtract (v[j] / b[j]) * b from v.
            # To avoid floats, we can check divisibility? 
            # No, linear independence over Q is the same as over Z for integer vectors?
            # Yes, if a rational combination exists, an integer combination exists (by clearing denominators).
            # So we can work with integers.
            
            # However, simple elimination with integers might blow up? 
            # D=12 is small.
            
            # Let's just use the property: 
            # If we maintain basis in row echelon form (unique pivot column for each basis vector),
            # then a new vector is independent if and only if it has a non-zero entry 
            # in a column that is NOT a pivot column of any basis vector.
            
            pass # Placeholder for logic below

    # Refactored logic for _is_independent:
    # Maintain basis as a list of (pivot_col, vector) pairs.
    # This is more efficient.
    pass

def construct():
    """Return up to 18 distinct integers in 2..40 with all subset products distinct."""
    t0 = time.time()
    
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # basis: list of tuples (pivot_col, vector)
    basis = []
    selected = []
    
    for n in range(2, 41):
        if time.time() > _DEADLINE:
            break
            
        vec = _prime_factor_vector(n)
        
        # Check independence
        # We try to reduce vec using the current basis
        v = list(vec)
        is_dep = False
        
        # Iterate through basis in order of pivot columns
        # To make it efficient, we can sort basis by pivot col, but D is small.
        for pivot_col, b_vec in basis:
            if v[pivot_col] != 0:
                # We need to eliminate v[pivot_col]
                # v = v - (v[pivot_col] / b_vec[pivot_col]) * b_vec
                # Since we are checking for ZERO, we can just check 
                # if the vector can be reduced to zero.
                
                # Actually, if we maintain row echelon form, 
                # the pivot column of b_vec has 0s in all other basis vectors.
                # So we can just do:
                factor = v[pivot_col]
                # We don't actually need to update v fully if we just want to know 
                # if it becomes zero. 
                # But we do need to update v to check against subsequent pivots.
                
                # To keep integers:
                # v = v * b_vec[pivot_col] - factor * b_vec
                # This scales v, but that's fine for independence.
                
                new_v = [0] * num_primes
                for i in range(num_primes):
                    new_v[i] = v[i] * b_vec[pivot_col] - factor * b_vec[i]
                v = new_v
                
        # After reduction, if v is all zeros, it's dependent.
        if all(x == 0 for x in v):
            is_dep = True
        else:
            # It's independent. 
            # We need to find its pivot column.
            # In row echelon form, the pivot is the first non-zero column.
            # However, our reduction order depends on the order of basis.
            # To maintain proper echelon form, we should insert it at the correct position.
            
            # Find the first non-zero index in v
            pivot_idx = -1
            for i in range(num_primes):
                if v[i] != 0:
                    pivot_idx = i
                    break
            
            if pivot_idx != -1:
                # Check if this pivot column is already taken
                # In a proper echelon basis, each pivot col is unique.
                # If we just append, we might violate the order.
                # But for independence check, as long as we reduce using ALL basis vectors, 
                # it works.
                
                # To be safe, let's just add it. 
                # The reduction logic above assumes that the basis vectors 
                # have pivots in distinct columns and are ordered.
                # If we just append, the next vector will be reduced by this one.
                
                basis.append((pivot_idx, v))
                selected.append(n)
                
                if len(selected) >= 18:
                    break

    # The selected numbers form a basis.
    # They are multiplicatively independent.
    # Size is 12.
    
    return [int(x) for x in selected]
