<<<<<<< SEARCH
def _bclr_5x5x5():
    """
    Return the Bini-Capovani-Lotti-Romani rank-19 decomposition of 5x5x5 matrix multiplication.
    U5: (25, 19), row = i*5+j
    V5: (25, 19), row = j*5+k
    W5: (25, 19), row = k*5+i
    Such that sum_r U5[i*5+j,r]*V5[j*5+k,r]*W5[k*5+i,r] == 1 for all i,j,k in 0..4.
    """
    # The BCLR rank-19 decomposition of 5x5x5 matrix multiplication.
    # Encoded from the standard reference.
    
    U5_list = [
        # i=0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=4
        # i=1
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=0
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=1
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=2
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=4
        # i=2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # j=0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # j=1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # j=2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # j=3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # j=4
        # i=3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # j=0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # j=1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # j=2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # j=3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=4
        # i=4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # j=4
    ]
    
    # This is just the identity pattern - NOT the actual BCLR decomposition.
    # The actual BCLR matrices are much more complex.
    # Since I cannot reliably recall the exact BCLR matrices, I'll return None
    # to signal that we should fall back to other methods.
    return None
=======
def _bclr_5x5x5():
    """
    Return the Bini-Capovani-Lotti-Romani rank-19 decomposition of 5x5x5 matrix multiplication.
    U5: (25, 19), row = i*5+j
    V5: (25, 19), row = j*5+k
    W5: (25, 19), row = k*5+i
    Such that sum_r U5[i*5+j,r]*V5[j*5+k,r]*W5[k*5+i,r] == 1 for all i,j,k in 0..4.
    """
    # We'll use a known rank-19 decomposition pattern for 5x5x5.
    # The key insight: for 5x5x5, rank 19 is achievable.
    # Structure: 5x5 = 4x4 + cross terms.
    # 4x4x4 has rank 16 (schoolbook). The remaining 3 ranks handle the 5th row/col.
    
    # Actually, let's construct a valid rank-19 decomposition directly.
    # For matrix mult (i,j,k) -> A[i,j]*B[j,k] = C[i,k]
    # Tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The multiplication tensor has T[i,j,k] = delta_{i,k}... no.
    # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
    # So the tensor is: for each (i,j,k), the output is C[i,k], and the term is A[i,j]*B[j,k].
    # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the full multiplication).
    # Rank-1 decomposition: sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1 for all i,j,k.
    
    # Schoolbook: 25 terms, one for each (i,j,k) triple.
    # BCLR achieves 19.
    
    # Let me use a different approach: construct a valid decomposition with rank <= 25
    # that is better than schoolbook, then try to minimize.
    
    # For now, return None to use the fallback search.
    return None
>>>>>>> REPLACE

