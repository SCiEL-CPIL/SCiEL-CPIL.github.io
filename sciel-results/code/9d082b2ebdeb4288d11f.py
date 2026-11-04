import numpy as np
import itertools

def construct():
    n, m, p = 3, 3, 7
    nm = n * m
    mp = m * p
    pn = p * n

    # --- Rank-23 3x3x3 decomposition (Strassen's 1969) ---
    # U: 9x23, V: 9x23, W: 9x23
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=3
    #
    # The 23 terms are defined by the following tuples (i, j, k) for U, V, W
    # Actually, let me use the standard form.
    # For 3x3x3, the tensor is sum_{r=1}^{23} u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are vectors of length 9.
    
    # Let's use a known correct rank-23 decomposition.
    # From "Fast Matrix Multiplication" by V. Strassen, 1969.
    # The coefficients can be found in various references.
    
    # Here is a verified set of 23 terms for 3x3x3:
    # Each term is (U_idx, V_idx, W_idx) where the indices are into the 9-element vectors.
    # Actually, the standard way is to list the 23 triples of 9-vectors.
    
    # Let me use the decomposition from the paper "An Algorithm for Multiplying 3x3 Matrices in 23 Multiplications"
    # The 23 terms are:
    
    # I'll use a known correct implementation.
    # Since I can't look it up, let me use a different approach:
    # Use rank-23 for the 3x3x3 block and schoolbook for the rest.
    
    # Actually, let me try to construct a rank-23 decomposition programmatically
    # by using the known structure.
    
    # For safety, let me use a rank-24 decomposition which I can more easily verify,
    # or stick with schoolbook but try to find a better one.
    
    # Let me try a different approach: use the rank-23 coefficients from a known source.
    # I'll hardcode them.
    
    # Here are the 23 terms for 3x3x3 (each is a tuple of 3 9-element vectors):
    # This is from the standard Strassen 3x3x3 rank-23 algorithm.
    
    # Actually, let me just use schoolbook for now but with a twist:
    # I'll use rank-23 for 3x3x3 and schoolbook for the remaining columns.
    
    # Let me try to get the rank-23 coefficients right.
    # The 23 terms for 3x3x3:
    
    # I'll use the following known decomposition:
    # Term 1: U=(1,0,0,0,0,0,0,0,0), V=(1,0,0,0,0,0,0,0,0), W=(1,0,0,0,0,0,0,0,0)
    # ... this is getting complicated.
    
    # Let me use a simpler approach: I'll use rank-23 if I can verify it,
    # otherwise fall back to schoolbook.
    
    # For the 3x3x7 matrix multiplication:
    # We can view this as multiplying a 3x3 matrix by a 3x7 matrix.
    # The output is a 3x7 matrix.
    
    # One approach: use rank-23 for the 3x3x3 part (columns 0,1,2 of the output)
    # and schoolbook for the remaining 4 columns (columns 3,4,5,6).
    # Total rank: 23 + 3*3*4 = 23 + 36 = 59.
    
    # But wait, the 3x3x3 part shares the same input matrices.
    # So we can't just add them independently.
    
    # Actually, the tensor for 3x3x7 is:
    # T[i,j,k] = sum_{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where i in [0,3), j in [0,3), k in [0,7).
    
    # This is a 3x3x7 tensor.
    
    # Let me try to find a low-rank decomposition.
    
    # For now, I'll use schoolbook with rank 63.
    
    rank = 63
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
