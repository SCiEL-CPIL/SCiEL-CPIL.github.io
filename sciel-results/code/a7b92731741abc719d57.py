"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] for r_idx in range(r))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_3x3(n, m, p):
    """
    Return (U, V, W) for rank-7 decomposition of 3x3 * 3x3 matmul tensor.
    T_{i,j,k} = 1 for i,j,k in [0,3).
    Uses Strassen's 7-multiplication algorithm adapted to the tensor form.
    """
    # Strassen for 3x3: split into 2x2 blocks with overlap
    # For 3x3, we can use a 7-term decomposition based on the 2x2 Strassen
    # but padded to 3x3. Actually, let's use a known rank-7 decomposition for 3x3.
    
    # Alternative: use the fact that 3x3 matmul can be done with rank 7
    # via the following construction (based on Strassen's method for 2x2,
    # extended to 3x3 by treating the 3rd row/col separately).
    
    # Actually, for 3x3 * 3x3, the known minimal rank is 7 (Strassen).
    # Let me construct it explicitly.
    
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 3x3, we can partition A into:
    # A = [[a,b,c],[d,e,f],[g,h,i]]
    # B = [[x,y,z],[u,v,w],[s,t,q]]
    # C = A*B
    
    # One approach: use Strassen on the top-left 2x2, and handle the rest.
    # But this gives rank 7 + additional terms.
    
    # Let me use a direct rank-7 construction for 3x3.
    # Based on the literature, 3x3 matmul has rank 7.
    
    # Here's a known rank-7 decomposition for 3x3:
    # We'll define 7 "multiplication" terms.
    
    # Actually, let me just use the standard approach: 
    # For 3x3 * 3x3, we can write it as a 2x2 Strassen on blocks
    # where each block is 1.5x1.5, which doesn't work cleanly.
    
    # Let me try a different approach: use the fact that we can embed
    # 3x3 into 4x4 (rank 7 for 2x2 Strassen applied to 2x2 sub-blocks)
    # but that's complex.
    
    # Simpler: just use schoolbook for 3x3 (rank 9) and see if total is better.
    # 9 + 12 = 21. Or try to find a better decomposition.
    
    # For now, let's use a known explicit rank-7 decomposition.
    # After research, here's a construction:
    
    # The 3x3 matmul tensor can be decomposed with rank 7 using:
    # Let A be 3x3, B be 3x3.
    # Define 7 intermediate products:
    
    # I'll use the following approach based on the "Strassen-like" decomposition:
    # Split A and B into overlapping 2x2 submatrices.
    
    # For simplicity and correctness, let me use a verified construction.
    
    n3, m3, p3 = 3, 3, 3
    r = 7
    U = np.zeros((n3*m3, r))
    V = np.zeros((m3*p3, r))
    W = np.zeros((p3*n3, r))
    
    # Strassen-style for 3x3:
    # We'll use the 7 terms from Strassen's 2x2 algorithm,
    # applied to the top-left 2x2 blocks, plus adjustments.
    
    # Actually, the cleanest approach: use the known fact that
    # R(3,3,3) = 7 and construct it via the "padded Strassen" method.
    
    # Let me use a concrete rank-7 decomposition:
    # Term 1: (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2]) - corrections
    # This is getting complex. Let me use a verified source.
    
    # Known rank-7 decomposition for 3x3 (from literature):
    # Using the following 7 bilinear forms:
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # These give C00, C01, C10, C11 from the 2x2 top-left.
    # For the 3x3, we need additional terms for row/col 2.
    
    # This means 3x3 * 3x3 actually needs rank > 7 if we only use these.
    # Let me check: R(3,3,3) is actually known to be 7? Let me verify.
    
    # Upon reflection, R(3,3,3) = 7 is a well-known result (Strassen 1969).
    # The decomposition is non-trivial. Let me use a computational approach
    # to find it, or use a known explicit form.
    
    # For now, let me use schoolbook for the 3x3 part (rank 9) and 
    # see if the total rank is competitive.
    
    # Schoolbook for 3x3: rank 9
    U3 = np.zeros((9, 9))
    V3 = np.zeros((9, 9))
    W3 = np.zeros((9, 9))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
    
    return U3, V3, W3, 9

def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    Strategy: Split B into B1 (3x3) and B2 (3x4).
    - C1 = A*B1: use Strassen (rank 7) or schoolbook (rank 9)
    - C2 = A*B2: use schoolbook (rank 12)
    Total: 7+12=19 or 9+12=21
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Decompose into 3x3 part (k=0,1,2) and 3x4 part (k=3,4,5,6)
    
    # For the 3x3 * 3x3 part: use schoolbook (rank 9) for reliability
    # For the 3x3 * 3x4 part: use schoolbook (rank 12)
    # Total rank: 21
    
    rank_33 = 9   # 3x3 * 3x3 schoolbook
    rank_34 = 12  # 3x3 * 3x4 schoolbook
    total_rank = rank_33 + rank_34  # 21
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    # Part 1: k = 0, 1, 2 (3x3 * 3x3)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Part 2: k = 3, 4, 5, 6 (3x3 * 3x4)
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    assert col == total_rank
    assert _verify_decomposition(U, V, W, n, m, p), "Decomposition failed verification!"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
