"""
Rank-44 exact decomposition for 3x4x4 matrix multiplication.
Uses Strassen (rank 7) for 2x2x2 blocks + schoolbook (rank 4) for 1x2x2 blocks.
"""
import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    U, V, W = _construct_rank44()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _construct_rank44():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        """Add 7 Strassen terms for a 2x2x2 block.
        A is rows rA..rA+1, cols cA..cA+1
        B is rows rB..rB+1, cols cB..cB+1
        Output C is rows rA..rA+1, cols cB..cB+1
        """
        nonlocal col
        # Strassen 7 terms for 2x2x2
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # M1 = (A11+A22)(B11+B22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        
        # M3 = A11(B21-B11)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M4 = A22(B12-B22)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        """Add 4 schoolbook terms for a 1x2x2 block.
        A is row rA, cols cA..cA+1
        B is rows rB..rB+1, cols cB..cB+1
        Output C is row rA, cols cB..cB+1
        """
        nonlocal col
        # 4 terms for 1x2 * 2x2
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1

    # 4 Strassen blocks (2x2x2) + 4 schoolbook blocks (1x2x2)
    # Rows 0-1 of A with cols 0-1,3 and B rows 0-1,3 with cols 0-1,3
    add_strassen(0, 0, 0, 0)      # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2, 0)      # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen(0, 0, 0, 2)      # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 2, 2)      # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    # Row 2 of A
    add_schoolbook_1x2x2(2, 0, 0, 0)  # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_schoolbook_1x2x2(2, 2, 2, 0)  # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    add_schoolbook_1x2x2(2, 0, 0, 2)  # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    add_schoolbook_1x2x2(2, 2, 2, 2)  # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)
