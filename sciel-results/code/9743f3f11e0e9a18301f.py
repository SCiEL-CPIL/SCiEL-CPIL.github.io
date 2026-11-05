"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    # Construct rank-44 decomposition using Strassen for 2x2x2 blocks
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        """Add 7 terms for 2x2x2 Strassen decomposition."""
        nonlocal col
        # Global indices
        # A: (rA+i, cA+j) -> (rA+i)*m + (cA+j)
        # B: (rB+j, cB+k) -> (rB+j)*p + (cB+k)
        # C: (cB+k, rA+i) -> (cB+k)*n + (rA+i)
        
        # M1 = (X11+X22)(Y11+Y22) -> Z11 + Z22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M2 = (X21+X22)Y11 -> Z21 - Z22
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        
        # M3 = X11(Y12-Y22) -> Z12 + Z22
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M4 = X22(Y21-Y11) -> Z11 + Z21
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        
        # M5 = (X11+X12)Y22 -> Z12 - Z11
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        
        # M6 = (X21-X11)(Y11+Y12) -> Z22
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M7 = (X12-X22)(Y21+Y22) -> Z11
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_rank4(rA, cA, rB, cB):
        """Add 4 terms for 1x2x2 schoolbook decomposition."""
        nonlocal col
        # A is 1x2: (rA, cA), (rA, cA+1)
        # B is 2x2: (rB+j, cB+k)
        # C is 1x2: (cB+k, rA)
        
        # C0 = A0*B00 + A1*B10
        U[rA*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + rA, col] = 1
        col += 1
        
        U[rA*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + rA, col] = 1
        col += 1
        
        # C1 = A0*B01 + A1*B11
        U[rA*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + rA, col] = 1
        col += 1
        
        U[rA*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + rA, col] = 1
        col += 1

    # Apply Strassen to 2x2x2 blocks
    # A is 3x4, B is 4x4, C is 3x4
    # Split A into 2x2 blocks: rows 0-1 and 2, cols 0-1 and 2-3
    # Split B into 2x2 blocks: rows 0-1 and 2-3, cols 0-1 and 2-3
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0)
    
    # Block (0,1): A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 0, 2)
    
    # Block (1,0): A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 2, 0)
    
    # Block (1,1): A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2, 2)
    
    # Row 2 of A: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    add_rank4(2, 0, 0, 0)
    
    # Row 2 of A: A[2, 2:4] * B[0:2, 2:4] -> C[2, 2:4]
    add_rank4(2, 2, 0, 2)
    
    # Row 2 of A: A[2, 0:2] * B[2:4, 0:2] -> C[2, 0:2]
    add_rank4(2, 0, 2, 0)
    
    # Row 2 of A: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    add_rank4(2, 2, 2, 2)
    
    # Verify the decomposition
    if verify_decomposition(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    U_sb, V_sb, W_sb = schoolbook()
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
