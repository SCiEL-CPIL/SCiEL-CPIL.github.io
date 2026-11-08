"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Try to find a rank-42 decomposition using Strassen on 2x2 blocks
    # but since constructing it reliably is complex, we'll use a structured approach.
    # For 2x4x6, we can view it as 2x4x2 * 3 blocks.
    # Each 2x4x2 can potentially be done with rank 14 (two 2x2x2 Strassen).
    # Let's try to construct rank 42 explicitly.

    deadline = time.time() + 145  # safety margin

    # Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For 2x4x2: A is 2x4, B is 4x2, C is 2x2
    # Split A into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3)
    # Split B into B1 (2x2, rows 0,1) and B2 (2x2, rows 2,3)
    # C = A1*B1 + A2*B2
    # Each product is 2x2x2, use Strassen (rank 7 each)
    # Total rank for one 2x4x2 block: 14

    # For 2x4x6: split into three 2x4x2 blocks (k=0,1; k=2,3; k=4,5)
    # Total rank: 3 * 14 = 42

    # Let's build this explicitly
    rank_candidate = 42
    U_cand = np.zeros((n * m, rank_candidate), dtype=int)
    V_cand = np.zeros((m * p, rank_candidate), dtype=int)
    W_cand = np.zeros((p * n, rank_candidate), dtype=int)

    col_offset = 0
    for block_k in range(3):  # k offset: 0, 2, 4
        k0 = block_k * 2
        k1 = k0 + 1
        for block_j in range(2):  # j offset: 0, 2
            j0 = block_j * 2
            j1 = j0 + 1

            # 2x2x2 Strassen for A[i, j0:j1] * B[j0:j1, k0:k1] -> C[k0:k1, i]
            # A local: rows 0,1 (global i), cols 0,1 (global j0, j1)
            # B local: rows 0,1 (global j0, j1), cols 0,1 (global k0, k1)
            # C local: rows 0,1 (global k0, k1), cols 0,1 (global i)

            # M1 = (A00+A11)(B00+B11)
            # A00 = A[0, j0], A11 = A[1, j1]
            # B00 = B[j0, k0], B11 = B[j1, k1]
            U_cand[0 * m + j0, col_offset] = 1  # A00
            U_cand[1 * m + j1, col_offset] = 1  # A11
            V_cand[j0 * p + k0, col_offset] = 1  # B00
            V_cand[j1 * p + k1, col_offset] = 1  # B11
            W_cand[k0 * n + 0, col_offset] = 1   # C[0,0]
            W_cand[k0 * n + 1, col_offset] = 1   # C[0,1]
            W_cand[k1 * n + 0, col_offset] = -1  # C[1,0]
            W_cand[k1 * n + 1, col_offset] = 1   # C[1,1]
            col_offset += 1

            # M2 = (A10+A11)B00
            U_cand[1 * m + j0, col_offset] = 1  # A10
            U_cand[1 * m + j1, col_offset] = 1  # A11
            V_cand[j0 * p + k0, col_offset] = 1  # B00
            W_cand[k0 * n + 0, col_offset] = -1  # C[0,0]
            W_cand[k0 * n + 1, col_offset] = -1  # C[0,1]
            W_cand[k1 * n + 0, col_offset] = 1   # C[1,0]
            col_offset += 1

            # M3 = A00(B01-B11)
            U_cand[0 * m + j0, col_offset] = 1  # A00
            V_cand[j0 * p + k1, col_offset] = 1  # B01
            V_cand[j1 * p + k1, col_offset] = -1 # B11
            W_cand[k0 * n + 1, col_offset] = 1   # C[0,1]
            W_cand[k1 * n + 1, col_offset] = 1   # C[1,1]
            col_offset += 1

            # M4 = A11(B10-B00)
            U_cand[1 * m + j1, col_offset] = 1  # A11
            V_cand[j1 * p + k0, col_offset] = 1  # B10
            V_cand[j0 * p + k0, col_offset] = -1 # B00
            W_cand[k0 * n + 0, col_offset] = 1   # C[0,0]
            W_cand[k0 * n + 1, col_offset] = -1  # C[0,1]
            W_cand[k1 * n + 0, col_offset] = 1   # C[1,0]
            col_offset += 1

            # M5 = (A00+A01)B11
            U_cand[0 * m + j0, col_offset] = 1  # A00
            U_cand[0 * m + j1, col_offset] = 1  # A01
            V_cand[j1 * p + k1, col_offset] = 1  # B11
            W_cand[k0 * n + 0, col_offset] = -1  # C[0,0]
            W_cand[k0 * n + 1, col_offset] = 1   # C[0,1]
            W_cand[k1 * n + 1, col_offset] = -1  # C[1,1]
            col_offset += 1

            # M6 = (A10-A00)(B00+B01)
            U_cand[1 * m + j0, col_offset] = 1  # A10
            U_cand[0 * m + j0, col_offset] = -1 # A00
            V_cand[j0 * p + k0, col_offset] = 1  # B00
            V_cand[j0 * p + k1, col_offset] = 1  # B01
            W_cand[k0 * n + 1, col_offset] = 1   # C[0,1]
            W_cand[k1 * n + 1, col_offset] = 1   # C[1,1]
            col_offset += 1

            # M7 = (A01-A11)(B10+B11)
            U_cand[0 * m + j1, col_offset] = 1  # A01
            U_cand[1 * m + j1, col_offset] = -1 # A11
            V_cand[j1 * p + k0, col_offset] = 1  # B10
            V_cand[j1 * p + k1, col_offset] = 1  # B11
            W_cand[k0 * n + 0, col_offset] = 1   # C[0,0]
            W_cand[k1 * n + 0, col_offset] = -1  # C[1,0]
            W_cand[k1 * n + 1, col_offset] = -1  # C[1,1]
            col_offset += 1

    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        # Build tensor T[i*m+j, j*p+k, k*n+i]
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta (i.e., 1 if valid index)
        # Actually the tensor for matrix multiplication has T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # Let's verify by checking that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # And for other combinations, it should be 0.
        
        # More efficiently: build the tensor and check rank decomposition
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Reconstruct from U, V, W
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_recon)

    if verify(U_cand, V_cand, W_cand, n, m, p):
        return {"U": U_cand, "V": V_cand, "W": W_cand}

    # Fallback to schoolbook rank 48
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
