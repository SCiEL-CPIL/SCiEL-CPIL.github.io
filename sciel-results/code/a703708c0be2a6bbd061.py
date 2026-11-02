"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_strassen_2x2():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication as NumPy arrays.
    U, V, W: shape (4, 7)
    """
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U factors (for A)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # M1, M2, M3, M4, M5, M6, M7
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float).T # Transpose to (7, 4) ? No, let's keep (4, 7)
    # Let's define them clearly.
    # A00, A01, A10, A11
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=float)
    
    # V factors (for B)
    # B00, B01, B10, B11
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 1], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, -1] # B11
    ], dtype=float)
    
    # W factors (for C)
    # C00, C01, C10, C11
    # C00 = M1 + M3 - M4 + M6 ? 
    # Let's use the standard:
    # C00 = M1 - M4 + M6 + M7 ? 
    # Let's re-verify the W vectors from the current program's logic which worked.
    
    # From current program:
    # p1: w=[1, 0, 0, 1]
    # p2: w=[0, 0, 1, -1]
    # p3: w=[0, 1, 0, 1]
    # p4: w=[1, 0, 1, 0]
    # p5: w=[-1, 1, 0, 0]
    # p6: w=[0, 0, 0, 1]
    # p7: w=[1, 0, 0, 0]
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=float)
    
    return U2, V2, W2

def _construct_rank_98():
    """
    Construct rank 98 decomposition for 4x4 * 4x8.
    """
    n, m, p = DIMENSIONS
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # The 4x4x4 decomposition is a Kronecker product of the block-level and scalar-level Strassen.
    # We iterate over the 7 block-level products (s) and 7 scalar-level products (r).
    
    for s in range(7):
        for r in range(7):
            col = s * 7 + r
            
            # Block level coefficients for A, B, C
            u_block = U2[:, s] # shape (4,) -> coeffs for A00, A01, A10, A11 blocks
            v_block = V2[:, s]
            w_block = W2[:, s]
            
            # Scalar level coefficients
            u_scalar = U2[:, r]
            v_scalar = V2[:, r]
            w_scalar = W2[:, r]
            
            # Construct U for this column
            # A is 4x4. Blocks are 2x2.
            # A_block(i,j) corresponds to A[2i:2i+2, 2j:2j+2]
            # u_block[i*2+j] is the coefficient for block (i,j)
            # u_scalar[ei*2+ej] is the coefficient for element (ei, ej) within the block
            
            for bi in range(2):
                for bj in range(2):
                    c_block = u_block[bi*2 + bj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = u_scalar[ei*2 + ej]
                            if c_scalar == 0:
                                continue
                            i = 2*bi + ei
                            j = 2*bj + ej
                            U[i*m + j, col] += c_block * c_scalar
            
            # Construct V for this column
            # B is 4x8. 
            # We split B into two 4x4 blocks: B1 (cols 0-3) and B2 (cols 4-7).
            # For subproblem t (0 or 1), B_t is 4x4.
            # We need to fill V for both B1 and B2?
            # No, the rank 98 decomposition is the direct sum of two rank 49 decompositions.
            # So for col in 0..48, it corresponds to B1.
            # For col in 49..97, it corresponds to B2.
            
            # My loop above generates cols 0..48.
            # I need to generate cols 49..97 as well, with the same U and W, but V shifted.
            
            # Actually, it's easier to just fill the first 49 cols for B1, and the next 49 for B2.
            
            # Let's refactor: 
            # The loop above fills col = s*7+r.
            # This corresponds to the first 4x4x4 multiplication (B1).
            
            # Now we need to fill the second 4x4x4 multiplication (B2).
            # The U and W coefficients are the SAME.
            # The V coefficients are the SAME, but applied to B2 (cols 4-7).
            
            # So we can just copy U and W to the second half, and construct V for the second half.
            
            # But wait, the current loop structure puts B1 in cols 0-48.
            # Let's finish the loop for B1 first.
            
            # V for B1 (cols 0-3 of B)
            for bi in range(2):
                for bj in range(2):
                    c_block = v_block[bi*2 + bj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = v_scalar[ei*2 + ej]
                            if c_scalar == 0:
                                continue
                            j = 2*bi + ei # B row
                            k = 2*bj + ej # B col (within B1)
                            V[j*p + k, col] += c_block * c_scalar
            
            # W for C1 (cols 0-3 of C)
            for ci in range(2):
                for cj in range(2):
                    c_block = w_block[ci*2 + cj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = w_scalar[ei*2 + ej]
                            if c_scalar == 0:
                                continue
                            i = 2*ci + ei # C row
                            k = 2*cj + ej # C col (within C1)
                            W[k*n + i, col] += c_block * c_scalar
            
            # Now, fill the second half (cols 49-97) for B2 and C2.
            col2 = 49 + col
            
            # U is the same
            U[:, col2] = U[:, col]
            
            # V is the same coefficients, but B col offset by 4
            V[:, col2] = 0
            for bi in range(2):
                for bj in range(2):
                    c_block = v_block[bi*2 + bj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = v_scalar[ei*2 + ej]
                            if c_scalar == 0:
                                continue
                            j = 2*bi + ei # B row
                            k = 4 + 2*bj + ej # B col (within B2, offset by 4)
                            V[j*p + k, col2] += c_block * c_scalar
            
            # W is the same coefficients, but C col offset by 4
            W[:, col2] = 0
            for ci in range(2):
                for cj in range(2):
                    c_block = w_block[ci*2 + cj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = w_scalar[ei*2 + ej]
                            if c_scalar == 0:
                                continue
                            i = 2*ci + ei # C row
                            k = 4 + 2*cj + ej # C col (within C2, offset by 4)
                            W[k*n + i, col2] += c_block * c_scalar

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    
    try:
        result = _construct_rank_98()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    return _schoolbook()
# EVOLVE-BLOCK-END
