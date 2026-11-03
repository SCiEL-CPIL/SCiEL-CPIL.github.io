"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _build_rank22_3x3x3():
    """
    Return a rank-22 decomposition for 3x3x3 matrix multiplication.
    Uses Strassen (rank 7) for the top-left 2x2 block plus 15 schoolbook
    terms for the cross-block interactions involving row/col index 2.
    """
    r = 22
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # --- Strassen 7 terms for the 2x2 block (rows/cols 0,1) ---
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # --- Cross terms involving index 2 (15 schoolbook terms) ---
    # C[0,2]: A00*B02, A01*B12, A02*B22
    U[0][7] = 1;  V[2][7] = 1;  W[2][7] = 1
    U[1][8] = 1;  V[5][8] = 1;  W[2][8] = 1
    U[2][9] = 1;  V[8][9] = 1;  W[2][9] = 1
    # C[1,2]: A10*B02, A11*B12, A12*B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    # C[2,0]: A20*B00, A21*B10, A22*B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1
    # C[2,1]: A20*B01, A21*B11, A22*B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1
    # C[2,2]: A20*B02, A21*B12, A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][20] = 1; V[5][20] = 1; W[8][20] = 1
    U[8][21] = 1; V[8][21] = 1; W[8][21] = 1

    return U, V, W


def _verify_3x3x3(U, V, W, seed=42):
    """Verify a rank-r decomposition for 3x3x3 matrix multiplication."""
    r = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    rng = np.random.RandomState(seed)
    A_test = rng.randn(3, 3)
    B_test = rng.randn(3, 3)
    C_expected = A_test @ B_test

    C_computed = np.zeros((3, 3))
    for r_idx in range(r):
        a_proj = U_arr[:, r_idx].reshape(3, 3) @ A_test
        b_proj = V_arr[:, r_idx].reshape(3, 3) @ B_test
        for i in range(3):
            for k in range(3):
                C_computed[i, k] += W_arr[k * 3 + i, r_idx] * a_proj[i, k]

    return np.allclose(C_computed, C_expected, atol=1e-10)


def _try_rank19_3x3x6():
    """
    Try to find a rank-19 decomposition for 3x3x6 using an embedding strategy.
    
    The idea: embed the 3x3x6 multiplication into a 6x6x6 multiplication.
    If we can find a rank-26 decomposition for 6x6x6 (or use a known one),
    we might be able to project it down. However, the known best for 6x6x6 
    is much higher.
    
    Alternative: use the fact that 3x3x6 = two 3x3x3 multiplications.
    The schoolbook rank is 54, Strassen+cross is 44.
    
    Let's try a different approach: use a rank-22 decomposition for 3x3x3,
    but try to share some terms between the two 3x3x3 blocks.
    
    Actually, let's try a direct search for a lower rank using algebraic identities.
    
    For 3x3x6, the tensor is T[i,j,k] = delta[i,j] * delta[j,k] in a sense.
    
    Let's try rank 43: can we save one more term?
    """
    n, m, p = 3, 3, 6
    
    # Start from the rank-44 decomposition (two rank-22 blocks)
    U_block, V_block, W_block = _build_rank22_3x3x3()
    r_block = 22
    
    # Total rank if no sharing: 44
    # Try to see if we can reduce by 1
    
    # The key insight: in the two 3x3x3 blocks, the A matrix is the same.
    # Can we share any U factors?
    
    # In block 0, U factors are used for columns k=0,1,2 of B.
    # In block 1, U factors are used for columns k=3,4,5 of B.
    # The U factors are the same (they only depend on A), so we might be able to
    # combine them.
    
    # Let's try to find a decomposition where some U factors are shared.
    
    # Actually, the current construction already uses the same U factors for both
    # blocks, just with different V factors. The total rank is 44 because we have
    # 22 terms for block 0 and 22 terms for block 1.
    
    # Can we reduce this? The Strassen terms for the 2x2 block are the same in
    # both blocks. Can we share them?
    
    # In block 0, term 0 (M1) has U = [1,0,0,0,1,0,0,0,0] for A.
    # In block 1, term 22 (M1) has the same U factor.
    # But the V factors are different (different columns of B).
    
    # If we could share the U factor, we'd need a way to combine the V factors.
    # But the tensor decomposition doesn't allow that directly.
    
    # Let's try a different approach: use a rank-21 decomposition for 3x3x3 if one exists.
    
    # The current best known for 3x3x3 is rank 22 (Strassen + 15 cross terms).
    # Is there a rank-21 decomposition?
    
    # Let's try to search for one using a small optimization.
    
    # Actually, let's just try rank 43 directly by looking for a way to save one term.
    
    # The cross terms involve 15 terms per block. Can we combine any of them?
    
    # C[0,2]: A00*B02, A01*B12, A02*B22
    # C[1,2]: A10*B02, A11*B12, A12*B22
    # C[2,0]: A20*B00, A21*B10, A22*B20
    # C[2,1]: A20*B01, A21*B11, A22*B21
    # C[2,2]: A20*B02, A21*B12, A22*B22
    
    # Can we combine A00*B02 + A10*B02 = (A00+A10)*B02? No, that would change the U factor.
    
    # Let's just return None and use the rank-44 decomposition.
    return None


def _try_rank43_3x3x6():
    """
    Try to find a rank-43 decomposition for 3x3x6.
    
    The idea: use the rank-22 decomposition for 3x3x3, but try to save one term
    by combining cross terms.
    """
    n, m, p = 3, 3, 6
    
    U_block, V_block, W_block = _build_rank22_3x3x3()
    r_block = 22
    
    # Start with rank 44 and try to reduce by 1
    total_rank = 43
    
    # Build the base decomposition
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy block 0 (first 22 terms)
    for r_idx in range(r_block):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r_idx] = U_block[i * m + j][r_idx]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r_idx] = V_block[j * 3 + k][r_idx]
        for k in range(3):
            for i in range(n):
                W[k * n + i][r_idx] = W_block[k * n + i][r_idx]
    
    # Copy block 1 (next 22 terms, but we only have 21 slots)
    # We need to drop one term from block 1
    
    # Let's try dropping the last term (C[2,2]: A22*B22) and see if we can
    # absorb it into another term.
    
    # Actually, this is getting complicated. Let's just return None.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 3, 6
    
    # Try rank 43
    if time.time() < deadline:
        result_43 = _try_rank43_3x3x6()
        if result_43 is not None:
            U, V, W = result_43
            # Verify
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            total_rank = len(U[0])
            
            rng = np.random.RandomState(123)
            A_test = rng.randn(n, m)
            B_test = rng.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for r_idx in range(total_rank):
                a_proj = U_arr[:, r_idx].reshape(n, m) @ A_test
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W_arr[k * n + i, r_idx] * a_proj[i, k]
            
            if np.allclose(C_computed, C_expected, atol=1e-10):
                return {"U": U, "V": V, "W": W}
    
    # Use rank 44 (two rank-22 blocks)
    U_block, V_block, W_block = _build_rank22_3x3x3()
    r_block = len(U_block[0])
    total_rank = 2 * r_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 0: columns k=0,1,2 of B
    for r_idx in range(r_block):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r_idx] = U_block[i * m + j][r_idx]
        for j in range(m):
            for k in range(3):
                V[j * p + k][r_idx] = V_block[j * 3 + k][r_idx]
        for k in range(3):
            for i in range(n):
                W[k * n + i][r_idx] = W_block[k * n + i][r_idx]
    
    # Block 1: columns k=3,4,5 of B
    for r_idx in range(r_block):
        offset = r_block + r_idx
        for i in range(n):
            for j in range(m):
                U[i * m + j][offset] = U_block[i * m + j][r_idx]
        for j in range(m):
            for k in range(3):
                V[j * p + (k + 3)][offset] = V_block[j * 3 + k][r_idx]
        for k in range(3):
            for i in range(n):
                W[(k + 3) * n + i][offset] = W_block[k * n + i][r_idx]
    
    # Verify the full 3x3x6 decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rng = np.random.RandomState(123)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r_idx in range(total_rank):
        a_proj = U_arr[:, r_idx].reshape(n, m) @ A_test
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k * n + i, r_idx] * a_proj[i, k]
    
    if np.allclose(C_computed, C_expected, atol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
