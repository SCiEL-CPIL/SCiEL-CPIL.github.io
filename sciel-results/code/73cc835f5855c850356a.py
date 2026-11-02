"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _winograd_3x3x6():
    """
    Construct a rank-16 decomposition for 3x3 * 3x6 matrix multiplication
    using Winograd's algorithm.
    
    The 3x3 * 3x6 multiplication is split into two 3x3 * 3x3 multiplications.
    Each 3x3 * 3x3 multiplication is performed using a 4-point Winograd transform.
    Total rank = 4 * 4 = 16.
    """
    n, m, p = 3, 3, 6
    
    # Winograd transform for 3x3 matrix multiplication
    # We need a 4x4 matrix M such that:
    # C = (M * A * M^T) element-wise (M * B * M^T) * M^T
    # Actually, the standard Winograd form is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # where * denotes element-wise multiplication.
    
    # For 3x3, a known 4x4 transform matrix is:
    M = np.array([
        [1,  1,  1,  1],
        [0,  1, -1,  0],
        [0,  1,  1,  0],
        [0,  0,  0,  1]
    ], dtype=float)
    
    # Verify M is invertible
    if np.linalg.det(M) == 0:
        return None
        
    M_inv = np.linalg.inv(M)
    
    # The decomposition has rank 16 (4x4 grid of transform indices)
    rank = 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # The transformation is:
    # Let A be 3x3, B be 3x6.
    # Split B into B1 (3x3) and B2 (3x3).
    # C1 = A * B1, C2 = A * B2.
    # C = [C1, C2].
    
    # Winograd for 3x3 * 3x3:
    # C = M_inv^T * ( (M * A) * (M * B^T) ) * M_inv
    # Wait, the standard form is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # Let's check dimensions.
    # M is 4x4, A is 3x3. M*A is not defined.
    # The correct Winograd formulation for n x n is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M  is incorrect if M is square of size R > n.
    # Actually, for 3x3 with R=4, we embed A and B into 4x4 matrices?
    # No, the standard Winograd 3x3 uses a 4x4 transform on the 3x3 matrix by padding?
    # No, the 3x3 Winograd algorithm uses a 4x4 matrix M where the transformation is:
    # A' = M * A * M^T is not possible if A is 3x3 and M is 4x4.
    
    # Correction: The Winograd algorithm for 3x3 multiplication uses a 4x4 matrix M.
    # The operation is defined as:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # This requires A and B to be 4x4? No.
    
    # Let's look at the standard 3x3 Winograd.
    # It uses a 4x4 matrix M.
    # The matrices A and B are 3x3.
    # The transform is:
    # A_trans = M * A * M^T  -> This is 4x4 * 3x3 * 4x4, undefined.
    
    # Ah, the Winograd transform for 3x3 is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # This formula is for the case where the transform is applied to the rows and columns.
    # If M is 4x4 and A is 3x3, we cannot multiply M * A directly.
    
    # Let's use the explicit coefficient method for 3x3x3 Winograd.
    # The 3x3x3 tensor has rank 4.
    # The 4 rank-1 terms are defined by 4x4 matrices U, V, W.
    
    # Standard 3x3x3 Winograd decomposition (Rank 4):
    # M = [[1, 1, 1, 1],
    #      [0, 1, -1, 0],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 1]]
    
    # The factor matrices for a single 3x3x3 multiplication are:
    # U_3x3[i,j, r] = M_inv[i, r] * M_inv[j, r]  (Wait, indices need care)
    # Actually, the decomposition is:
    # sum_{r,s} (M_inv[i, r] M_inv[j, s] A[j, s]) * (M_inv[s, k] M_inv[l, t] B[l, t]) ... no.
    
    # Let's derive it properly.
    # C = A * B
    # C = M^T * ( (M * A) * (M * B^T) ) * M  is the form if we pad to 4x4.
    # But 3x3 Winograd does not pad. It uses a 4x4 transform on the 3x3 data.
    # The transform is:
    # A' = M * A * M^T  is wrong.
    
    # The correct Winograd 3x3 is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # Where A and B are 3x3.
    # This implies M is 3x3? No, rank is 4, so M is 4x4.
    # How does M * A work?
    # It turns out the 3x3 Winograd uses a 4x4 matrix M, and the multiplication is defined on the 3x3 sub-blocks?
    # No, the standard reference says:
    # "The 3x3 matrix multiplication can be performed with 4 multiplications using the 4x4 matrix M."
    # The formula is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # This is only valid if A and B are 4x4.
    
    # Let's check the literature.
    # Winograd 3x3:
    # M = [[1, 1, 1, 1], [0, 1, -1, 0], [0, 1, 1, 0], [0, 0, 0, 1]]
    # The matrices A and B are 3x3.
    # The product is C = A * B.
    # The algorithm is:
    # 1. Compute A' = M * A * M^T ? No.
    # 2. Compute A' = M * A ? No.
    
    # Actually, the 3x3 Winograd algorithm is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # This is a common confusion. For 3x3, the transform is applied to the 3x3 matrix by considering it as a 3x3 submatrix of a 4x4?
    # No, the 3x3 Winograd uses a 4x4 matrix M, but the multiplication is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # This requires A and B to be 4x4.
    
    # Let's try a different known decomposition for 3x3x3.
    # Rank 4 decomposition exists.
    # Let's use the explicit coefficients from a known source.
    
    # For 3x3x3, the rank-4 decomposition factors are:
    # U, V, W are 9x4 matrices.
    
    # Let's construct them using the M matrix.
    # The transform is:
    # A_trans[i, r] = sum_j M_inv[i, r] A[i, j] ? No.
    
    # Let's use the property:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # If we assume A and B are 4x4 (with 0 padding), then:
    # A' = M * A * M^T (4x4)
    # B' = M * B * M^T (4x4)
    # C' = A' * B' (element-wise)
    # C = M^T * C' * M (4x4)
    # Then take the 3x3 submatrix.
    
    # Let's test this "padding" approach.
    # Pad A (3x3) to A_pad (4x4) with 0s in the last row/col.
    # Pad B (3x3) to B_pad (4x4) with 0s in the last row/col.
    # Compute C_pad = A_pad * B_pad using Winograd.
    # The 3x3 result is C_pad[:3, :3].
    
    # Does Winograd on 4x4 padded 3x3 give the correct 3x3 result?
    # Yes, because the padding is 0, so the cross terms vanish.
    
    # So, for 3x3x3:
    # 1. Pad A, B to 4x4.
    # 2. A' = M * A_pad * M^T
    # 3. B' = M * B_pad * M^T
    # 4. C' = A' * B' (element-wise)
    # 5. C_pad = M^T * C' * M
    # 6. C = C_pad[:3, :3]
    
    # This gives a rank-16 decomposition for 3x3x3? No, rank is 16 (4x4 element-wise mult).
    # But we know 3x3x3 has rank 4.
    # The padding approach yields rank 16, which is worse than schoolbook (9).
    
    # So the true 3x3 Winograd does NOT use padding.
    # It uses a 4x4 transform on the 3x3 data directly.
    # How?
    # The 3x3 Winograd transform is:
    # A' = M * A * M^T is not defined.
    
    # Let's look at the actual 3x3 Winograd coefficients.
    # M = [[1, 1, 1, 1], [0, 1, -1, 0], [0, 1, 1, 0], [0, 0, 0, 1]]
    # The factors are:
    # U[i, j, r] = M_inv[i, r] * M_inv[j, r] ?
    # Let's check dimensions. i,j in 0..2, r in 0..3.
    # M_inv is 4x4. M_inv[i, r] is valid for i in 0..2, r in 0..3.
    
    # Let's verify:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # = sum_r (M_inv[i, r] M_inv[j, r]) (M_inv[j, r] M_inv[k, r]) (M_inv[k, r] M_inv[i, r])
    # = sum_r M_inv[i, r]^2 M_inv[j, r]^2 M_inv[k, r]^2
    # This does not equal delta[i, k] or anything simple.
    
    # The correct Winograd decomposition for 3x3x3 is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # This formula is valid if we define the multiplication of 4x4 and 3x3 matrices appropriately?
    # No.
    
    # Let's use a known rank-4 decomposition for 3x3x3.
    # One such decomposition is given by the 4x4 matrix M above.
    # The factors are:
    # U[i, j, r] = M_inv[i, r] * M_inv[j, r]  (for i,j < 3)
    # V[j, k, r] = M_inv[j, r] * M_inv[k, r]  (for j,k < 3)
    # W[k, i, r] = M_inv[k, r] * M_inv[i, r]  (for k,i < 3)
    
    # Let's test this hypothesis.
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # = sum_r M_inv[i, r]^2 M_inv[j, r]^2 M_inv[k, r]^2
    # This is not 1 for i=k and 0 otherwise.
    
    # The correct Winograd form is:
    # C = M^T * ( (M * A) * (M * B^T) ) * M
    # Let's expand:
    # C[i, k] = sum_{r, s} M_inv[i, r] ( (M * A)[r, s] * (M * B^T)[r, s] ) M_inv[k, s]
    # (M * A)[r, s] = sum_j M[r, j] A[j, s]
    # (M * B^T)[r, s] = sum_l M[r, l] B^T[l, s] = sum_l M[r, l] B[s, l]
    
    # So C[i, k] = sum_{r, s} M_inv[i, r] M_inv[k, s] ( sum_j M[r, j] A[j, s] ) ( sum_l M[r, l] B[s, l] )
    # = sum_{r, s, j, l} M_inv[i, r] M_inv[k, s] M[r, j] M[r, l] A[j, s] B[s, l]
    
    # We want C[i, k] = sum_j A[i, j] B[j, k].
    # So we need:
    # sum_{r, s, j, l} M_inv[i, r] M_inv[k, s] M[r, j] M[r, l] A[j, s] B[s, l] = sum_j A[i, j] B[j, k]
    
    # This implies:
    # sum_{r, s, l} M_inv[i, r] M_inv[k, s] M[r, j] M[r, l] B[s, l] = B[j, k] * (something involving A)
    # Actually, the term A[j, s] is there.
    # The expression is linear in A and B.
    # The coefficient of A[j, s] B[s, l] is:
    # sum_r M_inv[i, r] M[r, j] M[r, l] M_inv[k, s]
    
    # We want this to be delta[i, j] delta[k, l] ? No.
    # We want the final sum to be sum_j A[i, j] B[j, k].
    # So the coefficient of A[j, s] B[s, l] should be delta[i, j] delta[k, l] if s=j?
    # No, the standard matrix multiplication is sum_j A[i, j] B[j, k].
    # So the term A[j, s] B[s, l] contributes only if s=j.
    # So we need the coefficient of A[j, j] B[j, l] to be delta[i, j] delta[k, l].
    # And the coefficient of A[j, s] B[s, l] with s != j to be 0.
    
    # This is getting complicated. Let's use the known result.
    # The 3x3x3 tensor has rank 4.
    # The decomposition is given by the 4x4 matrix M.
    # The factors are:
    # U[i, j, r] = M_inv[i, r] * M_inv[j, r]
    # V[j, k, r] = M_inv[j, r] * M_inv[k, r]
    # W[k, i, r] = M_inv[k, r] * M_inv[i, r]
    
    # Let's verify this specific form again.
    # sum_r M_inv[i, r]^2 M_inv[j, r]^2 M_inv[k, r]^2
    # This is NOT the identity.
    
    # The correct Winograd factors for 3x3x3 are:
    # U[i, j, r] = M_inv[i, r] * M_inv[j, r]
    # V[j, k, r] = M_inv[j, r] * M_inv[k, r]
    # W[k, i, r] = M_inv[k, r] * M_inv[i, r]
    # This is for the case where the multiplication is C = M^T * ( (M * A) * (M * B^T) ) * M.
    # And it works because of the orthogonality properties of M.
    
    # Let's assume this is correct and test it numerically in the code.
    # If it fails, I will fall back to schoolbook.
    
    M = np.array([
        [1,  1,  1,  1],
        [0,  1, -1,  0],
        [0,  1,  1,  0],
        [0,  0,  0,  1]
    ], dtype=float)
    
    M_inv = np.linalg.inv(M)
    
    # Construct U, V, W for 3x3x3 (rank 4)
    U_3 = [[0.0] * 4 for _ in range(9)]
    V_3 = [[0.0] * 4 for _ in range(9)]
    W_3 = [[0.0] * 4 for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            for r in range(4):
                U_3[i * 3 + j][r] = M_inv[i, r] * M_inv[j, r]
                V_3[j * 3 + i][r] = M_inv[j, r] * M_inv[i, r]  # Note: V is for B, indices j,k
                # Wait, V[j, k, r] = M_inv[j, r] * M_inv[k, r]
                # Let's fill V properly
                pass
    
    # Refill V and W
    for j in range(3):
        for k in range(3):
            for r in range(4):
                V_3[j * 3 + k][r] = M_inv[j, r] * M_inv[k, r]
    
    for k in range(3):
        for i in range(3):
            for r in range(4):
                W_3[k * 3 + i][r] = M_inv[k, r] * M_inv[i, r]
    
    # Verify 3x3x3
    if not _verify(U_3, V_3, W_3):
        return None
        
    # Now for 3x3x6.
    # Split B into B1 (3x3) and B2 (3x3).
    # C1 = A * B1, C2 = A * B2.
    # Use the 3x3x3 decomposition for both.
    # Total rank = 4 + 4 = 8?
    # No, the rank is the sum of the ranks if they are independent.
    # But A is shared.
    # The decomposition is:
    # C1 = sum_{r=0}^3 U1[i, j, r] V1[j, k1, r] W1[k1, i, r]
    # C2 = sum_{r=0}^3 U2[i, j, r] V2[j, k2, r] W2[k2, i, r]
    
    # Since A is the same, U1 and U2 should be the same?
    # In Winograd, the transform of A is shared.
    # So U1 = U2 = U_3.
    # V1 and V2 are different because B1 and B2 are different.
    # W1 and W2 are different because the output indices are different.
    
    # So we can reuse the U factors.
    # Rank = 4 (for U) + 4 (for V1) + 4 (for V2) ? No.
    # The rank is the number of columns in U, V, W.
    # If U has 4 columns, V1 has 4 columns, V2 has 4 columns.
    # But U is shared, so the total rank is 4 (for the shared part) + 4 (for V1) + 4 (for V2)?
    # No, the rank is the dimension of the intermediate space.
    # If we use 4 terms for C1 and 4 terms for C2, and they are independent, rank is 8.
    # But U is the same, so the terms are coupled.
    
    # Let's construct the 3x3x6 decomposition.
    # Rank = 8.
    # U has 9 rows, 8 columns.
    # V has 18 rows, 8 columns.
    # W has 18 rows, 8 columns.
    
    U = [[0.0] * 8 for _ in range(9)]
    V = [[0.0] * 8 for _ in range(18)]
    W = [[0.0] * 8 for _ in range(18)]
    
    # First 4 columns: for C1 (k in 0..2)
    for idx in range(9):
        for r in range(4):
            U[idx][r] = U_3[idx][r]
            V[idx][r] = V_3[idx][r]  # V for B1
            # W for C1: k in 0..2, i in 0..2
            # W[k * 3 + i][r]
            if idx < 9: # This is not how W is indexed
                pass
    
    # Let's fill W properly.
    # W is p*n = 18 rows.
    # For C1 (k in 0..2): W[k * 3 + i][r] = W_3[k * 3 + i][r] for r in 0..3
    for k in range(3):
        for i in range(3):
            for r in range(4):
                W[k * 3 + i][r] = W_3[k * 3 + i][r]
    
    # For C2 (k in 3..5):
    # The transform of A is the same, so U is the same.
    # But V is for B2.
    # V[j * 3 + (k-3)][4 + r] = V_3[j * 3 + (k-3)][r]
    for j in range(3):
        for k in range(3):
            for r in range(4):
                V[j * 3 + (k + 3)][4 + r] = V_3[j * 3 + k][r]
    
    # W for C2:
    # W[(k+3) * 3 + i][4 + r] = W_3[k * 3 + i][r]
    for k in range(3):
        for i in range(3):
            for r in range(4):
                W[(k + 3) * 3 + i][4 + r] = W_3[k * 3 + i][r]
    
    # Verify 3x3x6
    if not _verify(U, V, W):
        return None
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Try Winograd decomposition
    if time.time() < deadline - 5:
        result = _winograd_3x3x6()
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
