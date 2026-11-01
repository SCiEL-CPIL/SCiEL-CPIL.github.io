import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """
    Verify decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
    rank = len(U[0]) if U else 0
    # Convert to numpy arrays for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to tensors
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # C[i, k] = sum_c W[k, i, c] * (sum_j U[i, j, c] A[i, j]) * (sum_j V[j, k, c] B[j, k])
        # Let's compute this efficiently
        # factor_A[i, c] = sum_j U[i, j, c] A[i, j]
        # factor_B[k, c] = sum_j V[j, k, c] B[j, k]
        
        # U_t is (n, m, rank). A is (n, m).
        # We want (n, rank).
        # np.einsum('ijc,ij->ic', U_t, A)
        factor_A = np.einsum('ijc,ij->ic', U_t, A)
        
        # V_t is (m, p, rank). B is (m, p).
        # We want (p, rank).
        # np.einsum('jk c, jk -> kc', V_t, B)
        factor_B = np.einsum('jk c, jk -> kc', V_t, B)
        
        # C[i, k] = sum_c W[k, i, c] * factor_A[i, c] * factor_B[k, c]
        # W_t is (p, n, rank).
        # We need to align indices: W[k, i, c], factor_A[i, c], factor_B[k, c]
        # np.einsum('kic, ic, kc -> ik', W_t, factor_A, factor_B)
        
        C_computed = np.einsum('kic, ic, kc -> ik', W_t, factor_A, factor_B)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _try_rank_r(r, n, m, p, time_limit=20):
    """
    Attempt to find a rank-r decomposition by solving a linear system.
    The system is linear in the entries of U, V, W if we fix one set of variables?
    No, it's trilinear.
    However, we can use the "bilinear" trick.
    For a fixed rank r, we have r terms.
    C[i,k] = sum_c W[k,i,c] U[i,:,c] . A[i,:] * V[:,k,c] . B[:,k]
    
    This is not linear in U, V, W simultaneously.
    However, we can view it as:
    For each c, let u_c be the vector of U[:,c], v_c be vector of V[:,c], w_c be vector of W[:,c].
    The condition is that the tensor T = sum_c w_c ⊗ u_c ⊗ v_c equals the multiplication tensor M.
    
    We can set up a linear system if we guess some variables.
    A common approach for small ranks is to fix U and V to have a specific structure (e.g., sparse) and solve for W, or fix W and solve for U, V.
    
    Alternative: Use the fact that for rank r, the tensor must have rank <= r.
    We can check if the multiplication tensor can be expressed as sum of r rank-1 tensors.
    
    Let's try a structured search for r < 56.
    We know 56 works. We suspect 52 or 48 might work?
    Actually, for 2x4x8, the Strassen-like bound is often tight or close.
    
    Let's try to solve for r by assuming a sparse pattern for U and V.
    If we assume U and V are sparse (e.g., 1s and 0s), the problem becomes solving for W.
    But W is also part of the product.
    
    Let's use a different approach:
    The multiplication tensor M_{i,j,k} = delta_{j,j} (i.e., 1 if indices match in the summation).
    M is a 3-tensor of size 2x4x8.
    We want M = sum_{c=1}^r w_c ⊗ u_c ⊗ v_c.
    
    We can fix the first r-1 terms and solve for the last one? No.
    
    Let's try a random search with integer constraints for small r, but use the linear system check for consistency.
    Actually, a very effective method for these small sizes is to use the "tensor rank" computation via algebraic geometry or simple enumeration if the support is small.
    
    Given the time constraint, let's try to construct a rank 52 solution if possible, or verify 56.
    The current program has 56.
    Is there a known lower rank for 2x4x8?
    2x2x2 is 7.
    2x4x4 is 2*7 = 14? Or 2x2x2 blocks?
    If we split 4x8 into 2 blocks of 2x8?
    2x2x8 rank?
    
    Let's stick to the known 56 but verify it faster.
    If we can find a 52 solution, great.
    Let's try a heuristic:
    Split the 4x8 matrix into four 2x2 blocks?
    A is 2x4. B is 4x8.
    
    Let's just return the known 56 solution but with a faster verification.
    The previous verification was O(rank * n * m * p) in Python loops.
    The new verification uses numpy einsum, which is much faster.
    This allows us to potentially run a search for lower ranks within the time budget.
    
    Let's implement a simple search for r in [48, 52, 56] using random integer matrices and checking if the linear system for the "residual" has a solution?
    No, that's complex.
    
    Let's just use the 56 solution. It is safe.
    """
    pass

def _construct_strassen_56():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2
    U_s = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]
    ]
    V_s = [
        [1, 0, 0, 1], [0, 0, 0, 1], [1, 0, -1, 0], [0, -1, 0, 1],
        [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0], [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # We have 2x4 * 4x8.
    # Split A into two 2x2 blocks horizontally: A = [A1, A2]
    # Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = [A1, A2] * [[B11, B12], [B21, B22]] = [A1B11 + A2B21, A1B12 + A2B22]
    # C is 2x8, so two 2x4 blocks? No, 2x8 is two 2x4 blocks?
    # C = [C1, C2] where C1 is 2x4 and C2 is 2x4.
    # C1 = A1B11 + A2B21. Both are 2x2 * 2x2?
    # B11 is 2x2, B21 is 2x2.
    # So C1 is sum of two 2x2 products.
    # C2 is sum of two 2x2 products.
    
    # Total products: 4 independent 2x2x2 multiplications?
    # Wait, A1B11, A2B21, A1B12, A2B22.
    # Each 2x2x2 mult has rank 7.
    # Total rank 4 * 7 = 28?
    # Why did the previous code use 56?
    # Let's check the previous code's block structure.
    # It used A1_idx and A2_idx.
    # A1_idx = [0, 1, 4, 5] -> A[0,0], A[0,1], A[1,0], A[1,1] (A1)
    # A2_idx = [2, 3, 6, 7] -> A[0,2], A[0,3], A[1,2], A[1,3] (A2)
    # B_top = [[0,1,8,9], [2,3,10,11], ...] -> B[0,0], B[0,1], B[1,0], B[1,1] (B11)
    # B_bot = [[16,17,24,25], ...] -> B[2,0], B[2,1], B[3,0], B[3,1] (B21)
    # So for the first block of C (cols 0,1), it used A1*B11 + A2*B21.
    # That requires 2 * 7 = 14 terms.
    # There are 4 blocks of C (cols 0-1, 2-3, 4-5, 6-7).
    # C0: A1B11 + A2B21 (14)
    # C1: A1B12 + A2B22 (14)
    # C2: A1B13 + A2B23 (14)
    # C3: A1B14 + A2B24 (14)
    # Total 56.
    
    # Can we do better?
    # C = A B.
    # If we use Strassen for the whole thing?
    # 2x4x8.
    # Pad to 4x4x8? No.
    # Pad to 4x4x4?
    # 4x4x4 rank is 14 (Strassen).
    # If we pad 2x4 to 4x4 and 4x8 to 4x4 (zero padding), we get 4x4x4.
    # Rank 14.
    # But the zero padding might allow rank reduction?
    # The tensor is a sub-tensor of the 4x4x4 multiplication tensor.
    # The rank of a sub-tensor is <= rank of the full tensor.
    # So rank <= 14?
    # Wait, 2x4x8 is not a sub-tensor of 4x4x4 in a way that preserves the multiplication structure directly without mixing.
    # The indices are (i,j,k).
    # i in 0..1, j in 0..3, k in 0..7.
    # If we embed in 4x4x8, rank is 2 * (rank of 2x2x8?)
    
    # Actually, the known lower bound for 2x4x8 is not 14.
    # The 4x4x4 tensor has rank 14.
    # The 2x4x8 tensor is "smaller" in some dimensions but "larger" in others.
    # 2x4x8 has 64 entries. 4x4x4 has 64 entries.
    # They have the same number of non-zeros?
    # 2x4x8: 2*4*8 = 64.
    # 4x4x4: 4*4*4 = 64.
    # The multiplication tensor for 2x4x8 is isomorphic to a sub-tensor of 4x4x4?
    # No, the structure is different.
    
    # Let's stick to 56. It is safe.
    
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    # B blocks:
    # B is 4x8.
    # B11: rows 0,1 cols 0,1
    # B12: rows 0,1 cols 2,3
    # B13: rows 0,1 cols 4,5
    # B14: rows 0,1 cols 6,7
    # B21: rows 2,3 cols 0,1
    # B22: rows 2,3 cols 2,3
    # B23: rows 2,3 cols 4,5
    # B24: rows 2,3 cols 6,7
    
    # C blocks:
    # C0: cols 0,1
    # C1: cols 2,3
    # C2: cols 4,5
    # C3: cols 6,7
    
    # C0 = A1B11 + A2B21
    # C1 = A1B12 + A2B22
    # C2 = A1B13 + A2B23
    # C3 = A1B14 + A2B24
    
    blocks = [
        (A1_idx, [0, 1, 8, 9], [0, 1, 2, 3]), # A1, B11, C0
        (A2_idx, [16, 17, 24, 25], [0, 1, 2, 3]), # A2, B21, C0
        (A1_idx, [2, 3, 10, 11], [4, 5, 6, 7]), # A1, B12, C1
        (A2_idx, [18, 19, 26, 27], [4, 5, 6, 7]), # A2, B22, C1
        (A1_idx, [4, 5, 12, 13], [8, 9, 10, 11]), # A1, B13, C2
        (A2_idx, [20, 21, 28, 29], [8, 9, 10, 11]), # A2, B23, C2
        (A1_idx, [6, 7, 14, 15], [12, 13, 14, 15]), # A1, B14, C3
        (A2_idx, [22, 23, 30, 31], [12, 13, 14, 15]), # A2, B24, C3
    ]
    
    col_offset = 0
    for a_idx, b_idx, c_idx in blocks:
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[a_idx[r]][col] = U_s[f][r]
                V[b_idx[r]][col] = V_s[f][r]
                W[c_idx[r]][col] = W_s[r][f]
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    sol = _construct_strassen_56()
    
    # Fast verification
    if _verify(sol["U"], sol["V"], sol["W"], trials=20):
        return sol
    else:
        # Fallback to schoolbook
        n, m, p = 2, 4, 8
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
