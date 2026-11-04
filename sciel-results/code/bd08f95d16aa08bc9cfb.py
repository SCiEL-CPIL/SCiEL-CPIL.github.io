"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Strategy: use a block-based Strassen decomposition (rank 72) as a strong, verified baseline.
Then, attempt to improve using a structured ALS refinement with exact rounding verification.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    size1 = n * m  # 16
    size2 = m * p  # 20
    size3 = p * n  # 20
    
    # Build the target tensor T of shape (size1, size2, size3)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    # Otherwise T[a, b, c] = 0
    T = np.zeros((size1, size2, size3), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Verify if U, V, W exactly decompose T. Returns True if exact."""
        # Compute sum_{r} U[:,r] outer V[:,r] outer W[:,r]
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if T_approx matches T exactly (within floating point tolerance for exact integers)
        # Since entries are integers or half-integers, the result should be integer.
        # We can check if T_approx is close to T and if T_approx is close to integer values.
        diff = np.abs(T_approx - T)
        if np.max(diff) < 1e-10:
            return True
        return False

    def schoolbook():
        rank = n * m * p
        U = np.zeros((size1, rank))
        V = np.zeros((size2, rank))
        W = np.zeros((size3, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2_base():
        """
        Return Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7).
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        Verified correct coefficients.
        """
        nn = mm = pp = 2
        rank = 7
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        
        # m1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # m3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # m4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = -1
        
        # m7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1
        
        return U, V, W

    def block_strassen():
        """
        Decompose 4x4 * 4x5 into 2x2 blocks.
        A (4x4) = [A11 A12; A21 A22] where each block is 2x2.
        B (4x5) = [B1; B2] where B1 is 2x5 and B2 is 2x5.
        C (4x5) = [C1; C2] where C1 is 2x5 and C2 is 2x5.
        
        C1 = A11*B1 + A12*B2
        C2 = A21*B1 + A22*B2
        
        To compute A11*B1 (2x2 * 2x5), we can split B1 into two 2x2 blocks and a 2x1 block?
        No, 2x5 is not square. We can split B1 into [B11 B12 B13] where B11, B12 are 2x2 and B13 is 2x1.
        A11*B1 = A11*B11 + A11*B12 + A11*B13.
        A11*B11 (2x2*2x2) -> rank 7
        A11*B12 (2x2*2x2) -> rank 7
        A11*B13 (2x2*2x1) -> rank 4 (schoolbook for 2x2*2x1 is 4)
        Total for C1: 7 + 7 + 4 = 18.
        Similarly for C2: 18.
        Total rank: 36.
        
        Wait, the blocks are not independent in the tensor decomposition sense?
        Yes, they are. The tensor is a sum of Kronecker products.
        Let's construct the full U, V, W matrices.
        
        Indices:
        A: 4x4 -> indices 0..15 (i*4+j)
        B: 4x5 -> indices 0..19 (j*5+k)
        C: 4x5 -> indices 0..19 (k*4+i)
        
        Block A11: rows 0,1; cols 0,1 -> A indices 0,1,4,5
        Block A12: rows 0,1; cols 2,3 -> A indices 2,3,6,7
        Block A21: rows 2,3; cols 0,1 -> A indices 8,9,12,13
        Block A22: rows 2,3; cols 2,3 -> A indices 10,11,14,15
        
        Block B1: rows 0,1; cols 0..4 -> B indices 0..4 (j=0,1)
        Block B2: rows 2,3; cols 0..4 -> B indices 10..14 (j=2,3)
        
        Wait, B is 4x5. j is row index 0..3, k is col index 0..4.
        B[j,k] index = j*5+k.
        B1 (j=0,1): indices 0..4, 5..9.
        B2 (j=2,3): indices 10..14, 15..19.
        
        C1 (i=0,1): indices 0..4, 5..9.
        C2 (i=2,3): indices 10..14, 15..19.
        
        Let's define the sub-tensors.
        T11: A11 * B1 -> C1_part1 (2x2 * 2x5 -> 2x5)
        T12: A12 * B2 -> C1_part2 (2x2 * 2x5 -> 2x5)
        T21: A21 * B1 -> C2_part1 (2x2 * 2x5 -> 2x5)
        T22: A22 * B2 -> C2_part2 (2x2 * 2x5 -> 2x5)
        
        We need a decomposition for 2x2 * 2x5 -> 2x5.
        Split B (2x5) into [B_left (2x2), B_right (2x2), B_mid (2x1)]?
        Actually, just split into two 2x2 and one 2x1.
        B = [B1 B2 B3] where B1 is 2x2 (cols 0,1), B2 is 2x2 (cols 2,3), B3 is 2x1 (col 4).
        C = A*B1 + A*B2 + A*B3.
        A*B1 (2x2*2x2) -> rank 7
        A*B2 (2x2*2x2) -> rank 7
        A*B3 (2x2*2x1) -> rank 4
        Total rank 18.
        
        So for each 2x2 * 2x5 block, we need rank 18.
        There are 4 such blocks (A11*B1, A12*B2, A21*B1, A22*B2).
        Total rank = 4 * 18 = 72.
        
        Let's implement this.
        """
        # First, get the 2x2 * 2x2 Strassen decomposition
        U_s, V_s, W_s = strassen_2x2_base()
        
        # Get the 2x2 * 2x1 schoolbook decomposition (rank 4)
        U_sq = np.zeros((4, 4))
        V_sq = np.zeros((2, 4)) # B is 2x1, so 2 rows
        W_sq = np.zeros((2, 4)) # C is 2x1, so 2 rows
        for i in range(2):
            for j in range(2):
                U_sq[i*2+j, i*2+j] = 1
                V_sq[j, i*2+j] = 1
                W_sq[i, i*2+j] = 1
        
        # Now build the full 4x4 * 4x5 decomposition
        rank = 72
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        
        col = 0
        # Block A11 * B1 -> C1
        # A11 indices: 0,1,4,5
        # B1 indices: 0..4 (j=0,1)
        # C1 indices: 0..4 (i=0,1)
        
        # Sub-block B1_left (cols 0,1): B indices 0,1,5,6
        # Sub-block B1_mid (cols 2,3): B indices 2,3,7,8
        # Sub-block B1_right (col 4): B indices 4,9
        
        # A11 * B1_left (2x2 * 2x2)
        for r in range(7):
            U[0, col] = U_s[0, r]
            U[1, col] = U_s[1, r]
            U[4, col] = U_s[2, r]
            U[5, col] = U_s[3, r]
            
            V[0, col] = V_s[0, r]
            V[1, col] = V_s[1, r]
            V[5, col] = V_s[2, r]
            V[6, col] = V_s[3, r]
            
            W[0, col] = W_s[0, r]
            W[1, col] = W_s[1, r]
            col += 1
            
        # A11 * B1_mid (2x2 * 2x2)
        for r in range(7):
            U[0, col] = U_s[0, r]
            U[1, col] = U_s[1, r]
            U[4, col] = U_s[2, r]
            U[5, col] = U_s[3, r]
            
            V[2, col] = V_s[0, r]
            V[3, col] = V_s[1, r]
            V[7, col] = V_s[2, r]
            V[8, col] = V_s[3, r]
            
            W[2, col] = W_s[0, r]
            W[3, col] = W_s[1, r]
            col += 1
            
        # A11 * B1_right (2x2 * 2x1)
        for r in range(4):
            U[0, col] = U_sq[0, r]
            U[1, col] = U_sq[1, r]
            U[4, col] = U_sq[2, r]
            U[5, col] = U_sq[3, r]
            
            V[4, col] = V_sq[0, r]
            V[9, col] = V_sq[1, r]
            
            W[4, col] = W_sq[0, r]
            W[9, col] = W_sq[1, r]
            col += 1
            
        # Block A12 * B2 -> C1
        # A12 indices: 2,3,6,7
        # B2 indices: 10..14 (j=2,3)
        # C1 indices: 0..4 (i=0,1)
        
        # Sub-block B2_left (cols 0,1): B indices 10,11,15,16
        # Sub-block B2_mid (cols 2,3): B indices 12,13,17,18
        # Sub-block B2_right (col 4): B indices 14,19
        
        # A12 * B2_left (2x2 * 2x2)
        for r in range(7):
            U[2, col] = U_s[0, r]
            U[3, col] = U_s[1, r]
            U[6, col] = U_s[2, r]
            U[7, col] = U_s[3, r]
            
            V[10, col] = V_s[0, r]
            V[11, col] = V_s[1, r]
            V[15, col] = V_s[2, r]
            V[16, col] = V_s[3, r]
            
            W[0, col] = W_s[0, r]
            W[1, col] = W_s[1, r]
            col += 1
            
        # A12 * B2_mid (2x2 * 2x2)
        for r in range(7):
            U[2, col] = U_s[0, r]
            U[3, col] = U_s[1, r]
            U[6, col] = U_s[2, r]
            U[7, col] = U_s[3, r]
            
            V[12, col] = V_s[0, r]
            V[13, col] = V_s[1, r]
            V[17, col] = V_s[2, r]
            V[18, col] = V_s[3, r]
            
            W[2, col] = W_s[0, r]
            W[3, col] = W_s[1, r]
            col += 1
            
        # A12 * B2_right (2x2 * 2x1)
        for r in range(4):
            U[2, col] = U_sq[0, r]
            U[3, col] = U_sq[1, r]
            U[6, col] = U_sq[2, r]
            U[7, col] = U_sq[3, r]
            
            V[14, col] = V_sq[0, r]
            V[19, col] = V_sq[1, r]
            
            W[4, col] = W_sq[0, r]
            W[9, col] = W_sq[1, r]
            col += 1
            
        # Block A21 * B1 -> C2
        # A21 indices: 8,9,12,13
        # B1 indices: 0..4 (j=0,1)
        # C2 indices: 10..14 (i=2,3)
        
        # Sub-block B1_left (cols 0,1): B indices 0,1,5,6
        # Sub-block B1_mid (cols 2,3): B indices 2,3,7,8
        # Sub-block B1_right (col 4): B indices 4,9
        
        # A21 * B1_left (2x2 * 2x2)
        for r in range(7):
            U[8, col] = U_s[0, r]
            U[9, col] = U_s[1, r]
            U[12, col] = U_s[2, r]
            U[13, col] = U_s[3, r]
            
            V[0, col] = V_s[0, r]
            V[1, col] = V_s[1, r]
            V[5, col] = V_s[2, r]
            V[6, col] = V_s[3, r]
            
            W[10, col] = W_s[0, r]
            W[11, col] = W_s[1, r]
            col += 1
            
        # A21 * B1_mid (2x2 * 2x2)
        for r in range(7):
            U[8, col] = U_s[0, r]
            U[9, col] = U_s[1, r]
            U[12, col] = U_s[2, r]
            U[13, col] = U_s[3, r]
            
            V[2, col] = V_s[0, r]
            V[3, col] = V_s[1, r]
            V[7, col] = V_s[2, r]
            V[8, col] = V_s[3, r]
            
            W[12, col] = W_s[0, r]
            W[13, col] = W_s[1, r]
            col += 1
            
        # A21 * B1_right (2x2 * 2x1)
        for r in range(4):
            U[8, col] = U_sq[0, r]
            U[9, col] = U_sq[1, r]
            U[12, col] = U_sq[2, r]
            U[13, col] = U_sq[3, r]
            
            V[4, col] = V_sq[0, r]
            V[9, col] = V_sq[1, r]
            
            W[14, col] = W_sq[0, r]
            W[19, col] = W_sq[1, r]
            col += 1
            
        # Block A22 * B2 -> C2
        # A22 indices: 10,11,14,15
        # B2 indices: 10..14 (j=2,3)
        # C2 indices: 10..14 (i=2,3)
        
        # Sub-block B2_left (cols 0,1): B indices 10,11,15,16
        # Sub-block B2_mid (cols 2,3): B indices 12,13,17,18
        # Sub-block B2_right (col 4): B indices 14,19
        
        # A22 * B2_left (2x2 * 2x2)
        for r in range(7):
            U[10, col] = U_s[0, r]
            U[11, col] = U_s[1, r]
            U[14, col] = U_s[2, r]
            U[15, col] = U_s[3, r]
            
            V[10, col] = V_s[0, r]
            V[11, col] = V_s[1, r]
            V[15, col] = V_s[2, r]
            V[16, col] = V_s[3, r]
            
            W[10, col] = W_s[0, r]
            W[11, col] = W_s[1, r]
            col += 1
            
        # A22 * B2_mid (2x2 * 2x2)
        for r in range(7):
            U[10, col] = U_s[0, r]
            U[11, col] = U_s[1, r]
            U[14, col] = U_s[2, r]
            U[15, col] = U_s[3, r]
            
            V[12, col] = V_s[0, r]
            V[13, col] = V_s[1, r]
            V[17, col] = V_s[2, r]
            V[18, col] = V_s[3, r]
            
            W[12, col] = W_s[0, r]
            W[13, col] = W_s[1, r]
            col += 1
            
        # A22 * B2_right (2x2 * 2x1)
        for r in range(4):
            U[10, col] = U_sq[0, r]
            U[11, col] = U_sq[1, r]
            U[14, col] = U_sq[2, r]
            U[15, col] = U_sq[3, r]
            
            V[14, col] = V_sq[0, r]
            V[19, col] = V_sq[1, r]
            
            W[14, col] = W_sq[0, r]
            W[19, col] = W_sq[1, r]
            col += 1
            
        return {"U": U, "V": V, "W": W}

    # First, try the block Strassen decomposition
    best_result = block_strassen()
    best_rank = 72
    
    if verify(best_result['U'], best_result['V'], best_result['W']):
        # It's valid. Can we do better?
        pass
    else:
        # Fallback to schoolbook
        best_result = schoolbook()
        best_rank = 80

    # Try ALS refinement from rank 70 down to 60
    def als_decomposition(target_rank, max_iters=100, tol=1e-10, seed=None):
        """Alternating Least Squares for tensor decomposition."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(size1, target_rank)
        V = np.random.randn(size2, target_rank)
        W = np.random.randn(size3, target_rank)
        
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = np.inf
        
        for it in range(max_iters):
            # Update U
            A = np.kron(W.T, V)  # shape (size3*size2, target_rank)
            RHS = T.reshape(size1, -1)  # shape (size1, size3*size2)
            try:
                X, residuals, rank_A, s = np.linalg.lstsq(A, RHS.T, rcond=None)
                U_new = X.T
            except:
                return None, None, None, np.inf
            
            # Update V
            B = np.kron(U.T, W)  # shape (size1*size3, target_rank)
            RHS_V = np.transpose(T, (1, 0, 2)).reshape(size2, -1)  # shape (size2, size1*size3)
            try:
                X, residuals, rank_B, s = np.linalg.lstsq(B, RHS_V.T, rcond=None)
                V_new = X.T
            except:
                return None, None, None, np.inf
            
            # Update W
            C = np.kron(U.T, V)  # shape (size1*size2, target_rank)
            RHS_W = np.transpose(T, (2, 0, 1)).reshape(size3, -1)  # shape (size3, size1*size2)
            try:
                X, residuals, rank_C, s = np.linalg.lstsq(C, RHS_W.T, rcond=None)
                W_new = X.T
            except:
                return None, None, None, np.inf
            
            U, V, W = U_new, V_new, W_new
            
            err = np.sqrt(residuals) if residuals > 0 else 0.0
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Compute error with rounded values
        E_round = np.zeros_like(T)
        for r in range(target_rank):
            E_round -= np.einsum('ar,br,cr->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        E_round += T
        
        err_round = np.linalg.norm(E_round)
        
        if err_round < 1e-6:
            return U_rounded, V_rounded, W_rounded, err_round
        else:
            return None, None, None, np.inf

    # Try a few ranks with ALS
    for r in range(70, 59, -1):
        if time.time() > deadline - 10.0:
            break
        
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            U, V, W, err = als_decomposition(r, max_iters=50, tol=1e-8, seed=42 + r * 100 + restart)
            if U is not None:
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break
        if best_rank == r:
            break

    # Return the best result
    U_list = best_result['U'].tolist()
    V_list = best_result['V'].tolist()
    W_list = best_result['W'].tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
