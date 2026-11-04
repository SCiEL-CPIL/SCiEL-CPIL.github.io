import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 140  # safety margin from 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T_computed[i,j,:] += u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])
        
        return np.max(np.abs(T_computed - 1.0)) < tol

    def build_strassen_embedding(n, m, p, i0, j0, k0, U, V, W, col_start):
        """Embed a 2x2x2 Strassen block at position (i0, j0, k0)."""
        # Standard Strassen for 2x2 matrices:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5 + M6
        
        U_coeff = [
            {0: 1, 3: 1},   # M1: A00 + A11
            {2: 1, 3: 1},   # M2: A10 + A11
            {0: 1},          # M3: A00
            {3: 1},          # M4: A11
            {0: 1, 1: 1},   # M5: A00 + A10
            {2: 1, 0: -1},  # M6: A10 - A00
            {1: 1, 3: -1}   # M7: A01 - A11
        ]
        V_coeff = [
            {0: 1, 3: 1},   # M1: B00 + B11
            {0: 1},          # M2: B00
            {1: 1, 3: -1},  # M3: B01 - B11
            {2: 1, 0: -1},  # M4: B10 - B00
            {3: 1},          # M5: B11
            {0: 1, 1: 1},   # M6: B00 + B10
            {2: 1, 3: 1}    # M7: B01 + B11
        ]
        W_coeff = [
            {0: 1, 3: 1},   # M1: C00 + C11
            {2: 1, 3: -1},  # M2: C10 - C11
            {1: 1, 3: 1},   # M3: C01 + C11
            {0: 1, 2: 1},   # M4: C00 + C10
            {0: -1, 1: 1},  # M5: C01 - C00
            {3: 1},          # M6: C11
            {0: 1}           # M7: C00
        ]
        
        for l in range(7):
            col = col_start + l
            for li, coeff in U_coeff[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i0 + i_local, j0 + j_local, col] += coeff
            for lj, coeff in V_coeff[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j0 + j_local, k0 + k_local, col] += coeff
            for lc, coeff in W_coeff[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k0 + k_local, i0 + i_local, col] += coeff
    
    # Build a rank-30 construction:
    # Decompose 3x3x8 as follows:
    # - Use Strassen for the top-left 2x2 block of the 3x3 matrix (rows 0-1, cols 0-1)
    #   This handles all 8 columns of B for these 2x2 submatrices
    # - Handle the remaining terms (row 2 and col 2) with schoolbook
    
    # Actually, let me think more carefully.
    # The tensor is T[i,j,k] where A is n x m = 3 x 3, B is m x p = 3 x 8, C is p x n = 8 x 3
    # T[i,j,k] = sum_l A[i,l] * B[l,k]
    
    # A better approach: decompose by splitting the middle dimension m=3 into 2+1
    # Or split the output dimension p=8 into blocks
    
    # Let me use the divide-and-conquer from inspiration 2:
    # Split m=3 into m1=2 and m2=1
    # C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
    # 
    # A[:, :2] is 3x2, B[:2, :] is 2x8 -> product is 3x8
    # A[:, 2:] is 3x1, B[2:, :] is 1x8 -> product is 3x8
    #
    # For 3x2 @ 2x8: 
    #   Split n=3 into n1=2, n2=1
    #   C1 = A[:2, :2] @ B[:2, :] + A[2:, :2] @ B[:2, :]
    #   A[:2, :2] is 2x2, B[:2, :] is 2x8
    #   A[2:, :2] is 1x2, B[:2, :] is 2x8
    #
    # For 2x2 @ 2x8: Use 4 Strassen blocks (one for each pair of columns)
    #   Rank = 4 * 7 = 28
    # For 1x2 @ 2x8: 
    #   C = A[2,0] * B[0,:] + A[2,1] * B[1,:]
    #   This is 8 rank-1 terms (one for each column)
    #   Rank = 8
    #
    # Total for 3x2 @ 2x8: 28 + 8 = 36
    #
    # For 3x1 @ 1x8:
    #   C = A[i,2] * B[2,k] for each i,k
    #   This is 3*8 = 24 rank-1 terms
    #   Rank = 24
    #
    # Total: 36 + 24 = 60
    #
    # That's not great. Let me try a different decomposition.
    
    # Alternative: Split p=8 into 4 blocks of 2
    # C[:, 0:2] = A @ B[:, 0:2]
    # C[:, 2:4] = A @ B[:, 2:4]
    # etc.
    #
    # For each 3x3 @ 3x2 block:
    #   Use the current program's 17-rank construction
    #   4 blocks * 17 = 68
    #
    # That's what the current program does.
    
    # Let me try to improve the 3x3 @ 3x2 construction.
    # Current: 17 rank
    # Can we do better?
    
    # For 3x3 @ 3x2, the tensor has 3*3*2 = 18 entries.
    # The minimum rank is at least max(3*3, 3*2, 2*3) = 9 by trivial lower bound.
    # Actually, the tensor rank lower bound is more subtle.
    
    # Let me try to construct a better decomposition for 3x3 @ 3x2.
    
    def build_3x3_3x2_decomposition():
        """Build a decomposition for 3x3 @ 3x2 with minimal rank."""
        # Strategy: Use Strassen for a 2x2 submatrix, handle the rest with schoolbook
        # Split rows into 2+1, cols into 2+1
        
        n_local, m_local, p_local = 3, 3, 2
        U = np.zeros((n_local, m_local, 0))
        V = np.zeros((m_local, p_local, 0))
        W = np.zeros((p_local, n_local, 0))
        
        # Start with empty, build up
        rank = 0
        U = np.zeros((n_local, m_local, rank))
        V = np.zeros((m_local, p_local, rank))
        W = np.zeros((p_local, n_local, rank))
        
        # Use Strassen for the 2x2 block at (0,0) covering rows 0-1, cols 0-1, all k
        # But Strassen is for 2x2 @ 2x2, not 2x2 @ 2x2 where the second has 2 cols
        # 
        # Actually, let me think of this differently.
        # A is 3x3, B is 3x2, C is 2x3
        # 
        # C[k,i] = sum_j A[i,j] * B[j,k]
        # 
        # Let me split j into j1 (0,1) and j2 (2)
        # C[k,i] = sum_{j=0,1} A[i,j] * B[j,k] + A[i,2] * B[2,k]
        # 
        # The first part is (3x2) @ (2x2) = 3x2... wait, that's not right.
        # A[:, :2] is 3x2, B[:2, :] is 2x2
        # Product: (3x2) @ (2x2) = 3x2
        # But we want C which is 2x3.
        # 
        # I think I have the indices confused. Let me re-examine.
        # 
        # Matrix multiplication: C = A @ B where A is n x m, B is m x p
        # C is n x p, C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # In the tensor convention: T[i,j,k] where i indexes A rows, j indexes A cols/B rows, k indexes B cols
        # The output W[k*n+i] suggests C[k,i] = sum_j A[i,j] * B[j,k]
        # So C is p x n, not n x p.
        # 
        # This is the "transposed" convention. Let me work with it.
        # C[k,i] = sum_j A[i,j] * B[j,k]
        # A is n x m = 3 x 3
        # B is m x p = 3 x 8
        # C is p x n = 8 x 3
        # 
        # For 3x3 @ 3x2:
        # C[k,i] = sum_{j=0,1,2} A[i,j] * B[j,k] where k=0,1
        # 
        # Split j into 0,1 and 2:
        # C[k,i] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        # 
        # The first two terms: (A[i,0], A[i,1]) @ (B[0,k], B[1,k])
        # This is like a 3x2 matrix (rows i, cols j=0,1) times a 2x2 matrix (rows j=0,1, cols k=0,1)
        # 
        # For the 3x2 @ 2x2 part:
        # Let A1 = A[:, :2] (3x2), B1 = B[:2, :] (2x2)
        # C1[k,i] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
        # 
        # This is a 3x2 @ 2x2 product.
        # 
        # I can use Strassen-like decomposition for 2x2 @ 2x2 if I can reduce to that.
        # 
        # Actually, let me just use the known 17-rank construction from the current program
        # and try to optimize it further.
        
        # For now, return the 17-rank construction
        rank = 17
        U = np.zeros((n_local, m_local, rank))
        V = np.zeros((m_local, p_local, rank))
        W = np.zeros((p_local, n_local, rank))
        
        # Strassen part for 2x2 block at i in {0,1}, j in {0,1}, k in {0,1}
        U_coeff = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        V_coeff = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        W_coeff = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        
        col = 0
        for l in range(7):
            for li, coeff in U_coeff[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local, j_local, col] += coeff
            for lj, coeff in V_coeff[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local, k_local, col] += coeff
            for lc, coeff in W_coeff[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local, i_local, col] += coeff
            col += 1
        
        # Schoolbook for remaining terms
        # i in {0,1,2}, j=2, k in {0,1}
        for i in range(3):
            for k in range(2):
                U[i, 2, col] = 1.0
                V[2, k, col] = 1.0
                W[k, i, col] = 1.0
                col += 1
        # i=2, j in {0,1}, k in {0,1}
        for j in range(2):
            for k in range(2):
                U[2, j, col] = 1.0
                V[j, k, col] = 1.0
                W[k, 2, col] = 1.0
                col += 1
        
        return U, V, W
    
    # Build the full 3x3x8 decomposition using 4 blocks of 3x3x2
    U_block, V_block, W_block = build_3x3_3x2_decomposition()
    rank_block = U_block.shape[2]
    
    # Assemble
    rank_total = 4 * rank_block
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U: rows are i*m + j, same for all blocks
        for i in range(n):
            for j in range(m):
                U[i*m + j, col_start:col_end] = U_block[i, j, :]
        
        # V: rows are j*p + k_global, where k_global = k0 + k_local
        for j in range(m):
            for k_local in range(2):
                k_global = k0 + k_local
                V[j*p + k_global, col_start:col_end] = V_block[j, k_local, :]
        
        # W: rows are k_global*n + i, where k_global = k0 + k_local
        for k_local in range(2):
            k_global = k0 + k_local
            for i in range(n):
                W[k_global*n + i, col_start:col_end] = W_block[k_local, i, :]
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        # Try rounding
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        result_rounded = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        if validate(U_rounded, V_rounded, W_rounded):
            return result_rounded
        return schoolbook()
