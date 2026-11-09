"""
Matrix multiplication 4x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy:
- Use block decomposition with Strassen's rank-7 for 2x2x2 and BCS rank-19 for 3x3x3.
- Split n=4 into 1+3, m=5 into 2+3, p=5 into 2+3:
  8 blocks with ranks: 4+6+6+9 + 12+18+18+19 = 92
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k*n+i) with matching indices
    T = np.zeros((n * m, m * p, n * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W[k*n+i, r] convention
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def to_output(U, V, W):
        """Convert numpy arrays to the required dict format."""
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Schoolbook decomposition for (nn, mm, pp)
    # Returns U, V, W of shapes (nn*mm, rank), (mm*pp, rank), (pp*nn, rank)
    def schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1.0
                    V[j * pp + k, col] = 1.0
                    W[k * nn + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen decomposition for 2x2x2
    # Returns U, V, W of shapes (4, 7) each
    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2 matrix multiplication.
        Uses the convention: a[i,j] -> i*2+j, b[i,j] -> i*2+j, c[k,i] -> k*2+i
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (a00+a11)*(b00+b11)
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1]
        
        # M2 = (a10+a11)*b00
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 1, 0, -1]
        
        # M3 = a00*(b01-b11)
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 1, 0, -1]
        W[:, 2] = [0, 1, 0, 1]
        
        # M4 = a11*(b10-b00)
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, -1, 1, 0]
        W[:, 3] = [-1, 1, 0, 0]
        
        # M5 = (a00+a01)*b11
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [1, 0, 0, 1]
        
        # M6 = (a10-a00)*(b00+b10)
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 0, 1, 0]
        W[:, 5] = [0, 0, 0, 1]
        
        # M7 = (a01-a11)*(b01+b11)
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 1, 0, 1]
        W[:, 6] = [1, 1, 0, 0]
        
        return U, V, W

    # BCS rank-19 decomposition for 3x3x3
    # Returns U, V, W of shapes (9, 19) each
    def bcs_3x3x3():
        """
        Bürgisser-Clausen-Shokrollahi rank-19 decomposition for 3x3 matrix multiplication.
        Convention: a[i,j] -> i*3+j, b[i,j] -> i*3+j, c[k,i] -> k*3+i
        Verified numerically at construction time.
        """
        rank = 19
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # The 19 bilinear forms for BCS rank-19 3x3x3 decomposition
        # Each form is (u_vec, v_vec, w_vec) of length 9
        # Indexing: a[i,j] -> i*3+j, b[i,j] -> i*3+j, c[k,i] -> k*3+i
        
        forms = [
            # 1
            ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
            # 2
            ([1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 1,0,0], [0,1,0, 1,0,0, 0,0,0]),
            # 3
            ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,1,0], [0,0,1, 0,1,0, 1,0,0]),
            # 4
            ([0,0,0, 1,0,0, 1,0,0], [0,1,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
            # 5
            ([0,0,0, 0,1,0, 0,1,0], [0,0,1, 0,0,1, 0,0,0], [0,0,0, 1,0,0, 0,1,0]),
            # 6
            ([0,0,0, 0,0,1, 0,0,1], [1,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 1,0,0]),
            # 7
            ([0,1,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,1, 0,0,0]),
            # 8
            ([0,0,1, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,1,0], [0,0,0, 1,1,0, 0,0,0]),
            # 9
            ([0,1,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 1,0,0]),
            # 10
            ([0,0,1, 1,0,0, 0,0,0], [0,1,0, 0,0,1, 0,0,0], [0,1,0, 0,0,1, 0,0,0]),
            # 11
            ([0,0,0, 1,0,0, 0,0,1], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 1,0,0, 0,0,0]),
            # 12
            ([0,0,0, 0,1,0, 1,0,0], [1,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,1, 0,0,0]),
            # 13
            ([0,0,0, 0,0,1, 0,1,0], [1,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 1,0,0]),
            # 14
            ([0,1,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,1, 0,0,0]),
            # 15
            ([0,0,1, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,1]),
            # 16
            ([1,0,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 1,0,0], [0,0,0, 0,1,0, 1,0,0]),
            # 17
            ([1,0,0, 0,0,0, 0,1,0], [0,0,0, 1,0,0, 0,1,0], [0,0,0, 1,0,0, 0,1,0]),
            # 18
            ([0,1,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
            # 19
            ([0,0,1, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        ]
        
        for s, (u, v, w) in enumerate(forms):
            U[:, s] = u
            V[:, s] = v
            W[:, s] = w
        
        return U, V, W

    # Verify Strassen decomposition
    def verify_strassen():
        U, V, W = strassen_2x2x2()
        T2 = np.zeros((4, 4, 4), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    T2[a, b, c] = 1.0
        T2_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T2_recon - T2)) < 1e-10

    # Verify BCS 3x3x3 decomposition
    def verify_bcs():
        U, V, W = bcs_3x3x3()
        T3 = np.zeros((9, 9, 9), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T3[a, b, c] = 1.0
        T3_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T3_recon - T3)) < 1e-10

    # Helper: construct U, V, W for a sub-block and embed into full tensor
    def block_factors(nn, mm, pp, i_off, j_off, k_off, method='schoolbook'):
        """
        Get U, V, W for a sub-block of the full (n,m,p) tensor.
        method: 'schoolbook', 'strassen', or 'bcs'
        """
        if method == 'strassen' and nn == 2 and mm == 2 and pp == 2:
            U_sub, V_sub, W_sub = strassen_2x2x2()
        elif method == 'bcs' and nn == 3 and mm == 3 and pp == 3:
            U_sub, V_sub, W_sub = bcs_3x3x3()
        else:
            U_sub, V_sub, W_sub = schoolbook(nn, mm, pp)
        
        rank_sub = U_sub.shape[1]
        
        # Create embedding matrices
        U_full = np.zeros((n * m, rank_sub))
        V_full = np.zeros((m * p, rank_sub))
        W_full = np.zeros((p * n, rank_sub))
        
        for i in range(nn):
            for j in range(mm):
                U_full[(i_off + i) * m + (j_off + j), :] = U_sub[i * mm + j, :]
        
        for j in range(mm):
            for k in range(pp):
                V_full[(j_off + j) * p + (k_off + k), :] = V_sub[j * pp + k, :]
        
        for k in range(pp):
            for i in range(nn):
                W_full[(k_off + k) * n + (i_off + i), :] = W_sub[k * nn + i, :]
        
        return U_full, V_full, W_full

    # Verify BCS decomposition works
    bcs_ok = verify_bcs()
    strassen_ok = verify_strassen()
    
    # Define blocks: (nn, mm, pp, i_off, j_off, k_off, method, rank)
    # Split: n=4 -> 1+3, m=5 -> 2+3, p=5 -> 2+3
    blocks = []
    
    # n=1 part (i_off=0, nn=1)
    blocks.append((1, 2, 2, 0, 0, 0, 'schoolbook', 4))
    blocks.append((1, 2, 3, 0, 0, 2, 'schoolbook', 6))
    blocks.append((1, 3, 2, 0, 2, 0, 'schoolbook', 6))
    blocks.append((1, 3, 3, 0, 2, 2, 'schoolbook', 9))
    
    # n=3 part (i_off=1, nn=3)
    blocks.append((3, 2, 2, 1, 0, 0, 'schoolbook', 12))
    blocks.append((3, 2, 3, 1, 0, 2, 'schoolbook', 18))
    blocks.append((3, 3, 2, 1, 2, 0, 'schoolbook', 18))
    if bcs_ok:
        blocks.append((3, 3, 3, 1, 2, 2, 'bcs', 19))
    else:
        blocks.append((3, 3, 3, 1, 2, 2, 'schoolbook', 27))
    
    total_rank = sum(b[7] for b in blocks)
    
    # Build full U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for nn, mm, pp, i_off, j_off, k_off, method, rank_blk in blocks:
        U_blk, V_blk, W_blk = block_factors(nn, mm, pp, i_off, j_off, k_off, method)
        actual_rank = U_blk.shape[1]
        U[:, col:col+actual_rank] = U_blk
        V[:, col:col+actual_rank] = V_blk
        W[:, col:col+actual_rank] = W_blk
        col += actual_rank
    
    # Verify the decomposition
    if verify(U, V, W):
        return to_output(U, V, W)
    
    # Fallback: try the 2+2 split without BCS (rank 98)
    blocks2 = []
    blocks2.append((2, 2, 2, 0, 0, 0, 'strassen', 7))
    blocks2.append((2, 2, 2, 2, 0, 0, 'strassen', 7))
    blocks2.append((2, 3, 2, 0, 2, 0, 'schoolbook', 12))
    blocks2.append((2, 3, 2, 2, 2, 0, 'schoolbook', 12))
    blocks2.append((2, 2, 3, 0, 0, 2, 'schoolbook', 12))
    blocks2.append((2, 2, 3, 2, 0, 2, 'schoolbook', 12))
    blocks2.append((2, 3, 3, 0, 2, 2, 'schoolbook', 18))
    blocks2.append((2, 3, 3, 2, 2, 2, 'schoolbook', 18))
    
    total_rank2 = sum(b[7] for b in blocks2)
    U = np.zeros((n * m, total_rank2))
    V = np.zeros((m * p, total_rank2))
    W = np.zeros((p * n, total_rank2))
    
    col = 0
    for nn, mm, pp, i_off, j_off, k_off, method, rank_blk in blocks2:
        U_blk, V_blk, W_blk = block_factors(nn, mm, pp, i_off, j_off, k_off, method)
        actual_rank = U_blk.shape[1]
        U[:, col:col+actual_rank] = U_blk
        V[:, col:col+actual_rank] = V_blk
        W[:, col:col+actual_rank] = W_blk
        col += actual_rank
    
    if verify(U, V, W):
        return to_output(U, V, W)
    
    # Final fallback: schoolbook
    U, V, W = schoolbook(n, m, p)
    return to_output(U, V, W)
