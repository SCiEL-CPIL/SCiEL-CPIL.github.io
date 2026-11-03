"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a Strassen-based decomposition with rank 98.
    """
    start = time.time()
    deadline = start + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook rank-128 as fallback
    def _schoolbook():
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
    
    def _verify(U, V, W):
        """Check decomposition correctness."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    # Strassen 2x2 factors - carefully verified
    # Each factor: (u_vec[4], v_vec[4], w_vec[4])
    # u: A00, A01, A10, A11 (row-major)
    # v: B00, B01, B10, B11 (row-major)
    # w: C00, C01, C10, C11 (row-major)
    #
    # Standard Strassen:
    # p1 = (A00+A11)(B00+B11)
    # p2 = (A10+A11)(B00)
    # p3 = A00(B01-B11)
    # p4 = A11(B10-B00)
    # p5 = (A00+A01)(B11)
    # p6 = (A10-A00)(B00+B01)
    # p7 = (A01-A11)(B10+B11)
    #
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # p2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # p3=A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # p4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # p5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # p6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # p7=(A01-A11)(B10+B11)
    ]
    
    # Verify Strassen 2x2 correctness
    def _verify_strassen_2x2():
        rank = 7
        U3 = np.zeros((2, 2, rank))
        V3 = np.zeros((2, 2, rank))
        W3 = np.zeros((2, 2, rank))
        for s in range(7):
            u, v, w = S[s]
            for idx in range(4):
                i, j = idx // 2, idx % 2
                U3[i, j, s] = u[idx]
                V3[i, j, s] = v[idx]
                W3[i, j, s] = w[idx]
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    strassen_ok = _verify_strassen_2x2()
    
    if not strassen_ok:
        return _schoolbook()
    
    # Build rank-98 decomposition for 4x4 * 4x8
    # Strategy: Split B into two 4x4 blocks (B_left, B_right)
    # For each block, compute A * B_block using recursive Strassen (rank 49 each)
    # Total rank = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)
    
    for t in range(2):  # Two 4x4 subproblems (left and right halves of B)
        for s in range(7):  # Top-level Strassen factor (block level)
            u_block, v_block, w_block = S[s]
            for r in range(7):  # Scalar Strassen factor (element level)
                u_scalar, v_scalar, w_scalar = S[r]
                col = t * 49 + s * 7 + r
                
                # U: A is 4x4, split into four 2x2 blocks
                # u_block[bi*2+bj] gives coefficient for A-block(bi,bj)
                # u_scalar[ei*2+ej] gives coefficient within the 2x2 block
                for bi in range(2):
                    for bj in range(2):
                        cb = u_block[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_scalar[ei*2+ej]
                                if cs == 0:
                                    continue
                                i = 2*bi + ei
                                j = 2*bj + ej
                                U[i*m + j][col] += cb * cs
                
                # V: B block for subproblem t
                # B is 4x8, split into left (cols 0-3) and right (cols 4-7)
                # For subproblem t, B_block is 4x4 with columns offset by 4*t
                for bi in range(2):
                    for bj in range(2):
                        cb = v_block[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_scalar[ei*2+ej]
                                if cs == 0:
                                    continue
                                j = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                V[j*p + k][col] += cb * cs
                
                # W: C block for subproblem t
                # C is 4x8, split into left (cols 0-3) and right (cols 4-7)
                # W index: k*n + i where k is column of C, i is row
                for bi in range(2):
                    for bj in range(2):
                        cb = w_block[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_scalar[ei*2+ej]
                                if cs == 0:
                                    continue
                                i = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                W[k*n + i][col] += cb * cs
    
    # Verify the full decomposition
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
