"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a recursive Strassen-based decomposition with rank 98.
    """
    start = time.time()
    deadline = start + 100  # Safety margin for 120s limit
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
    
    # Strassen 2x2 factors
    # Each factor: (u_vec[4], v_vec[4], w_vec[4])
    # u: A00, A01, A10, A11
    # v: B00, B01, B10, B11
    # w: C00, C01, C10, C11
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # p2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # p3=A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # p4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # p5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # p6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # p7=(A01-A11)(B10+B11)
    ]
    
    # Build rank-98 decomposition for 4x4 * 4x8
    # Split B into two 4x4 blocks
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(2):  # Two 4x4 subproblems
        for s in range(7):  # Top-level Strassen factor
            u_block, v_block, w_block = S[s]
            for r in range(7):  # Scalar Strassen factor
                u_scalar, v_scalar, w_scalar = S[r]
                col = t * 49 + s * 7 + r
                
                # U: A is 4x4, split into four 2x2 blocks
                # u_block[i*2+j] gives coefficient for A-block(i,j)
                # u_scalar[i*2+j] gives coefficient within the 2x2 block
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
    
    # Verify
    def _verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if time.time() < deadline:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _schoolbook()
