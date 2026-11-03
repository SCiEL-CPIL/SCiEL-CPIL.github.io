"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0  # Safety margin from 120s limit

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a 2-level Strassen decomposition with rank 98.
    
    Approach: Split B (4x8) into two 4x4 blocks B1, B2.
    Compute C1 = A*B1 and C2 = A*B2 using 2-level Strassen (rank 49 each).
    Total rank: 98.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    
    def _time_left():
        return DEADLINE_SECONDS - (time.time() - start)
    
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
    # Indices correspond to 2x2 blocks flattened: 00, 01, 10, 11
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
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-check deadline
    if _time_left() < 1.0:
        return _schoolbook()
    
    # We treat the 4x4x8 problem as two independent 4x4x4 problems.
    # The first 49 ranks correspond to C1 = A * B[:, 0:4]
    # The next 49 ranks correspond to C2 = A * B[:, 4:8]
    
    for t in range(2):  # t=0 for left half of B/C, t=1 for right half
        offset = t * 49
        for s in range(7):  # Top-level Strassen factor (for 4x4 split into 2x2 blocks)
            u_block, v_block, w_block = S[s]
            for r in range(7):  # Bottom-level Strassen factor (for 2x2 split into 1x1 blocks)
                u_scalar, v_scalar, w_scalar = S[r]
                col = offset + s * 7 + r
                
                # U: A is 4x4.
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
                
                # V: B is 4x8.
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
                                k_loc = 2*bj + ej
                                k = 4*t + k_loc
                                V[j*p + k][col] += cb * cs
                
                # W: C is 4x8.
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
                                k_loc = 2*bj + ej
                                k = 4*t + k_loc
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
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _schoolbook()
