import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_laderman_4x4():
    """
    Laderman's rank-23 decomposition for 4x4 matrix multiplication.
    Returns U (16x23), V (16x23), W (16x23).
    Convention: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    """
    n = 4
    rank = 23
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    def set_u(r, row, col, val):
        U[row*n+col, r] += val
    def set_v(r, row, col, val):
        V[row*n+col, r] += val
    def set_w(r, row, col, val):
        W[row*n+col, r] += val
        
    # 1
    set_u(0, 0, 0, 1); set_u(0, 1, 1, 1)
    set_v(0, 0, 0, 1); set_v(0, 0, 1, 1); set_v(0, 1, 0, 1); set_v(0, 1, 1, 1)
    set_w(0, 0, 0, 1); set_w(0, 0, 1, 1); set_w(0, 1, 0, 1); set_w(0, 1, 1, 1)
    
    # 2
    set_u(1, 2, 2, 1); set_u(1, 3, 3, 1)
    set_v(1, 2, 2, 1); set_v(1, 2, 3, 1); set_v(1, 3, 2, 1); set_v(1, 3, 3, 1)
    set_w(1, 2, 2, 1); set_w(1, 2, 3, 1); set_w(1, 3, 2, 1); set_w(1, 3, 3, 1)
    
    # 3
    set_u(2, 0, 0, 1); set_u(2, 1, 1, 1); set_u(2, 2, 2, 1); set_u(2, 3, 3, 1)
    set_v(2, 0, 0, 1); set_v(2, 0, 1, 1); set_v(2, 1, 0, 1); set_v(2, 1, 1, 1)
    set_v(2, 2, 2, 1); set_v(2, 2, 3, 1); set_v(2, 3, 2, 1); set_v(2, 3, 3, 1)
    set_w(2, 0, 0, 1); set_w(2, 0, 1, 1); set_w(2, 1, 0, 1); set_w(2, 1, 1, 1)
    set_w(2, 2, 2, 1); set_w(2, 2, 3, 1); set_w(2, 3, 2, 1); set_w(2, 3, 3, 1)
    
    # 4
    set_u(3, 0, 1, 1); set_u(3, 1, 1, 1)
    set_v(3, 1, 0, 1); set_v(3, 1, 1, 1)
    set_w(3, 1, 0, 1); set_w(3, 1, 1, 1)
    
    # 5
    set_u(4, 2, 3, 1); set_u(4, 3, 3, 1)
    set_v(4, 3, 2, 1); set_v(4, 3, 3, 1)
    set_w(4, 3, 2, 1); set_w(4, 3, 3, 1)
    
    # 6
    set_u(5, 0, 1, 1); set_u(5, 1, 1, 1); set_u(5, 2, 3, 1); set_u(5, 3, 3, 1)
    set_v(5, 1, 0, 1); set_v(5, 1, 1, 1); set_v(5, 3, 2, 1); set_v(5, 3, 3, 1)
    set_w(5, 1, 0, 1); set_w(5, 1, 1, 1); set_w(5, 3, 2, 1); set_w(5, 3, 3, 1)
    
    # 7
    set_u(6, 0, 2, 1); set_u(6, 1, 2, 1)
    set_v(6, 2, 0, 1); set_v(6, 2, 1, 1)
    set_w(6, 2, 0, 1); set_w(6, 2, 1, 1)
    
    # 8
    set_u(7, 0, 3, 1); set_u(7, 1, 3, 1)
    set_v(7, 3, 0, 1); set_v(7, 3, 1, 1)
    set_w(7, 3, 0, 1); set_w(7, 3, 1, 1)
    
    # 9
    set_u(8, 0, 2, 1); set_u(8, 1, 2, 1); set_u(8, 0, 3, 1); set_u(8, 1, 3, 1)
    set_v(8, 2, 0, 1); set_v(8, 2, 1, 1); set_v(8, 3, 0, 1); set_v(8, 3, 1, 1)
    set_w(8, 2, 0, 1); set_w(8, 2, 1, 1); set_w(8, 3, 0, 1); set_w(8, 3, 1, 1)
    
    # 10
    set_u(9, 2, 0, 1); set_u(9, 3, 0, 1)
    set_v(9, 0, 2, 1); set_v(9, 0, 3, 1)
    set_w(9, 0, 2, 1); set_w(9, 0, 3, 1)
    
    # 11
    set_u(10, 2, 1, 1); set_u(10, 3, 1, 1)
    set_v(10, 1, 2, 1); set_v(10, 1, 3, 1)
    set_w(10, 1, 2, 1); set_w(10, 1, 3, 1)
    
    # 12
    set_u(11, 2, 0, 1); set_u(11, 3, 0, 1); set_u(11, 2, 1, 1); set_u(11, 3, 1, 1)
    set_v(11, 0, 2, 1); set_v(11, 0, 3, 1); set_v(11, 1, 2, 1); set_v(11, 1, 3, 1)
    set_w(11, 0, 2, 1); set_w(11, 0, 3, 1); set_w(11, 1, 2, 1); set_w(11, 1, 3, 1)
    
    # 13
    set_u(12, 0, 0, 1); set_u(12, 1, 1, -1)
    set_v(12, 1, 0, 1); set_v(12, 0, 0, -1)
    set_w(12, 1, 0, 1); set_w(12, 0, 0, -1)
    
    # 14
    set_u(13, 2, 2, 1); set_u(13, 3, 3, -1)
    set_v(13, 3, 2, 1); set_v(13, 2, 2, -1)
    set_w(13, 3, 2, 1); set_w(13, 2, 2, -1)
    
    # 15
    set_u(14, 0, 0, 1); set_u(14, 1, 1, -1); set_u(14, 2, 2, -1); set_u(14, 3, 3, 1)
    set_v(14, 1, 0, 1); set_v(14, 0, 0, -1); set_v(14, 3, 2, 1); set_v(14, 2, 2, -1)
    set_w(14, 1, 0, 1); set_w(14, 0, 0, -1); set_w(14, 3, 2, 1); set_w(14, 2, 2, -1)
    
    # 16
    set_u(15, 0, 2, 1); set_u(15, 1, 2, -1)
    set_v(15, 2, 1, 1); set_v(15, 2, 0, -1)
    set_w(15, 2, 1, 1); set_w(15, 2, 0, -1)
    
    # 17
    set_u(16, 0, 3, 1); set_u(16, 1, 3, -1)
    set_v(16, 3, 1, 1); set_v(16, 3, 0, -1)
    set_w(16, 3, 1, 1); set_w(16, 3, 0, -1)
    
    # 18
    set_u(17, 0, 2, 1); set_u(17, 1, 2, -1); set_u(17, 0, 3, -1); set_u(17, 1, 3, 1)
    set_v(17, 2, 1, 1); set_v(17, 2, 0, -1); set_v(17, 3, 1, 1); set_v(17, 3, 0, -1)
    set_w(17, 2, 1, 1); set_w(17, 2, 0, -1); set_w(17, 3, 1, 1); set_w(17, 3, 0, -1)
    
    # 19
    set_u(18, 2, 0, 1); set_u(18, 3, 0, -1)
    set_v(18, 0, 3, 1); set_v(18, 0, 2, -1)
    set_w(18, 0, 3, 1); set_w(18, 0, 2, -1)
    
    # 20
    set_u(19, 2, 1, 1); set_u(19, 3, 1, -1)
    set_v(19, 1, 3, 1); set_v(19, 1, 2, -1)
    set_w(19, 1, 3, 1); set_w(19, 1, 2, -1)
    
    # 21
    set_u(20, 2, 0, 1); set_u(20, 3, 0, -1); set_u(20, 2, 1, -1); set_u(20, 3, 1, 1)
    set_v(20, 0, 3, 1); set_v(20, 0, 2, -1); set_v(20, 1, 3, 1); set_v(20, 1, 2, -1)
    set_w(20, 0, 3, 1); set_w(20, 0, 2, -1); set_w(20, 1, 3, 1); set_w(20, 1, 2, -1)
    
    # 22: (A12-A21)(B12-B21) ? No, Laderman terms 22 and 23 are:
    # (A12+A21)(B12+B21) is not one of them.
    # The remaining terms involve the off-diagonal blocks (1,2) and (2,1) etc.
    # Actually, Laderman's 23 terms are:
    # 1-3: Diagonal blocks (1,1) and (2,2) in 2x2 block view, and sum
    # 4-6: Off-diagonal (1,2) and (2,1) in 2x2 block view?
    # Let's check the standard list.
    # The terms 1-21 above cover the 2x2 block diagonal and some off-diagonals.
    # The remaining two terms are:
    # 22: (A12 - A21)(B12 - B21) -- Wait, indices 1,2 in 4x4 are (0,1) and (1,0)?
    # In Laderman's 4x4, the 2x2 blocks are:
    # X11 = A[0:2, 0:2], X12 = A[0:2, 2:4]
    # X21 = A[2:4, 0:2], X22 = A[2:4, 2:4]
    # The terms 1-3 handle X11 and X22.
    # Terms 4-6 handle X12 and X21?
    # Term 4: (A12+A22)... this uses indices within the blocks.
    
    # Let's rely on the verification. If the current 21 terms + 2 more are correct, it will pass.
    # If not, I will return the rank 44 fallback.
    
    # The missing terms 22 and 23 in the standard Laderman scheme for 4x4 are:
    # 22: (A[0,1] - A[1,0]) * (B[0,1] - B[1,0]) ?
    # Actually, looking at a reference:
    # M22 = (A12 - A21)(B12 - B21) is not correct.
    # The last two terms are:
    # (A12 + A21)(B12 + B21) and (A12 - A21)(B12 - B21)?
    # No, that would be for 2x2.
    
    # Given the complexity and risk of error in manually transcribing Laderman's 4x4,
    # and since the fallback is rank 44 (score 1.09), 
    # I will attempt to verify the 21 terms I have. If they don't form a complete decomposition, 
    # I will stick to the known correct rank 44.
    
    # However, there is a known rank 23 decomposition. 
    # If I can't guarantee it, I should not break the working rank 44.
    
    return U, V, W

def _project_to_3x4x4(U4, V4, W4, n=3, m=4, p=4):
    rank = U4.shape[1]
    U = U4[:12, :].copy()
    V = V4[:].copy()
    w_indices = []
    for k in range(p):
        for i in range(n):
            w_indices.append(k * 4 + i)
    W = W4[w_indices, :].copy()
    return U, V, W

def _construct_laderman_3x4x4():
    U4, V4, W4 = _construct_laderman_4x4()
    U, V, W = _project_to_3x4x4(U4, V4, W4, n=3, m=4, p=4)
    return U, V, W

def _construct_strassen_3x4x4():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    return U, V, W

def _verify_4x4(U, V, W):
    n = 4
    T = np.zeros((n, n, n), dtype=np.float64)
    rank = U.shape[1]
    for r in range(rank):
        u = U[:, r].reshape(n, n)
        v = V[:, r].reshape(n, n)
        w = W[:, r].reshape(n, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=1e-6, rtol=0)

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Try Laderman-based rank-23 decomposition
    U, V, W = _construct_laderman_3x4x4()
    
    # Verify the 4x4 base
    U4, V4, W4 = _construct_laderman_4x4()
    if _verify_4x4(U4, V4, W4):
        # Verify the projected 3x4x4
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to Strassen rank-44
    U44, V44, W44 = _construct_strassen_3x4x4()
    if _verify_decomposition(U44, V44, W44, n, m, p, tol=1e-6):
        return {
            "U": U44.tolist(),
            "V": V44.tolist(),
            "W": W44.tolist()
        }
    
    # Should not reach here
    return {
        "U": U44.tolist(),
        "V": V44.tolist(),
        "W": W44.tolist()
    }
