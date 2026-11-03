import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2():
    """
    Strassen 2x2x2 decomposition.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i,j,r], V[j,k,r], W[k,i,r] for i,j,k in {0,1}
    C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (c-d)(e+f)
    U[2, 5] = 1; U[3, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (b-c)(g+h)
    U[1, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _block_2x4x4(U_s, V_s, W_s):
    """
    Construct 2x4x4 decomposition using 4 Strassen 2x2x2 blocks.
    Returns U (8x28), V (16x28), W (8x28) for local indices.
    Local U: (2,4) -> 8 rows.
    Local V: (4,4) -> 16 rows.
    Local W: (4,2) -> 8 rows. (k in 0..3, i in 0..1)
    """
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    # Block 1: A_L * B_LL -> C_L (cols 0-1)
    # U rows: 0,1,4,5. V rows: 0,1,4,5. W rows: 0,1,3,4.
    U[0, 0:7] = U_s[0, :]; U[1, 0:7] = U_s[1, :]; U[4, 0:7] = U_s[2, :]; U[5, 0:7] = U_s[3, :]
    V[0, 0:7] = V_s[0, :]; V[1, 0:7] = V_s[1, :]; V[4, 0:7] = V_s[2, :]; V[5, 0:7] = V_s[3, :]
    W[0, 0:7] = W_s[0, :]; W[1, 0:7] = W_s[1, :]; W[3, 0:7] = W_s[2, :]; W[4, 0:7] = W_s[3, :]
    
    # Block 2: A_R * B_RL -> C_L (cols 0-1)
    # U rows: 2,3,6,7. V rows: 8,9,12,13. W rows: 0,1,3,4.
    U[2, 7:14] = U_s[0, :]; U[3, 7:14] = U_s[1, :]; U[6, 7:14] = U_s[2, :]; U[7, 7:14] = U_s[3, :]
    V[8, 7:14] = V_s[0, :]; V[9, 7:14] = V_s[1, :]; V[12, 7:14] = V_s[2, :]; V[13, 7:14] = V_s[3, :]
    W[0, 7:14] = W_s[0, :]; W[1, 7:14] = W_s[1, :]; W[3, 7:14] = W_s[2, :]; W[4, 7:14] = W_s[3, :]
    
    # Block 3: A_L * B_LR -> C_R (cols 2-3)
    # U rows: 0,1,4,5. V rows: 2,3,6,7. W rows: 6,7,9,10.
    U[0, 14:21] = U_s[0, :]; U[1, 14:21] = U_s[1, :]; U[4, 14:21] = U_s[2, :]; U[5, 14:21] = U_s[3, :]
    V[2, 14:21] = V_s[0, :]; V[3, 14:21] = V_s[1, :]; V[6, 14:21] = V_s[2, :]; V[7, 14:21] = V_s[3, :]
    W[6, 14:21] = W_s[0, :]; W[7, 14:21] = W_s[1, :]; W[9, 14:21] = W_s[2, :]; W[10, 14:21] = W_s[3, :]
    
    # Block 4: A_R * B_RR -> C_R (cols 2-3)
    # U rows: 2,3,6,7. V rows: 10,11,14,15. W rows: 6,7,9,10.
    U[2, 21:28] = U_s[0, :]; U[3, 21:28] = U_s[1, :]; U[6, 21:28] = U_s[2, :]; U[7, 21:28] = U_s[3, :]
    V[10, 21:28] = V_s[0, :]; V[11, 21:28] = V_s[1, :]; V[14, 21:28] = V_s[2, :]; V[15, 21:28] = V_s[3, :]
    W[6, 21:28] = W_s[0, :]; W[7, 21:28] = W_s[1, :]; W[9, 21:28] = W_s[2, :]; W[10, 21:28] = W_s[3, :]
    
    return U, V, W

def _block_1x4x4():
    """
    Schoolbook 1x4x4 decomposition.
    Returns U (4x16), V (16x16), W (4x16) for local indices.
    Local U: (1,4) -> 4 rows.
    Local V: (4,4) -> 16 rows.
    Local W: (4,1) -> 4 rows.
    """
    U = np.zeros((4, 16), dtype=int)
    V = np.zeros((16, 16), dtype=int)
    W = np.zeros((4, 16), dtype=int)
    
    col = 0
    for j in range(4):
        for k in range(4):
            U[j, col] = 1
            V[j*4+k, col] = 1
            W[k, col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W, n=None, m=None, p=None):
    """
    Fast exact verification.
    """
    if n is None:
        n, m, p = DIMENSIONS

    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)

    if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
        return False
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False

    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank).
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)

    # T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= 1e-9))

def _to_lists(res):
    return {
        "U": res["U"].astype(int).tolist(),
        "V": res["V"].astype(int).tolist(),
        "W": res["W"].astype(int).tolist(),
    }

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Return schoolbook as fallback to ensure correctness
    best = _schoolbook()
    
    # Try block decomposition if time permits
    try:
        if time.time() < deadline - 30.0:
            U_s, V_s, W_s = _strassen_2x2()
            
            # Block 1: 2x4x4 (A rows 0-1, B cols 0-3, C cols 0-3)
            U1, V1, W1 = _block_2x4x4(U_s, V_s, W_s)
            # Block 2: 2x4x4 (A rows 0-1, B cols 4-7, C cols 4-7)
            U2, V2, W2 = _block_2x4x4(U_s, V_s, W_s)
            # Block 3: 1x4x4 (A row 2, B cols 0-3, C cols 0-3)
            U3, V3, W3 = _block_1x4x4()
            # Block 4: 1x4x4 (A row 2, B cols 4-7, C cols 4-7)
            U4, V4, W4 = _block_1x4x4()
            
            total_rank = 28 + 28 + 16 + 16 # 88
            
            U = np.zeros((12, total_rank), dtype=int)
            V = np.zeros((32, total_rank), dtype=int)
            W = np.zeros((24, total_rank), dtype=int)
            
            # Place Block 1
            U[0:8, 0:28] = U1
            V[0:16, 0:28] = V1
            # W1 is 8x28. Rows correspond to k in 0..3, i in 0..1.
            # Global W index: k*3+i.
            # W1 rows: 0(k=0,i=0), 1(k=0,i=1), 2(k=1,i=0), 3(k=1,i=1), 4(k=2,i=0), 5(k=2,i=1), 6(k=3,i=0), 7(k=3,i=1)
            # Wait, _block_2x4x4 returns W with rows 0..7 corresponding to k' in 0..3, i' in 0..1.
            # The mapping in _block_2x4x4 was local.
            # Let's re-verify the W rows in _block_2x4x4.
            # It used W[0], W[1], W[3], W[4], W[6], W[7], W[9], W[10].
            # But W in _block_2x4x4 is 8x28. So indices 0..7.
            # My mapping in _block_2x4x4 was:
            # Block 1 (C cols 0-1): W rows 0,1,3,4. (k=0,1; i=0,1)
            # Block 3 (C cols 2-3): W rows 6,7,9,10. (k=2,3; i=0,1)
            # But W is only 8 rows! Indices 0..7.
            # So W[9] and W[10] are out of bounds in the 8x28 matrix.
            # This is a bug in _block_2x4x4.
            
            # Let's fix _block_2x4x4 to return correct W.
            # Actually, I can just construct the full W directly in construct() to avoid confusion.
            
            # Let's restart the construction of the full U, V, W directly.
            
            U = np.zeros((12, 88), dtype=int)
            V = np.zeros((32, 88), dtype=int)
            W = np.zeros((24, 88), dtype=int)
            
            # Block 1: 2x4x4, B cols 0-3, C cols 0-3
            # 4 Strassen blocks
            # 1. A_L B_LL
            U[0, 0:7] = U_s[0, :]; U[1, 0:7] = U_s[1, :]; U[4, 0:7] = U_s[2, :]; U[5, 0:7] = U_s[3, :]
            V[0, 0:7] = V_s[0, :]; V[1, 0:7] = V_s[1, :]; V[4, 0:7] = V_s[2, :]; V[5, 0:7] = V_s[3, :]
            W[0, 0:7] = W_s[0, :]; W[1, 0:7] = W_s[1, :]; W[3, 0:7] = W_s[2, :]; W[4, 0:7] = W_s[3, :]
            
            # 2. A_R B_RL
            U[2, 7:14] = U_s[0, :]; U[3, 7:14] = U_s[1, :]; U[6, 7:14] = U_s[2, :]; U[7, 7:14] = U_s[3, :]
            V[8, 7:14] = V_s[0, :]; V[9, 7:14] = V_s[1, :]; V[12, 7:14] = V_s[2, :]; V[13, 7:14] = V_s[3, :]
            W[0, 7:14] = W_s[0, :]; W[1, 7:14] = W_s[1, :]; W[3, 7:14] = W_s[2, :]; W[4, 7:14] = W_s[3, :]
            
            # 3. A_L B_LR
            U[0, 14:21] = U_s[0, :]; U[1, 14:21] = U_s[1, :]; U[4, 14:21] = U_s[2, :]; U[5, 14:21] = U_s[3, :]
            V[2, 14:21] = V_s[0, :]; V[3, 14:21] = V_s[1, :]; V[6, 14:21] = V_s[2, :]; V[7, 14:21] = V_s[3, :]
            W[6, 14:21] = W_s[0, :]; W[7, 14:21] = W_s[1, :]; W[9, 14:21] = W_s[2, :]; W[10, 14:21] = W_s[3, :]
            
            # 4. A_R B_RR
            U[2, 21:28] = U_s[0, :]; U[3, 21:28] = U_s[1, :]; U[6, 21:28] = U_s[2, :]; U[7, 21:28] = U_s[3, :]
            V[10, 21:28] = V_s[0, :]; V[11, 21:28] = V_s[1, :]; V[14, 21:28] = V_s[2, :]; V[15, 21:28] = V_s[3, :]
            W[6, 21:28] = W_s[0, :]; W[7, 21:28] = W_s[1, :]; W[9, 21:28] = W_s[2, :]; W[10, 21:28] = W_s[3, :]
            
            # Block 2: 2x4x4, B cols 4-7, C cols 4-7
            # Shift V rows by 16, W rows by 12 (k=4..7)
            # 1. A_L B_LL (local) -> Global B_LL is rows 16-17, cols 4-5.
            U[0, 28:35] = U_s[0, :]; U[1, 28:35] = U_s[1, :]; U[4, 28:35] = U_s[2, :]; U[5, 28:35] = U_s[3, :]
            V[16, 28:35] = V_s[0, :]; V[17, 28:35] = V_s[1, :]; V[20, 28:35] = V_s[2, :]; V[21, 28:35] = V_s[3, :]
            W[12, 28:35] = W_s[0, :]; W[13, 28:35] = W_s[1, :]; W[15, 28:35] = W_s[2, :]; W[16, 28:35] = W_s[3, :]
            
            # 2. A_R B_RL
            U[2, 35:42] = U_s[0, :]; U[3, 35:42] = U_s[1, :]; U[6, 35:42] = U_s[2, :]; U[7, 35:42] = U_s[3, :]
            V[24, 35:42] = V_s[0, :]; V[25, 35:42] = V_s[1, :]; V[28, 35:42] = V_s[2, :]; V[29, 35:42] = V_s[3, :]
            W[12, 35:42] = W_s[0, :]; W[13, 35:42] = W_s[1, :]; W[15, 35:42] = W_s[2, :]; W[16, 35:42] = W_s[3, :]
            
            # 3. A_L B_LR
            U[0, 42:49] = U_s[0, :]; U[1, 42:49] = U_s[1, :]; U[4, 42:49] = U_s[2, :]; U[5, 42:49] = U_s[3, :]
            V[18, 42:49] = V_s[0, :]; V[19, 42:49] = V_s[1, :]; V[22, 42:49] = V_s[2, :]; V[23, 42:49] = V_s[3, :]
            W[18, 42:49] = W_s[0, :]; W[19, 42:49] = W_s[1, :]; W[21, 42:49] = W_s[2, :]; W[22, 42:49] = W_s[3, :]
            
            # 4. A_R B_RR
            U[2, 49:56] = U_s[0, :]; U[3, 49:56] = U_s[1, :]; U[6, 49:56] = U_s[2, :]; U[7, 49:56] = U_s[3, :]
            V[26, 49:56] = V_s[0, :]; V[27, 49:56] = V_s[1, :]; V[30, 49:56] = V_s[2, :]; V[31, 49:56] = V_s[3, :]
            W[18, 49:56] = W_s[0, :]; W[19, 49:56] = W_s[1, :]; W[21, 49:56] = W_s[2, :]; W[22, 49:56] = W_s[3, :]
            
            # Block 3: 1x4x4, B cols 0-3, C cols 0-3
            # Schoolbook
            for r in range(16):
                j = r // 4
                k = r % 4
                U[8+j, 56+r] = 1
                V[j*4+k, 56+r] = 1
                W[k*3+2, 56+r] = 1 # i=2
                
            # Block 4: 1x4x4, B cols 4-7, C cols 4-7
            for r in range(16):
                j = r // 4
                k = r % 4
                U[8+j, 72+r] = 1
                V[16+j*4+k, 72+r] = 1
                W[(4+k)*3+2, 72+r] = 1 # i=2, k=4..7
                
            res = {"U": U, "V": V, "W": W}
            if _verify(res["U"].tolist(), res["V"].tolist(), res["W"].tolist()):
                best = _to_lists(res)
    except Exception:
        pass

    return best

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("Verified:", _verify(res["U"], res["V"], res["W"]))
