import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 130.0   # safety margin under 150s kill

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the tensor product of Strassen's 2x2x2 decomposition to create a
    rank-49 decomposition for 4x4x4, then duplicates it for the two 4x4 blocks
    of the 4x8 output.
    """
    # Step 1: Get 2x2x2 Strassen tensors
    U2, V2, W2 = _build_strassen_2x2x2()
    
    # Step 2: Construct 4x4x4 tensors via tensor product (Kronecker product)
    # Rank is 7 * 7 = 49
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    # Step 3: Construct 4x4x8 tensors
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    # Step 4: Vectorized verification
    n, m, p = 4, 4, 8
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(total_rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n=4, m=4, p=8):
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

def _verify(U, V, W, n=4, m=4, p=8, atol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U[:,r].reshape(n,m), V[:,r].reshape(m,p), W[:,r].reshape(p,n))
    return np.allclose(result, 1.0, atol=atol)

def _optimize_rank(target_rank, n=4, m=4, p=8, max_iter=1000, seed=0):
    """Optimize for a decomposition of given target rank using ALS."""
    np.random.seed(seed)
    
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        
        # Alternating least squares
        for i in range(n):
            for j in range(m):
                A = np.array([V[j*p+k,:] * W[k*n+i,:] for k in range(p)])
                try:
                    U[i*m+j,:] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except: pass
        
        for j in range(m):
            for k in range(p):
                A = np.array([U[i*m+j,:] * W[k*n+i,:] for i in range(n)])
                try:
                    V[j*p+k,:] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except: pass
        
        for k in range(p):
            for i in range(n):
                A = np.array([U[i*m+j,:] * V[j*p+k,:] for j in range(m)])
                try:
                    W[k*n+i,:] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except: pass
        
        # Compute error efficiently
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                    err += (val - 1.0)**2
        err = np.sqrt(err/(n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            break
    
    if best_err < 1e-6:
        # Try rounding to half-integers or quarters
        for scale in [2, 4, 8]:
            Ur = np.round(best_U * scale) / scale
            Vr = np.round(best_V * scale) / scale
            Wr = np.round(best_W * scale) / scale
            if _verify(Ur, Vr, Wr):
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    
    return None

def construct():
    # Get the reliable rank-98 solution first
    rank98_result = None
    try:
        if time.time() < DEADLINE:
            rank98_result = _build_rank98()
    except Exception:
        pass
    
    # Try to find lower-rank decompositions using optimization
    if rank98_result is not None:
        for target in [90, 85, 80, 75, 70, 65, 60]:
            if time.time() > DEADLINE - 15:
                break
            for seed in [0, 1, 2, 3]:
                if time.time() > DEADLINE - 15:
                    break
                try:
                    res = _optimize_rank(target, max_iter=400, seed=seed)
                    if res is not None:
                        return res
                except Exception:
                    pass
    
    if rank98_result is not None:
        return rank98_result
    
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    # Verify manually if needed
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
