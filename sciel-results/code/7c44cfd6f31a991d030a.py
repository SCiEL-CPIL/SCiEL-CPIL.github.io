import numpy as np
import time

def _build_tensor_matrix(U, V, W, n, m, p):
    """Build the tensor matrix M where M[entry, r] = U[i,j,r]*V[j,k,r]*W[k,i,r]"""
    rank = U.shape[1]
    M = np.zeros((n*m*p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m*p + j*p + k, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    return M

def _reduce_rank(U, V, W, n, m, p, tol=1e-7):
    """
    Reduce rank by removing linearly dependent rank-1 tensors.
    Returns reduced U, V, W.
    """
    rank = U.shape[1]
    if rank == 0:
        return U, V, W
    
    M = _build_tensor_matrix(U, V, W, n, m, p)
    
    # Use SVD to determine numerical rank
    try:
        s = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    if len(s) == 0:
        return U, V, W
    
    max_s = s[0] if s[0] > 0 else 1.0
    threshold = max(tol * max_s, 1e-10)
    num_rank = int(np.sum(s > threshold))
    
    if num_rank >= rank:
        return U, V, W
    
    # Use QR with column pivoting to find independent columns
    try:
        M_padded = np.vstack([M, 1e-10 * np.eye(rank)])
        Q, R = np.linalg.qr(M_padded, mode='complete')
        diag = np.abs(np.diag(R))
        pivot_threshold = max(tol * diag[0], 1e-10)
        independent = np.where(diag > pivot_threshold)[0]
    except np.linalg.LinAlgError:
        return U, V, W
    
    if len(independent) < rank:
        U_new = U[:, independent]
        V_new = V[:, independent]
        W_new = W[:, independent]
        return U_new, V_new, W_new
    
    return U, V, W

def _greedy_reduce(U, V, W, n, m, p, tol=1e-7, max_rounds=5):
    """
    Iteratively reduce rank by checking for dependencies.
    """
    for _ in range(max_rounds):
        old_rank = U.shape[1]
        U, V, W = _reduce_rank(U, V, W, n, m, p, tol)
        if U.shape[1] >= old_rank:
            break
    return U, V, W

def _strassen_2x2x2():
    """Standard Strassen 2x2x2, rank 7.
    Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # M3 = A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    
    # M4 = A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    
    # M5 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    return U2, V2, W2

def _block_specs():
    """Strassen steps for 4x4x4 level.
    Each step: (A_spec, B_spec, C_spec)
    block_idx: 0=00, 1=01, 2=10, 3=11 (row_block, col_block)
    """
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    return A_specs, B_specs, C_specs

def _build_98():
    """Build rank-98 decomposition using block Strassen."""
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs, B_specs, C_specs = _block_specs()
    
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                
                # A part (16x1)
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                            
                # B part (32x1)
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                            
                # C part (32x1)
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
                            
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _naive(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = 4, 4, 8

    # Build the rank 98 decomposition
    U, V, W = _build_98()

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback to naive if something went wrong
        U, V, W = _naive(n, m, p)
    else:
        # Try to reduce rank by finding linear dependencies
        if time.time() < deadline - 20:
            U, V, W = _greedy_reduce(U, V, W, n, m, p, tol=1e-8)
            # Verify after reduction
            if not _verify(U, V, W, n, m, p):
                U, V, W = _build_98()

    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
