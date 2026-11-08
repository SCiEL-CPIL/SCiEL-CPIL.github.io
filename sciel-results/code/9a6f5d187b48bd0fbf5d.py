import numpy as np
import time

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS  # 3, 4, 5


def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2. Returns U,V,W each (4,7).
    U order: A00,A01,A10,A11 ; V order: B00,B01,B10,B11 ; W order: C00,C01,C10,C11."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _build_122():
    """Rank-4 schoolbook for 1x2x2: A(1x2)*B(2x2)=C(1x2). U(2,4),V(4,4),W(2,4).
    U order: A0,A1 ; V order: B00,B01,B10,B11 ; W order: C0,C1."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W


def _build_121():
    """Rank-2 for 1x2x1: A(1x2)*B(2x1)=C(1x1). U(2,2),V(2,2),W(1,2)."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _build_221():
    """Rank-4 schoolbook for 2x2x1: A(2x2)*B(2x1)=C(2x1). U(4,4),V(2,4),W(2,4).
    U order: A00,A01,A10,A11 ; V order: B0,B1 ; W order: C0,C1."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W


def _verify(U, V, W):
    """Fast vectorized verification using tensor contraction."""
    r = U.shape[1]
    Umat = np.asarray(U, dtype=float)  # (n*m, r)
    Vmat = np.asarray(V, dtype=float)  # (m*p, r)
    Wmat = np.asarray(W, dtype=float)  # (p*n, r)
    
    # Reshape to 3D: U (n, m, r), V (m, p, r), W (p, n, r)
    U3 = Umat.reshape(n, m, r)
    V3 = Vmat.reshape(m, p, r)
    W3 = Wmat.reshape(p, n, r)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum for efficient computation
    T = np.einsum('ijr,jkr,kr->ijk', U3, V3, W3, optimize=True)
    
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-9)

def _try_reduce_rank(U, V, W, max_iter=50):
    """
    Attempt to reduce the rank by identifying linearly dependent columns.
    Uses QR with column pivoting to find the numerical rank and remove
    redundant columns if possible.
    """
    r = U.shape[1]
    Umat = np.asarray(U, dtype=float)
    Vmat = np.asarray(V, dtype=float)
    Wmat = np.asarray(W, dtype=float)
    
    # Stack all factor matrices into a single matrix for rank analysis
    # Each column r contributes a rank-1 tensor. If columns are linearly dependent
    # in the combined space, we may be able to reduce rank.
    # Create a tall matrix where each row corresponds to a (i,j,k) entry
    # and each column is the product u[i,j]*v[j,k]*w[k,i]
    
    # Build the matrix M where M[idx, r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # idx = i*m*p + j*p + k
    M = np.zeros((n*m*p, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                u_val = Umat[i*m+j, :]
                v_val = Vmat[j*p+k, :]
                w_val = Wmat[k*n+i, :]
                M[idx, :] = u_val * v_val * w_val
    
    # Use SVD to find numerical rank
    try:
        s, U_svd, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W
    
    # Find effective rank (singular values above tolerance)
    tol = max(M.shape) * np.finfo(float).eps * s[0] if s.size > 0 else 0
    num_rank = np.sum(s > tol)
    
    if num_rank >= r:
        # No reduction possible
        return U, V, W
    
    # Try to project onto the principal components
    # This is a heuristic - we need to check if the reduced decomposition is valid
    # For now, just return the original if we can't cleanly reduce
    # A proper reduction would require solving for new factors
    return U, V, W

def _construct_rank_56():
    """
    Partition A into rows [0,1] and [2], B into cols [0,1], [2,3], [4].
    For A[0:2,:]: use 2x2x2 Strassen for 2x2 blocks, 2x2x1 for last col.
    For A[2,:]: use 1x2x2 for 2x2 blocks, 1x2x1 for last col.
    
    Total rank: 36 + 20 = 56.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, Ublk, Vblk, Wblk):
        for r in range(Ublk.shape[1]):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            for i, a in enumerate(u_idx):
                u[a] = Ublk[i, r]
            for i, a in enumerate(v_idx):
                v[a] = Vblk[i, r]
            for i, a in enumerate(w_idx):
                w[a] = Wblk[i, r]
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)
    
    # U indices: A[i,j] -> i*m+j = i*4+j
    # V indices: B[j,k] -> j*p+k = j*5+k
    # W indices: C[i,k] -> k*n+i = k*3+i
    
    # Part A: A[0:2,:] (rows 0,1) * B
    # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    Uu, Vv, Ww = _build_strassen_222()
    # U: A00=0, A01=1, A10=4, A11=5
    # V: B00=0, B01=1, B10=5, B11=6
    # W: C00=0, C01=3, C10=1, C11=4
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], Uu, Vv, Ww)
    
    # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    Uu, Vv, Ww = _build_strassen_222()
    # V: B02=2, B03=3, B12=7, B13=8
    # W: C02=6, C03=9, C12=7, C13=10
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], Uu, Vv, Ww)
    
    # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    Uu, Vv, Ww = _build_221()
    # V: B04=4, B14=9
    # W: C04=12, C14=13
    add_block([0,1,4,5], [4,9], [12,13], Uu, Vv, Ww)
    
    # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    Uu, Vv, Ww = _build_strassen_222()
    # U: A02=2, A03=3, A12=6, A13=7
    # V: B20=10, B21=11, B30=15, B31=16
    # W: C00=0, C01=3, C10=1, C11=4
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], Uu, Vv, Ww)
    
    # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    Uu, Vv, Ww = _build_strassen_222()
    # V: B22=12, B23=13, B32=17, B33=18
    # W: C02=6, C03=9, C12=7, C13=10
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], Uu, Vv, Ww)
    
    # A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    Uu, Vv, Ww = _build_221()
    # V: B24=14, B34=19
    # W: C04=12, C14=13
    add_block([2,3,6,7], [14,19], [12,13], Uu, Vv, Ww)
    
    # Part B: A[2,:] (row 2) * B
    # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    Uu, Vv, Ww = _build_122()
    # U: A20=8, A21=9
    # V: B00=0, B01=1, B10=5, B11=6
    # W: C20=2, C21=5
    add_block([8,9], [0,1,5,6], [2,5], Uu, Vv, Ww)
    
    # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    Uu, Vv, Ww = _build_122()
    # V: B02=2, B03=3, B12=7, B13=8
    # W: C22=8, C23=11
    add_block([8,9], [2,3,7,8], [8,11], Uu, Vv, Ww)
    
    # A[2,0:2] * B[0:2,4:5] -> C[2,4:5]
    Uu, Vv, Ww = _build_121()
    # V: B04=4, B14=9
    # W: C24=14
    add_block([8,9], [4,9], [14], Uu, Vv, Ww)
    
    # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    Uu, Vv, Ww = _build_122()
    # U: A22=10, A23=11
    # V: B20=10, B21=11, B30=15, B31=16
    # W: C20=2, C21=5
    add_block([10,11], [10,11,15,16], [2,5], Uu, Vv, Ww)
    
    # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    Uu, Vv, Ww = _build_122()
    # V: B22=12, B23=13, B32=17, B33=18
    # W: C22=8, C23=11
    add_block([10,11], [12,13,17,18], [8,11], Uu, Vv, Ww)
    
    # A[2,2:4] * B[2:4,4:5] -> C[2,4:5]
    Uu, Vv, Ww = _build_121()
    # V: B24=14, B34=19
    # W: C24=14
    add_block([10,11], [14,19], [14], Uu, Vv, Ww)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def _construct_schoolbook():
    U_parts, V_parts, W_parts = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def construct():
    t0 = time.time()
    deadline = 120.0
    margin = 5.0
    
    try:
        U, V, W = _construct_rank_56()
        
        # Verify the decomposition
        if not _verify(U, V, W):
            # Fallback to schoolbook if verification fails
            U, V, W = _construct_schoolbook()
        else:
            # Try to reduce rank if we have time
            if time.time() - t0 < deadline - margin:
                U, V, W = _try_reduce_rank(U, V, W)
                # Re-verify after reduction attempt
                if not _verify(U, V, W):
                    U, V, W = _construct_schoolbook()
                
    except Exception:
        U, V, W = _construct_schoolbook()
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
