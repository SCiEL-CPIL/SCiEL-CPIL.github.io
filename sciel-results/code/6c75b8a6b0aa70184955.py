import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check if the decomposition exactly matches the tensor."""
    T_comp = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_comp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_comp, T, atol=tol)


def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_parts, V_parts, W_parts = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m); u[i * m + j] = 1
                v = np.zeros(m * p); v[j * p + k] = 1
                w = np.zeros(p * n); w[k * n + i] = 1
                U_parts.append(u); V_parts.append(v); W_parts.append(w)
    return {"U": np.array(U_parts).T.tolist(),
            "V": np.array(V_parts).T.tolist(),
            "W": np.array(W_parts).T.tolist()}


def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matmul, rank 7."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _schoolbook_2x2x1():
    """Schoolbook for 2x2x1, rank 4."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W


def _schoolbook_1x2x2():
    """Schoolbook for 1x2x2, rank 4."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W


def _schoolbook_1x2x1():
    """Schoolbook for 1x2x1, rank 2."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _block_decomposition(n, m, p):
    """
    Build a block-structured decomposition.
    Partition A's rows into blocks, B's columns into blocks.
    Use Strassen for 2x2x2 blocks, schoolbook otherwise.
    Returns (U, V, W, rank).
    """
    U_parts, V_parts, W_parts = [], [], []
    
    # Row blocks of A (size n=3): [0,1], [2], [3] -> but n=3, so rows 0,1,2
    # Actually A is n x m = 3 x 4. Rows indexed 0..2.
    # B is m x p = 4 x 5. Cols indexed 0..4.
    # We partition A rows into: [0,1] and [2]
    # We partition B cols into: [0,1], [2,3], [4]
    
    row_blocks = [(0, 1), (2, 2)]  # (start, end) inclusive-ish, or (start, count)
    col_blocks = [(0, 2), (2, 2), (4, 1)]  # (start, count)
    
    for (r_start, r_count) in row_blocks:
        for (c_start, c_count) in col_blocks:
            # U indices: A rows r_start..r_start+r_count-1, all m columns
            u_idx = list(range(r_start * m, (r_start + r_count) * m))
            # V indices: B cols c_start..c_start+c_count-1, all m rows
            v_idx = list(range(c_start, m * p))  # Wait, B is m x p, indexed by (j, k) -> j*p+k
            # Actually v_idx should be: for each j in 0..m-1, for each k in c_start..c_start+c_count-1
            v_idx = [j * p + k for j in range(m) for k in range(c_start, c_start + c_count)]
            # W indices: C rows r_start..r_start+r_count-1, cols c_start..c_start+c_count-1
            # C indexed by (i, k) -> k*n + i
            w_idx = [k * n + i for i in range(r_start, r_start + r_count) for k in range(c_start, c_start + c_count)]
            
            # Get block decomposition
            if r_count == 2 and c_count == 2:
                U_blk, V_blk, W_blk = _strassen_2x2x2()
            elif r_count == 2 and c_count == 1:
                U_blk, V_blk, W_blk = _schoolbook_2x2x1()
            elif r_count == 1 and c_count == 2:
                U_blk, V_blk, W_blk = _schoolbook_1x2x2()
            else:
                U_blk, V_blk, W_blk = _schoolbook_1x2x1()
            
            rank = U_blk.shape[1]
            for r in range(rank):
                u = np.zeros(n * m)
                for i, idx in enumerate(u_idx):
                    u[idx] = U_blk[i, r]
                v = np.zeros(m * p)
                for i, idx in enumerate(v_idx):
                    v[idx] = V_blk[i, r]
                w = np.zeros(p * n)
                for i, idx in enumerate(w_idx):
                    w[idx] = W_blk[i, r]
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W, U.shape[1]


def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses block-structured decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)
    
    # Build block decomposition
    U, V, W, rank = _block_decomposition(n, m, p)
    
    # Verify
    if time.time() < _DEADLINE - 5:
        T = _build_tensor(n, m, p)
        if not _verify_decomposition(U, V, W, T):
            return _schoolbook_fallback(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
