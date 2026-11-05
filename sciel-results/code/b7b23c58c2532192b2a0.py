import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 120.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """
    Strassen 2x2x2 decomposition, rank 7.
    Returns U, V, W as numpy arrays of shape (4, 7).
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x8_decomposition():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    1. Uses Strassen 2x2x2 (rank 7) to build 4x4x4 (rank 49) via Kronecker product.
    2. Takes direct sum of two 4x4x4 decompositions to get 4x4x8 (rank 98).
    
    Returns U, V, W as numpy arrays.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Build 4x4x4 tensors (rank 49)
    # U4 shape: (16, 49)
    # V4 shape: (16, 49)
    # W4 shape: (16, 49)
    
    # Reshape U2, V2, W2 to 3D: (2, 2, 7)
    # U2[i, j, r] corresponds to U[i*2+j, r]
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    # Kronecker product structure:
    # U4[i*4+j, r1*7+r2] = U2[i//2, j//2, r1] * U2[i%2, j%2, r2]
    # We can use einsum for efficient computation
    # U4_5d[i1, i2, j1, j2, r1, r2] = U2_3d[i1, j1, r1] * U2_3d[i2, j2, r2]
    # Then reshape to (16, 49)
    
    U4_5d = np.einsum('ikr,jls->ijklrs', U2_3d, U2_3d)
    U4 = U4_5d.reshape(16, 49).astype(np.int16)
    
    V4_5d = np.einsum('ikr,jls->ijklrs', V2_3d, V2_3d)
    V4 = V4_5d.reshape(16, 49).astype(np.int16)
    
    W4_5d = np.einsum('ikr,jls->ijklrs', W2_3d, W2_3d)
    W4 = W4_5d.reshape(16, 49).astype(np.int16)
    
    # Build 4x4x8 tensors (rank 98)
    # U shape: (16, 98)
    # V shape: (32, 98)
    # W shape: (32, 98)
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: columns 0-48
    # U is the same for both blocks
    U[:, 0:49] = U4
    
    # V: For block 0, V[j*8+k, c] = V4[j*4+k, c] for k in 0..3
    # V4 row index is j*4+k. V row index is j*8+k.
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    
    # W: For block 0, W[k*4+i, c] = W4[k*4+i, c] for k in 0..3
    # W4 row index is k*4+i. W row index is k*4+i.
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
            
    # Block 1: columns 49-97
    U[:, 49:98] = U4
    
    # V: For block 1, V[j*8+k, c] = V4[j*4+(k-4), c] for k in 4..7
    for j in range(4):
        for k in range(4):
            V[j*8 + (k + 4), 49:98] = V4[j*4 + k, :]
    
    # W: For block 1, W[k*4+i, c] = W4[(k-4)*4+i, c] for k in 4..7
    for k in range(4):
        for i in range(4):
            W[(k + 4)*4 + i, 49:98] = W4[k*4 + i, :]
            
    return U, V, W

def _verify_decomposition(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """
    Verify the tensor decomposition on random test matrices.
    """
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    
    # Efficient verification using matrix operations
    # C_comp = sum_r (A * U_r) @ (B * V_r) * W_r
    # Where U_r, V_r, W_r are reshaped appropriately
    
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        
        Ar = A * Ur
        Br = B * Vr
        Mr = Ar @ Br
        C_comp += Mr * Wr.T
        
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    if time.time() > _DEADLINE:
        # Fallback if time is already up (should not happen)
        return {"U": np.zeros((16, 1), dtype=np.int16).tolist(), 
                "V": np.zeros((32, 1), dtype=np.int16).tolist(), 
                "W": np.zeros((32, 1), dtype=np.int16).tolist()}
        
    try:
        U, V, W = _build_4x4x8_decomposition()
        
        # Verify with a couple of seeds to ensure correctness
        ok1, err1 = _verify_decomposition(U, V, W, seed=42)
        ok2, err2 = _verify_decomposition(U, V, W, seed=123)
        
        if ok1 and ok2:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # If verification fails, return schoolbook (rank 128) as fallback
            # This is extremely unlikely to happen with the Strassen construction
            n, m, p = 4, 4, 8
            rank = n * m * p
            U_f = np.zeros((n * m, rank), dtype=np.int16)
            V_f = np.zeros((m * p, rank), dtype=np.int16)
            W_f = np.zeros((p * n, rank), dtype=np.int16)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_f[i * m + j, col] = 1
                        V_f[j * p + k, col] = 1
                        W_f[k * n + i, col] = 1
                        col += 1
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
            
    except Exception as e:
        # In case of any error, return schoolbook
        n, m, p = 4, 4, 8
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
