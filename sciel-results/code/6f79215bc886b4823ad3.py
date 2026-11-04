import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 140.0  # Safety margin under 150s kill

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
    
    # Step 2: Construct 4x4x4 tensors via Kronecker product structure
    # For each pair (c0, c1) in 7x7, the rank-49 term is the outer product
    # of the two rank-7 terms. We can build this using broadcasting.
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    # Precompute index mappings for 4x4 from 2x2 components
    # i = 2*i0 + i1, j = 2*j0 + j1
    # U4[i*4+j, c0*7+c1] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
    
    # Create index arrays for vectorized computation
    # For U: index i*4+j where i=2*i0+i1, j=2*j0+j1
    # So the linear index is (2*i0+i1)*4 + (2*j0+j1) = 8*i0*j0 + 4*i0 + 8*j0... 
    # Let's just use the loop but with numpy operations inside
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            # U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            # i ranges 0..3, j ranges 0..3
            # Let's vectorize over i,j
            i_vals = np.arange(4)
            j_vals = np.arange(4)
            I, J = np.meshgrid(i_vals, j_vals, indexing='ij')
            i0 = I // 2
            i1 = I % 2
            j0 = J // 2
            j1 = J % 2
            # U2 indices: (i0*2+j0, c0) and (i1*2+j1, c1)
            idx1 = i0 * 2 + j0
            idx2 = i1 * 2 + j1
            U4_vals = U2[idx1, c0] * U2[idx2, c1]
            U4[I * 4 + J, c] = U4_vals
            
            # V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            J2, K = np.meshgrid(i_vals, j_vals, indexing='ij')
            j0v = J2 // 2
            j1v = J2 % 2
            k0v = K // 2
            k1v = K % 2
            idx1v = j0v * 2 + k0v
            idx2v = j1v * 2 + k1v
            V4_vals = V2[idx1v, c0] * V2[idx2v, c1]
            V4[J2 * 4 + K, c] = V4_vals
            
            # W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
            K2, I2 = np.meshgrid(i_vals, j_vals, indexing='ij')
            k0w = K2 // 2
            k1w = K2 % 2
            i0w = I2 // 2
            i1w = I2 % 2
            idx1w = k0w * 2 + i0w
            idx2w = k1w * 2 + i1w
            W4_vals = W2[idx1w, c0] * W2[idx2w, c1]
            W4[K2 * 4 + I2, c] = W4_vals
    
    # Step 3: Construct 4x4x8 tensors
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: columns 0-3 of output (p dimension)
    U[:, 0:49] = U4
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local, 0:49] = V4[j*4 + k_local, :]
            W[k_local*4, 0:49] = W4[k_local*4, :]  # This is wrong, let me fix
    
    # Let me redo this more carefully
    # W[k*n+i, r] where n=4, so W index is k*4+i
    # For block 0, k_global = k_local (0..3)
    # For block 1, k_global = 4 + k_local (4..7)
    
    # Reset and do it properly
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
                for i in range(4):
                    W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    # Step 4: Verification using vectorized einsum
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

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
