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
            # U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            
            # V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            
            # W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    # Step 3: Construct 4x4x8 tensors
    # Dimensions: n=4, m=4, p=8
    # U shape: (n*m, rank) = (16, 98)
    # V shape: (m*p, rank) = (32, 98)
    # W shape: (p*n, rank) = (32, 98)
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # We split the output dimension p=8 into two blocks of size 4.
    # Block 0: output columns 0-3
    # Block 1: output columns 4-7
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        
        # U is identical for both blocks (A is the same 4x4 matrix)
        U[:, col_start:col_end] = U4
        
        # V: For block 'block', the relevant part of B is B[:, block*4 : (block+1)*4]
        # V4 is for 4x4x4, so V4[j*4+k, c] corresponds to B_block[j,k]
        # In the full 4x8 context, V[j*8+k, c] where k is global index.
        # For block 0, k_global = 0..3, so V[j*8+k, c] = V4[j*4+k, c]
        # For block 1, k_global = 4..7, so V[j*8+k, c] = V4[j*4+(k-4), c]
        
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        
        # W: Similar logic. W4[k*4+i, c] corresponds to C_block[k,i]
        # In full context, W[k_global*4+i, c]
        # For block 0, k_global = 0..3, W[k*4+i, c] = W4[k*4+i, c]
        # For block 1, k_global = 4..7, W[k*4+i, c] = W4[(k-4)*4+i, c]
        
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    # Step 4: Verification
    # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # n=4, m=4, p=8
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return None # Decomposition failed
    
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
    # Try the rank-98 block-Strassen decomposition
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    
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
                if s != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
