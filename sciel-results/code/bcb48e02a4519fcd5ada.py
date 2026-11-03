import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-10)

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
    # A1 is A[:, 0]
    # B1 is B[0, :]
    # C_part1[i, k] = A[i, 0] * B[0, k]
    for i in range(n):
        for k in range(p):
            # U index: i*m + 0
            # V index: 0*p + k
            # W index: k*n + i
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
    # A2 is A[:, 1:3]
    # B2 is B[1:3, :]
    # Split B2 into 4 blocks of 2x2
    
    # Strassen decomposition for 2x2 * 2x2
    # A2 local indices:
    # a00 = A[0,1] -> global 0*m + 1 = 1
    # a01 = A[0,2] -> global 0*m + 2 = 2
    # a10 = A[1,1] -> global 1*m + 1 = 4
    # a11 = A[1,2] -> global 1*m + 2 = 5
    
    for block in range(4):
        # B block columns: block*2, block*2+1
        # B rows: 1, 2
        # b00 = B[1, block*2] -> index 1*p + block*2
        # b01 = B[1, block*2+1] -> index 1*p + block*2+1
        # b10 = B[2, block*2] -> index 2*p + block*2
        # b11 = B[2, block*2+1] -> index 2*p + block*2+1
        
        # Outputs C:
        # c00 -> C[0, block*2] -> W index (block*2)*n + 0
        # c01 -> C[0, block*2+1] -> W index (block*2+1)*n + 0
        # c10 -> C[1, block*2] -> W index (block*2)*n + 1
        # c11 -> C[1, block*2+1] -> W index (block*2+1)*n + 1
        
        # Strassen steps:
        # 1. (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1 * p + block * 2][col] = 1
        V[2 * p + block * 2 + 1][col] = 1
        # Contribution to c00 (+), c11 (+)
        W[(block * 2) * n + 0][col] = 1
        W[(block * 2 + 1) * n + 1][col] = 1
        col += 1
        
        # 2. (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1 * p + block * 2][col] = 1
        # Contribution to c10 (+), c11 (-)
        W[(block * 2) * n + 1][col] = 1
        W[(block * 2 + 1) * n + 1][col] = -1
        col += 1
        
        # 3. a00(b01-b11)
        U[1][col] = 1
        V[1 * p + block * 2 + 1][col] = 1
        V[2 * p + block * 2 + 1][col] = -1
        # Contribution to c01 (+), c11 (+)
        W[(block * 2 + 1) * n + 0][col] = 1
        W[(block * 2 + 1) * n + 1][col] = 1
        col += 1
        
        # 4. a11(b10-b00)
        U[5][col] = 1
        V[2 * p + block * 2][col] = 1
        V[1 * p + block * 2][col] = -1
        # Contribution to c00 (+), c10 (+)
        W[(block * 2) * n + 0][col] = 1
        W[(block * 2) * n + 1][col] = 1
        col += 1
        
        # 5. (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2 * p + block * 2 + 1][col] = 1
        # Contribution to c00 (-), c01 (+)
        W[(block * 2) * n + 0][col] = -1
        W[(block * 2 + 1) * n + 0][col] = 1
        col += 1
        
        # 6. (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1 * p + block * 2][col] = 1
        V[1 * p + block * 2 + 1][col] = 1
        # Contribution to c11 (+)
        W[(block * 2 + 1) * n + 1][col] = 1
        col += 1
        
        # 7. (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2 * p + block * 2][col] = 1
        V[2 * p + block * 2 + 1][col] = 1
        # Contribution to c00 (+)
        W[(block * 2) * n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    n, m, p = 2, 3, 8
    U, V, W = result["U"], result["V"], result["W"]
    print(f"Rank: {len(U[0])}")
    print(f"Verified: {_verify_decomposition(U, V, W, n, m, p)}")
