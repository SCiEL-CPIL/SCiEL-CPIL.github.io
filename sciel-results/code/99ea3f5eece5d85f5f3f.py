import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's algorithm on 7 independent 2x2 blocks.
    Includes verification and fallback.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Format: (A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11)
    # Note: B indices are local (0,0), (0,1), (1,0), (1,1)
    # Note: C indices are local (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11)
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # M2 = (A10 + A11) * B00
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        # M3 = A00 * (B01 - B11)
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        # M4 = A11 * (B10 - B00)
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        # M5 = (A00 + A01) * B11
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # M6 = (A10 - A00) * (B00 + B01)
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        # M7 = (A01 - A11) * (B10 + B11)
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]
    
    for b in range(num_blocks):
        for s in range(7):
            col = b * 7 + s
            a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11 = strassen_terms[s]
            
            # Fill U (A coefficients)
            U[0][col] = a00
            U[1][col] = a01
            U[2][col] = a10
            U[3][col] = a11
            
            # Fill V (B coefficients)
            # Global B indices:
            # B[0, 2b] -> row 0*p + 2b = 2b
            # B[0, 2b+1] -> row 0*p + 2b+1 = 2b+1
            # B[1, 2b] -> row 1*p + 2b = p + 2b
            # B[1, 2b+1] -> row 1*p + 2b+1 = p + 2b+1
            
            if b00 != 0: V[2*b][col] = b00
            if b01 != 0: V[2*b+1][col] = b01
            if b10 != 0: V[p + 2*b][col] = b10
            if b11 != 0: V[p + 2*b+1][col] = b11
            
            # Fill W (C coefficients)
            # Global C indices:
            # C[0, 2b] -> row 2b*n + 0 = 4b
            # C[1, 2b] -> row 2b*n + 1 = 4b+1
            # C[0, 2b+1] -> row (2b+1)*n + 0 = 4b+2
            # C[1, 2b+1] -> row (2b+1)*n + 1 = 4b+3
            
            if c00 != 0: W[4*b][col] = c00
            if c01 != 0: W[4*b+1][col] = c01
            if c10 != 0: W[4*b+2][col] = c10
            if c11 != 0: W[4*b+3][col] = c11
            
            if time.time() > deadline:
                # Fallback to schoolbook if we run out of time (unlikely)
                return _construct_schoolbook(n, m, p)
                
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback if verification fails
        return _construct_schoolbook(n, m, p)

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    """
    rank = len(U[0])
    # Use numpy for speed in verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape to tensors
    # U: (n*m, rank) -> (n, m, rank)
    U_t = U_np.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_t = V_np.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_t = W_np.reshape(p, n, rank)
    
    # Compute tensor product sum
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a contraction over r.
    # We can do this via einsum or simple loops if small.
    # n=2, m=2, p=14, rank=49.
    
    # T[i,j,k]
    # Let's just check a few random indices or all indices (small enough)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
