"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms: sum(U[a,r] * V[b,r] * W[c,r]) should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms: sum(U[a,r] * V[b,r] * W[c,r]) should be 0
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.add((i * m + j, j * p + k, k * n + i))
                
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in valid_indices:
                    continue
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_block(n, m, p):
    """
    Decomposition into 8 independent 2x2x2 multiplications using Strassen's algorithm.
    Rank = 8 * 7 = 56.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # Indices for block t:
        # B columns: 2t, 2t+1
        # C columns: 2t, 2t+1
        # V indices:
        # B[0, 2t] -> V[0 * p + 2t] = V[2t]
        # B[0, 2t+1] -> V[0 * p + 2t+1] = V[2t+1]
        # B[1, 2t] -> V[1 * p + 2t] = V[16 + 2t]
        # B[1, 2t+1] -> V[1 * p + 2t+1] = V[16 + 2t + 1]
        
        # W indices:
        # C[0, 2t] -> W[2t * n + 0] = W[4t]
        # C[1, 2t] -> W[2t * n + 1] = W[4t+1]
        # C[0, 2t+1] -> W[(2t+1) * n + 0] = W[4t+2]
        # C[1, 2t+1] -> W[(2t+1) * n + 1] = W[4t+3]
        
        # U indices:
        # A[0,0] -> U[0]
        # A[0,1] -> U[1]
        # A[1,0] -> U[2]
        # A[1,1] -> U[3]
        
        # Strassen intermediates for 2x2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Reconstructed C:
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # Define coefficients for U, V, W for each intermediate M
        # U: [A00, A01, A10, A11]
        # V: [B00, B01, B10, B11] mapped to specific indices
        # W: [C00, C01, C10, C11] mapped to specific indices
        
        # Note: V indices in the global V matrix:
        # B00 -> 2t
        # B01 -> 2t+1
        # B10 -> 16+2t
        # B11 -> 16+2t+1
        
        # W indices in the global W matrix:
        # C00 -> 4t
        # C01 -> 4t+2
        # C10 -> 4t+1
        # C11 -> 4t+3
        
        v_idx = {
            'B00': 2*t,
            'B01': 2*t+1,
            'B10': 16+2*t,
            'B11': 16+2*t+1
        }
        
        w_idx = {
            'C00': 4*t,
            'C01': 4*t+2,
            'C10': 4*t+1,
            'C11': 4*t+3
        }
        
        # U indices are fixed: A00=0, A01=1, A10=2, A11=3
        
        # M1
        U[0][col] += 1
        U[3][col] += 1
        V[v_idx['B00']][col] += 1
        V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1
        W[w_idx['C11']][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1
        U[3][col] += 1
        V[v_idx['B00']][col] += 1
        W[w_idx['C10']][col] += 1
        W[w_idx['C11']][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[v_idx['B01']][col] += 1
        V[v_idx['B11']][col] -= 1
        W[w_idx['C01']][col] += 1
        W[w_idx['C11']][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[v_idx['B10']][col] += 1
        V[v_idx['B00']][col] -= 1
        W[w_idx['C00']][col] += 1
        W[w_idx['C10']][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1
        U[1][col] += 1
        V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] -= 1
        W[w_idx['C01']][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1
        U[0][col] -= 1
        V[v_idx['B00']][col] += 1
        V[v_idx['B01']][col] += 1
        W[w_idx['C11']][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1
        U[3][col] -= 1
        V[v_idx['B10']][col] += 1
        V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1
        col += 1
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # The Strassen block decomposition is optimal for this structure.
    # Rank 56 is the theoretical minimum for 8 independent 2x2 multiplications.
    U, V, W = _strassen_block(n, m, p)
    
    # Verify to ensure correctness. If it fails (should not), fallback.
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
