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
    # We can vectorize this partially, but explicit loop is clear.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms: sum(U[a,r] * V[b,r] * W[c,r]) should be 0
    # Only check a subset to save time if rank is large, but here rank is small.
    # Full check is 4*32*32 = 4096 checks.
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.add((i * m + j, j * p + k, k * n + i))
                
    # To speed up, we can check only non-zero interactions?
    # But U, V, W are sparse.
    # For now, keep full check but optimized loop.
    for a in range(n * m):
        u_row = U[a]
        # If u_row is all zeros, skip
        if not np.any(u_row):
            continue
        for b in range(m * p):
            v_row = V[b]
            if not np.any(v_row):
                continue
            # Pre-compute u_row * v_row? No, W changes.
            for c in range(p * n):
                if (a, b, c) in valid_indices:
                    continue
                val = np.dot(u_row, v_row * W[c])
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
    This is the theoretical minimum for this tensor structure (direct sum of 8 M_222 tensors).
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
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
    
    # Rank 56 is the theoretical minimum for 8 independent 2x2 multiplications.
    U, V, W = _strassen_block(n, m, p)
    
    # Verify to ensure correctness.
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
