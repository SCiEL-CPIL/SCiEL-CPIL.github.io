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
    
    # Check spurious terms
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
        
        # M1: (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1
        W[w_idx['C10']][col] += 1; W[w_idx['C11']][col] -= 1
        col += 1
        
        # M3: A00(B01-B11)
        U[0][col] += 1
        V[v_idx['B01']][col] += 1; V[v_idx['B11']][col] -= 1
        W[w_idx['C01']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[3][col] += 1
        V[v_idx['B10']][col] += 1; V[v_idx['B00']][col] -= 1
        W[w_idx['C00']][col] += 1; W[w_idx['C10']][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[0][col] += 1; U[1][col] += 1
        V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] -= 1; W[w_idx['C01']][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[2][col] += 1; U[0][col] -= 1
        V[v_idx['B00']][col] += 1; V[v_idx['B01']][col] += 1
        W[w_idx['C11']][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[v_idx['B10']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1
        col += 1
        
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_attempts=50):
    """Attempt to reduce rank by removing zero or redundant columns."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    rank = U.shape[1]
    
    # Try removing columns that are linearly dependent
    for attempt in range(max_attempts):
        # Check if any column is all zeros
        zero_cols = []
        for r in range(rank):
            if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
                zero_cols.append(r)
        
        if zero_cols:
            for r in sorted(zero_cols, reverse=True):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
            rank = U.shape[1]
            continue
        
        # Check if any two columns can be merged (linear dependence)
        merged = False
        for r1 in range(rank):
            for r2 in range(r1 + 1, rank):
                # Check if column r2 is a scalar multiple of column r1
                if np.allclose(U[:, r2], 0) and np.allclose(V[:, r2], 0) and np.allclose(W[:, r2], 0):
                    continue
                # Try to find scalar alpha such that col_r2 = alpha * col_r1
                # This is complex; skip for now
                pass
        
        if not merged:
            break
    
    U_list = [[U[i, r] for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[V[i, r] for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[W[i, r] for r in range(W.shape[1])] for i in range(W.shape[0])]
    
    # Round to nearest integer/half-integer
    for lst in [U_list, V_list, W_list]:
        for row in lst:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Generate the Strassen block decomposition (rank 56)
    U, V, W = _strassen_block(n, m, p)
    
    # Verify correctness
    if _verify(U, V, W, n, m, p):
        # Try to reduce rank
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p)
        
        # Verify reduced version
        if _verify(U_red, V_red, W_red, n, m, p):
            return {"U": U_red, "V": V_red, "W": W_red}
        
        # Return original if reduction failed
        return {"U": U, "V": V, "W": W}
    
    # Fallback: should not happen
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
