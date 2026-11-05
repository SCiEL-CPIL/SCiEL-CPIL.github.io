import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # Convention: T2[i,j,k] = sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] = 1 for all i,j,k
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # (i=0,j=0)
        [0, 0, 0, 0, 1, 0, 1],    # (i=0,j=1)
        [0, 1, 0, 0, 0, 1, 0],    # (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1]    # (i=1,j=1)
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # (j=0,k=0)
        [0, 0, 1, 0, 0, 1, 0],    # (j=0,k=1)
        [0, 0, 0, 1, 0, 0, 1],    # (j=1,k=0)
        [1, 0, -1, 0, 1, 0, 1]    # (j=1,k=1)
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],    # (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0]    # (k=1,i=1)
    ], dtype=int)
    
    # Verify Strassen decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.dot(U2[i*2+j, :] * V2[j*2+k, :], W2[k*2+i, :]))
                if s != 1:
                    # Fallback to schoolbook if Strassen is broken
                    return _schoolbook_4x4()
    
    n = m = p = 4
    rank = 49
    
    U4 = np.zeros((n*m, rank), dtype=int)
    V4 = np.zeros((m*p, rank), dtype=int)
    W4 = np.zeros((p*n, rank), dtype=int)
    
    # Build 4x4 factors using Kronecker product structure
    # Index mapping: i = 2*i_hi + i_lo, j = 2*j_hi + j_lo, k = 2*k_hi + k_lo
    # Rank index: r = 7*r1 + r2
    
    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            u_idx_hi = i_hi * 2 + j_hi
            u_idx_lo = i_lo * 2 + j_lo
            for r1 in range(7):
                a = U2[u_idx_hi, r1]
                if a == 0:
                    continue
                for r2 in range(7):
                    b = U2[u_idx_lo, r2]
                    if b == 0:
                        continue
                    U4[i * 4 + j, r1 * 7 + r2] = a * b
    
    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            v_idx_hi = j_hi * 2 + k_hi
            v_idx_lo = j_lo * 2 + k_lo
            for r1 in range(7):
                a = V2[v_idx_hi, r1]
                if a == 0:
                    continue
                for r2 in range(7):
                    b = V2[v_idx_lo, r2]
                    if b == 0:
                        continue
                    V4[j * 4 + k, r1 * 7 + r2] = a * b
    
    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            w_idx_hi = k_hi * 2 + i_hi
            w_idx_lo = k_lo * 2 + i_lo
            for r1 in range(7):
                a = W2[w_idx_hi, r1]
                if a == 0:
                    continue
                for r2 in range(7):
                    b = W2[w_idx_lo, r2]
                    if b == 0:
                        continue
                    W4[k * 4 + i, r1 * 7 + r2] = a * b
    
    # Verify correctness
    ok = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.dot(U4[i*4+j, :] * V4[j*4+k, :], W4[k*4+i, :]))
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        return _schoolbook_4x4()
    
    # Attempt rank reduction via Gaussian elimination
    # Check if any rank-1 term (column) can be removed
    # We look for a column r such that the contribution of that term is linearly
    # dependent on the others.
    
    # For each column r, check if removing it still gives a valid decomposition
    # This is expensive, so we only try if we have time
    
    if time.time() < deadline - 30:
        # Try to find a removable column
        # The idea: if there exists a column r such that for all (i,j,k):
        # U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] = 0 or can be compensated
        # Actually, we need the sum over all r to equal 1, so we can't just remove.
        # Instead, we check if any column is entirely zero (which it isn't in Kronecker)
        # Or if we can find a linear dependency.
        
        # A simpler approach: check if the Kronecker product has any redundant structure.
        # For 4x4 matrix multiplication, the known rank is 49 (Strassen's square).
        # It is not known if a lower rank exists.
        # So we just return the rank-49 decomposition.
        
        pass
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

def _schoolbook_4x4():
    n = m = p = 4
    rank = n * m * p  # 64
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
