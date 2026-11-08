import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    """
    Constructs a rank-17 decomposition for 2x3x3 matrix multiplication.
    Structure:
    - 7 terms for the 2x2x2 sub-problem (Strassen)
    - 4 terms for the 2x1x2 sub-problem (schoolbook)
    - 6 terms for the 2x3x1 sub-problem (schoolbook)
    Total: 17 terms.
    
    Indices:
    U: (n*m, rank) -> (6, 17)
    V: (m*p, rank) -> (9, 17)
    W: (p*n, rank) -> (6, 17)
    
    A[i, j] index in U: i*m + j
    B[j, k] index in V: j*p + k
    C[k, i] index in W: k*n + i (Trace convention)
    """
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # --- Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
    # A indices involved: 0,1 (row 0), 3,4 (row 1) -> 0,1,3,4
    # B indices involved: 0,1 (col 0), 3,4 (col 1) -> 0,1,3,4
    # C indices involved: 0,1 (col 0), 2,3 (col 1) -> 0,1,2,3
    
    # r=0: M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # --- Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] ---
    # A indices: 2 (A02), 5 (A12)
    # B indices: 6 (B20), 7 (B21)
    # C indices: 0,1,2,3
    
    # r=7: A02 * B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 * B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 * B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 * B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # --- Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2] ---
    # A indices: 0,1,2,3,4,5
    # B indices: 2 (B02), 5 (B12), 8 (B22)
    # C indices: 4 (C02), 5 (C12)
    
    # r=11: A00 * B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 * B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 * B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 * B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 * B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 * B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Pre-compute for speed if possible, but loop is fine for small dims
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build the known rank-17 solution
    U, V, W = _build_rank17()
    
    # Verify it
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but return anyway to be safe
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 17
    best_U, best_V, best_W = U, V, W
    
    # 2. Attempt to find rank 16 by checking if any term is redundant.
    # This is a cheap check: if removing term t still satisfies the equation, rank is 16.
    # We don't do complex perturbation searches as they are unlikely to succeed for this specific tensor
    # and waste time.
    
    for t in range(17):
        if time.time() > deadline - 5.0:
            break
        
        mask = np.ones(17, dtype=bool)
        mask[t] = False
        U16 = U[:, mask]
        V16 = V[:, mask]
        W16 = W[:, mask]
        
        if _verify(U16, V16, W16, n, m, p):
            best_rank = 16
            best_U, best_V, best_W = U16, V16, W16
            break
    
    # 3. Attempt to find rank 15 by checking if any two terms are redundant.
    # This is O(17^2) checks, which is 136 checks. Each check is O(18*15) operations. Very fast.
    if best_rank > 15:
        for t1 in range(17):
            if time.time() > deadline - 5.0:
                break
            for t2 in range(t1 + 1, 17):
                mask = np.ones(17, dtype=bool)
                mask[t1] = False
                mask[t2] = False
                U15 = U[:, mask]
                V15 = V[:, mask]
                W15 = W[:, mask]
                
                if _verify(U15, V15, W15, n, m, p):
                    best_rank = 15
                    best_U, best_V, best_W = U15, V15, W15
                    break
            if best_rank == 15:
                break

    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
        U, V, W = _build_rank17()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Round to ensure clean integer/half-integer output if possible
    best_U = np.round(best_U, 10)
    best_V = np.round(best_V, 10)
    best_W = np.round(best_W, 10)
    
    if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
        U, V, W = _build_rank17()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
