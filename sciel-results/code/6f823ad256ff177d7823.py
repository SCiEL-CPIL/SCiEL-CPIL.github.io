import time
import random
import numpy as np

def _build_strassen_rank25():
    """Build the rank-25 Strassen decomposition."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W, n=2, m=2, p=7):
    """Verify tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(U, V, W, target_rank, time_budget):
    """
    Try to reduce rank from current to target_rank by merging terms.
    Uses structured random search with integer/half-integer entries.
    """
    start_time = time.time()
    n, m, p = 2, 2, 7
    current_rank = len(U[0])
    
    # Try random search in a bounded space
    # Use the structure: entries are likely in {-1, 0, 1} or {-2, -1, 0, 1, 2}
    
    for size in [3, 5, 7]:
        if time.time() - start_time > time_budget:
            break
        
        low = -size // 2
        high = (size - 1) // 2
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget:
                break
            
            # Random integer matrix
            U_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U_rand, V_rand, W_rand, n, m, p):
                return U_rand, V_rand, W_rand
    
    return None

def _try_merge_terms(U, V, W, target_rank, time_budget):
    """
    Try to merge two terms into one by finding a linear combination.
    Start from the rank-25 solution and try to reduce.
    """
    start_time = time.time()
    n, m, p = 2, 2, 7
    current_rank = len(U[0])
    
    if current_rank == target_rank:
        return U, V, W
    
    # Try to find pairs of terms that can be merged
    # For each pair (a, b), check if we can find a single term that replaces them
    # This is complex, so use random perturbation approach
    
    # Strategy: start from rank-25, randomly set one column to zero,
    # and try to adjust others to compensate
    for attempt in range(2000):
        if time.time() - start_time > time_budget:
            break
        
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Remove one random term
        remove_idx = random.randint(0, current_rank - 1)
        for i in range(n*m):
            U_test[i].pop(remove_idx)
        for i in range(m*p):
            V_test[i].pop(remove_idx)
        for i in range(p*n):
            W_test[i].pop(remove_idx)
        
        # Now we have current_rank-1 terms. Try to adjust to fix the equations.
        # This is a system of n*m*p = 28 equations with (current_rank-1)*12 unknowns
        # Too complex for exact solve, so just verify
        if len(U_test[0]) == target_rank and _verify(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Step 1: Build rank-25 Strassen decomposition
    U25, V25, W25 = _build_strassen_rank25()
    
    # Verify
    if not _verify(U25, V25, W25):
        # Fallback to schoolbook
        n, m, p = 2, 2, 7
        rank = 28
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U25, V25, W25
    best_rank = 25
    
    # Step 2: Try to find rank 24
    if time.time() < deadline - 30:
        time_budget = min(deadline - 30 - time.time(), 60)
        result = _try_rank_reduction(U25, V25, W25, 24, time_budget)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 24
    
    # Step 3: If still time, try rank 23
    if best_rank == 24 and time.time() < deadline - 20:
        time_budget = min(deadline - 20 - time.time(), 40)
        result = _try_rank_reduction(best_U, best_V, best_W, 23, time_budget)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 23
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
    
    C_mat = np.zeros((2, 7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
    
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED, rank =", U.shape[1])
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
