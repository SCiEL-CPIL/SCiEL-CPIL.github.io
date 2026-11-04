"""Matrix multiplication 2x3 by 3x5 with rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    where a = i*m+j, b = j*p+k, c = k*n+i
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals T[a,b,c]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[2][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00-A10)*B11
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1
    
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A01+A11)*(B00+B01)
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00-A10)*(B10+B11)
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A01-A11)*(B00-B10)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = -1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _rank28_decomposition(n, m, p):
    """
    Rank-28 decomposition using two Strassen 2x2x2 blocks:
    - Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (7 terms)
    - Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5]  (7 terms)
    - Schoolbook: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]  (4 terms)
    - Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]  (10 terms)
    Total: 7 + 7 + 4 + 10 = 28
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (6, 28)
    V = [[0] * rank for _ in range(m * p)]  # (15, 28)
    W = [[0] * rank for _ in range(p * n)]  # (10, 28)
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    a_map1 = [0, 1, 3, 4]
    b_map1 = [0, 1, 5, 6]
    c_map1 = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map1[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map1[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map1[s]][col] += W_s[s][r]
        col += 1
    
    # Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5]
    a_map2 = [0, 1, 3, 4]
    b_map2 = [3, 4, 8, 9]
    c_map2 = [6, 7, 8, 9]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map2[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map2[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map2[s]][col] += W_s[s][r]
        col += 1
    
    # Schoolbook: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    for i in range(2):
        for j in range(2):
            k = 2
            U[i * 3 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _try_rank_search(n, m, p, T, target_rank, max_iters=200, n_restarts=5, deadline=None):
    """
    Attempt to find an exact rank-r decomposition via iterative refinement.
    Uses alternating minimization on the tensor factorization:
      T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    For each factor, we solve a linear system. We use a real-valued search
    and then check if the solution is (approximately) rational with small denominators.
    """
    nA, nB, nC = n*m, m*p, p*n
    rng = np.random.RandomState(42)
    
    best_result = None
    best_err = np.inf
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize with random factors
        U = rng.randn(nA, target_rank)
        V = rng.randn(nB, target_rank)
        W = rng.randn(nC, target_rank)
        
        err = np.inf
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U: for each r, U[:,r] should satisfy sum_b,c V[b,r]*W[c,r]*U[a,r] = T[a,b,c]
            # Fix V,W; solve for U. For each r, the contribution to T from rank-r is
            # outer(U[:,r], V[:,r], W[:,r]). We want to minimize ||T - sum_r outer(U_r, V_r, W_r)||^2.
            # Fix V and W, the problem for U is linear:
            # For each a: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
            # Reshape: T is (nA, nB, nC). For fixed r, define M_r[b,c] = V[b,r]*W[c,r]
            # Then T[a,:,:] = sum_r U[a,r] * M_r
            # This is a linear system: for each a, T[a] (vectorized) = sum_r U[a,r] * vec(M_r)
            
            # Build the system: T_flat[a, (b,c)] = sum_r U[a,r] * M_flat[r, (b,c)]
            # So T_flat = U @ M_flat^T  where T_flat is (nA, nB*nC), M_flat is (target_rank, nB*nC)
            M_flat = np.zeros((target_rank, nB * nC))
            for r in range(target_rank):
                M_flat[r, :] = (V[:, r] * W[:, r].reshape(1, -1)).ravel()
            
            T_flat = T.reshape(nA, -1)
            # Solve: T_flat ≈ U @ M_flat.T  =>  U ≈ T_flat @ pinv(M_flat.T)
            # Or solve least squares: U = argmin ||T_flat - U @ M_flat.T||_F
            # This is: for each a, T_flat[a] = U[a] @ M_flat.T
            # U[a] = T_flat[a] @ pinv(M_flat.T)
            # Equivalent: U = T_flat @ pinv(M_flat.T)
            
            try:
                M_T = M_flat.T  # (nB*nC, target_rank)
                # U (nA, target_rank) = T_flat (nA, nB*nC) @ pinv(M_T)
                U_new = T_flat @ np.linalg.pinv(M_T)
            except np.linalg.LinAlgError:
                break
            
            # Update V: fix U, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each b: T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
            # N_flat[r, (a,c)] = U[a,r]*W[c,r]
            N_flat = np.zeros((target_rank, nA * nC))
            for r in range(target_rank):
                N_flat[r, :] = (U[:, r] * W[:, r].reshape(1, -1)).ravel()
            T_flat2 = T.reshape(-1, nB, nC).transpose(1, 0, 2).reshape(nB, -1)
            # T_flat2[b, (a,c)] = sum_r V[b,r] * N_flat[r, (a,c)]
            # V = T_flat2 @ pinv(N_flat.T)
            try:
                N_T = N_flat.T
                V_new = T_flat2 @ np.linalg.pinv(N_T)
            except np.linalg.LinAlgError:
                break
            
            # Update W: fix U, V
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For each c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
            P_flat = np.zeros((target_rank, nA * nB))
            for r in range(target_rank):
                P_flat[r, :] = (U[:, r] * V[:, r]).ravel()
            T_flat3 = T.reshape(nA, nB, nC).transpose(2, 0, 1).reshape(nC, -1)
            # T_flat3[c, (a,b)] = sum_r W[c,r] * P_flat[r, (a,b)]
            # W = T_flat3 @ pinv(P_flat.T)
            try:
                P_T = P_flat.T
                W_new = T_flat3 @ np.linalg.pinv(P_T)
            except np.linalg.LinAlgError:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.max(np.abs(T_recon - T))
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_result is not None and best_err < 1e-9:
        U_b, V_b, W_b = best_result
        # Convert to list of lists with rounded values
        U_out = [[round(U_b[i, r], 6) for r in range(target_rank)] for i in range(nA)]
        V_out = [[round(V_b[i, r], 6) for r in range(target_rank)] for i in range(nB)]
        W_out = [[round(W_b[i, r], 6) for r in range(target_rank)] for i in range(nC)]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    T = _build_target_tensor(n, m, p)
    
    # Try rank-28 decomposition first (known valid)
    result28 = _rank28_decomposition(n, m, p)
    valid28, diff28 = _verify_decomposition(result28['U'], result28['V'], result28['W'], T)
    
    if not valid28:
        return _schoolbook_decomposition(n, m, p)
    
    # Try lower ranks using iterative search
    for target_rank in [25, 26, 27]:
        if time.time() > deadline - 30:  # need 30s buffer
            break
        result = _try_rank_search(n, m, p, T, target_rank, 
                                   max_iters=150, n_restarts=3, deadline=deadline - 30)
        if result is not None:
            valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)
            if valid:
                return result
    
    # Fallback to rank-28
    return result28

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
