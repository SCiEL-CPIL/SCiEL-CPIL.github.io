"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    Strategy: Use alternating least-squares for direct rank search,
    with structured decomposition as fallback.
    """
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    def schoolbook_full():
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
    
    def full_verify(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True
    
    def als_solve(rank, max_iter=5000, seed=42, tol=1e-7):
        """
        Alternating Least Squares for the full 3x4x6 problem.
        Returns (U, V, W) as numpy arrays or None.
        """
        if not check_time():
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Build constraint matrix for each variable
        # For U[a, r]: sum over (b,c) where (i,j) from a, (j,k) from b, (k,i) from c
        # T[a,b,c] = 1 if i==i2, j==j2, k==k2 else 0
        
        # Precompute valid (b,c) pairs for each a
        U_constraints = {}
        for i in range(n):
            for j in range(m):
                a = i * m + j
                pairs = []
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    pairs.append((b, c))
                U_constraints[a] = pairs
        
        V_constraints = {}
        for j in range(m):
            for k in range(p):
                b = j * p + k
                pairs = []
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    pairs.append((a, c))
                V_constraints[b] = pairs
        
        W_constraints = {}
        for k in range(p):
            for i in range(n):
                c = k * n + i
                pairs = []
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    pairs.append((a, b))
                W_constraints[c] = pairs
        
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            if not check_time():
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # Build system: for each k, sum_r U[a,r] * V[b_k,r] * W[c_k,r] = 1
                    # This is: V[a,:] @ diag(W) @ U[a,:] = 1 for each valid (b,c)
                    # Actually: for each k, sum_r U[a,r] * V[b_k,r] * W[c_k,r] = 1
                    # This is a linear system in U[a,:]
                    # Let M be matrix where M[k,r] = V[b_k,r] * W[c_k,r]
                    # Then M @ U[a,:] = ones(p)
                    M = np.zeros((p, rank))
                    for idx, k in enumerate(range(p)):
                        b = j * p + k
                        c = k * n + i
                        M[idx, :] = V[b, :] * W[c, :]
                    # Solve M @ x = ones(p)
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[a, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    M = np.zeros((n, rank))
                    for idx, i in enumerate(range(n)):
                        a = i * m + j
                        c = k * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[b, :] = x
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    M = np.zeros((m, rank))
                    for idx, j in enumerate(range(m)):
                        a = i * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[c, :] = x
                    except:
                        pass
            
            # Check convergence every 10 iterations
            if iteration % 10 == 0:
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < tol:
                    # Also check zero entries
                    zero_ok = True
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                i1 = a // m
                                j1 = a % m
                                j2 = b // p
                                k2 = b % p
                                k3 = c // n
                                i3 = c % n
                                if i1 != i3 or j1 != j2 or k2 != k3:
                                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                    if abs(val) > tol:
                                        zero_ok = False
                                        break
                            if not zero_ok:
                                break
                        if not zero_ok:
                            break
                    if zero_ok:
                        return U, V, W
                
                # Early stopping if loss not improving
                loss = max_err
                if loss > prev_loss * 1.01 and iteration > 100:
                    break
                prev_loss = loss
        
        return None
    
    def als_solve_3x3x3(rank, max_iter=5000, seed=42, tol=1e-7):
        """
        ALS for 3x3x3 matrix multiplication tensor.
        Returns (U, V, W) as numpy arrays of shape (9, rank) or None.
        """
        if not check_time():
            return None
        
        np.random.seed(seed)
        U = np.random.randn(9, rank) * 0.1
        V = np.random.randn(9, rank) * 0.1
        W = np.random.randn(9, rank) * 0.1
        
        # U index: i*3+j, V index: j*3+k, W index: k*3+i
        # For U[i*3+j]: sum over k of V[j*3+k] * W[k*3+i] = 1
        # For V[j*3+k]: sum over i of U[i*3+j] * W[k*3+i] = 1
        # For W[k*3+i]: sum over j of U[i*3+j] * V[j*3+k] = 1
        
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            if not check_time():
                return None
            
            # Update U
            for i in range(3):
                for j in range(3):
                    a = i * 3 + j
                    M = np.zeros((3, rank))
                    for k in range(3):
                        b = j * 3 + k
                        c = k * 3 + i
                        M[k, :] = V[b, :] * W[c, :]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(3), rcond=None)
                        U[a, :] = x
                    except:
                        pass
            
            # Update V
            for j in range(3):
                for k in range(3):
                    b = j * 3 + k
                    M = np.zeros((3, rank))
                    for i in range(3):
                        a = i * 3 + j
                        c = k * 3 + i
                        M[i, :] = U[a, :] * W[c, :]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(3), rcond=None)
                        V[b, :] = x
                    except:
                        pass
            
            # Update W
            for k in range(3):
                for i in range(3):
                    c = k * 3 + i
                    M = np.zeros((3, rank))
                    for j in range(3):
                        a = i * 3 + j
                        b = j * 3 + k
                        M[j, :] = U[a, :] * V[b, :]
                    try:
                        x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(3), rcond=None)
                        W[c, :] = x
                    except:
                        pass
            
            if iteration % 10 == 0:
                max_err = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            a = i * 3 + j
                            b = j * 3 + k
                            c = k * 3 + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < tol:
                    # Check zero entries
                    zero_ok = True
                    for a in range(9):
                        for b in range(9):
                            for c in range(9):
                                i1 = a // 3
                                j1 = a % 3
                                j2 = b // 3
                                k2 = b % 3
                                k3 = c // 3
                                i3 = c % 3
                                if i1 != i3 or j1 != j2 or k2 != k3:
                                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                    if abs(val) > tol:
                                        zero_ok = False
                                        break
                            if not zero_ok:
                                break
                        if not zero_ok:
                            break
                    if zero_ok:
                        return U, V, W
                
                loss = max_err
                if loss > prev_loss * 1.01 and iteration > 100:
                    break
                prev_loss = loss
        
        return None
    
    def build_from_3x3x3(U3, V3, W3, rank_3x3):
        """
        Build the full 3x4x6 decomposition from two 3x3x3 decompositions and two rank-1 terms.
        """
        total_rank = 2 * rank_3x3 + 2
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        
        # Block 1: Left half (k=0,1,2)
        for i in range(3):
            for j in range(3):
                U[i*4 + j, 0:rank_3x3] = U3[i*3 + j, :]
        for j in range(3):
            for k in range(3):
                V[j*6 + k, 0:rank_3x3] = V3[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, 0:rank_3x3] = W3[k*3 + i, :]
        
        # Term 2: A2 * B1_bottom (3x1x3) -> Rank 1
        r_offset = 2 * rank_3x3
        for i in range(3):
            U[i*4 + 3, r_offset] = 1
        for k in range(3):
            V[3*6 + k, r_offset] = 1
        for k in range(3):
            for i in range(3):
                W[k*3 + i, r_offset] = 1
                
        # Block 2: Right half (k=3,4,5)
        for i in range(3):
            for j in range(3):
                U[i*4 + j, rank_3x3:2*rank_3x3] = U3[i*3 + j, :]
        for j in range(3):
            for k in range(3):
                V[j*6 + 3 + k, rank_3x3:2*rank_3x3] = V3[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, rank_3x3:2*rank_3x3] = W3[k*3 + i, :]
                
        # Term 4: A2 * B2_bottom (3x1x3) -> Rank 1
        r_offset2 = 2 * rank_3x3 + 1
        for i in range(3):
            U[i*4 + 3, r_offset2] = 1
        for k in range(3):
            V[3*6 + 3 + k, r_offset2] = 1
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, r_offset2] = 1
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    best_result = schoolbook_full()
    if not full_verify(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p
    
    # Strategy 1: Try direct ALS on full problem with decreasing ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Target rank 36 (half of schoolbook) via structured search or direct ALS.
    for rank in [36, 34, 32, 30]:
        if not check_time():
            break
        for seed in range(10):
            if not check_time():
                break
            res = als_solve(rank, max_iter=5000, seed=seed, tol=1e-8)
            if res is not None:
                U, V, W = res
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if full_verify(U_list, V_list, W_list, tol=1e-6):
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        if best_rank < n * m * p:
            break
    
    # Strategy 2: Try 3x3x3 decomposition approach (Fallback if direct fails)
    if best_rank == n * m * p:
        for rank_3x3 in [19, 20, 21, 22, 23, 24]:
            if not check_time():
                break
            total_rank = 2 * rank_3x3 + 2
            if total_rank >= best_rank:
                continue
            for seed in range(5):
                if not check_time():
                    break
                res_3x3 = als_solve_3x3x3(rank_3x3, max_iter=5000, seed=seed, tol=1e-7)
                if res_3x3 is not None:
                    U3, V3, W3 = res_3x3
                    # Verify 3x3x3 solution
                    ok = True
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                a = i * 3 + j
                                b = j * 3 + k
                                c = k * 3 + i
                                val = np.sum(U3[a, :] * V3[b, :] * W3[c, :])
                                if abs(val - 1.0) > 1e-6:
                                    ok = False
                                    break
                            if not ok: break
                        if not ok: break
                    if ok:
                        full_res = build_from_3x3x3(U3, V3, W3, rank_3x3)
                        if full_verify(full_res["U"], full_res["V"], full_res["W"], tol=1e-6):
                            best_rank = total_rank
                            best_result = full_res
                            break
            if best_rank < n * m * p:
                break
    
    return best_result
# EVOLVE-BLOCK-END
