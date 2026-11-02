import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Time limit guard
    start_time = time.time()
    time_limit = 100  # 100 seconds safety margin
    
    # Fallback: Schoolbook
    def schoolbook():
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
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """
        Verifies if the given decomposition exactly performs matrix multiplication.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = U.shape[1]
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Reshape U, V, W to (n, m, rank), (m, p, rank), (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # For each output entry (i, k), the sum over r of W[k, i, r] * (sum_j U[i, j, r] * A[i, j]) * (sum_j V[j, k, r] * B[j, k])
        # Wait, the standard tensor definition is:
        # C_ik = sum_r W_ki_r * M_r
        # M_r = sum_j U_ij_r A_ij * sum_j V_jk_r B_jk ? No.
        
        # Correct tensor equation:
        # T_{(i,j), (j,k), (i,k)} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1
        # All other T entries are 0.
        
        # We check if sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} == 1 for all i, j, k.
        
        # Efficient check:
        # Construct matrices A, B as basis vectors?
        # Or just iterate.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        
        return True

    # Strategy:
    # 1. Try to find a low-rank decomposition for 3x3x3 core.
    # 2. If found, extend to 3x3x6 using the direct sum structure (or block diagonal if possible, but here A is shared).
    #    For 3x3x6, B is 3x6. We can view this as two 3x3 multiplications A*B1 and A*B2.
    #    If we have a rank-R decomposition for 3x3x3, can we use it for 3x3x6?
    #    The tensor for 3x3x6 is T_6.
    #    T_6[(i,j), (j,k), (i,k)] = 1 for k in 0..5.
    #    If we use the same U for both halves?
    #    Let U be (9, R).
    #    Let V1 be (9, R) for k=0..2, V2 be (9, R) for k=3..5.
    #    Let W1 be (9, R) for k=0..2, W2 be (9, R) for k=3..5.
    #    Then we need:
    #    sum_r U_ij_r V1_jk_r W1_ki_r = 1 for k=0..2
    #    sum_r U_ij_r V2_j(k-3)_r W2_(k-3)i_r = 1 for k=3..5
    #    This implies that the decomposition for the first half (U, V1, W1) and second half (U, V2, W2) must both be valid 3x3x3 decompositions sharing the same U.
    #    If we find a 3x3x3 decomposition (U, V, W), we can just set V1=V, W1=W, V2=V, W2=W.
    #    Then the total rank is R (since U is shared, we don't double the rank? No, the rank is the number of terms r.
    #    If we use the same r for both, it's just one set of terms.
    #    Wait, the tensor is a sum of rank-1 terms.
    #    T_6 = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    #    For this to work, v_r must handle both k=0..2 and k=3..5.
    #    If we set v_r to be [V_core_r; V_core_r], and w_r to be [W_core_r; W_core_r], and u_r = U_core_r,
    #    Then for k=0..2, the term is U_core_ij_r * V_core_jk_r * W_core_ki_r = 1.
    #    For k=3..5, let k' = k-3. The term is U_core_ij_r * V_core_jk'_r * W_core_k'i_r = 1.
    #    So yes, if we have a rank-R decomposition for 3x3x3, we can construct a rank-R decomposition for 3x3x6 by repeating the V and W blocks.
    #    Wait, V is (m*p, R) = (18, R).
    #    V[j, k, r] for k=0..2 is V_core[j, k, r].
    #    V[j, k, r] for k=3..5 is V_core[j, k-3, r].
    #    W is (p*n, R) = (18, R).
    #    W[k, i, r] for k=0..2 is W_core[k, i, r].
    #    W[k, i, r] for k=3..5 is W_core[k-3, i, r].
    #    U is (n*m, R) = (9, R).
    #    U[i, j, r] is U_core[i, j, r].
    #    
    #    So the rank of 3x3x6 is at most the rank of 3x3x3.
    #    Known rank of 3x3x3 is 23 (Bini et al.) or 24? 23 is the current best known upper bound for 3x3x3 over reals?
    #    Actually, 23 is the best known for 3x3x3.
    #    So we should search for a rank-23 (or lower, though 19 is lower bound) decomposition for 3x3x3.
    #    If we find one, we can return it for 3x3x6.
    #    If not, we return schoolbook (54).
    #    Or we can try to find a direct 3x3x6 decomposition, but that's harder.
    #    The direct sum argument shows rank(3x3x6) <= rank(3x3x3).
    #    Is it possible that rank(3x3x6) < rank(3x3x3)?
    #    Unlikely, as 3x3x3 is a sub-tensor (by setting half of B to 0).
    #    So the minimal rank for 3x3x6 is likely 23.
    
    # Let's search for a 3x3x3 decomposition.
    # We will use a random search + least squares approach similar to inspiration 2.
    
    n_core = 3
    m_core = 3
    p_core = 3
    
    best_core_rank = 27
    found_core = False
    core_U = core_V = core_W = None
    
    # Search ranks from 23 upwards
    for R in range(23, 28):
        if time.time() - start_time > time_limit:
            break
            
        # Try random searches for this rank R
        attempts = 5000
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random U, V with entries -1, 0, 1
            U_rand = np.random.randint(-1, 2, size=(9, R))
            V_rand = np.random.randint(-1, 2, size=(9, R))
            
            W_core = np.zeros((9, R))
            valid = True
            
            # Solve for each output (i, k)
            for i in range(3):
                for k in range(3):
                    # Build 3 x R matrix M where M[j, r] = U[i,j,r] * V[j,k,r]
                    M = np.zeros((3, R))
                    for j in range(3):
                        for r in range(R):
                            M[j, r] = U_rand[i * 3 + j, r] * V_rand[j * 3 + k, r]
                    
                    # Solve M @ w_vec = ones(3)
                    # w_vec is W[i*3+k, :]
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(3), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is close to integers or half-integers
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))
                    
                    if error < 1e-9:
                        W_core[i * 3 + k, :] = w_vec
                    else:
                        valid = False
                        break
            
            if valid:
                found_core = True
                core_U = U_rand.tolist()
                core_V = V_rand.tolist()
                core_W = W_core.tolist()
                best_core_rank = R
                break
        
        if found_core:
            break
            
    if found_core:
        # Construct 3x3x6 solution from core
        R_total = best_core_rank
        U_total = [[0] * R_total for _ in range(n * m)]
        V_total = [[0] * R_total for _ in range(m * p)]
        W_total = [[0] * R_total for _ in range(p * n)]
        
        # U is shared
        for i in range(3):
            for j in range(3):
                for r in range(R_total):
                    U_total[i * 3 + j][r] = core_U[i * 3 + j][r]
        
        # V and W are repeated
        for j in range(3):
            for k in range(3):
                for r in range(R_total):
                    V_total[j * 6 + k][r] = core_V[j * 3 + k][r]
                    V_total[j * 6 + (k + 3)][r] = core_V[j * 3 + k][r]
                    
        for k in range(3):
            for i in range(3):
                for r in range(R_total):
                    W_total[k * 3 + i][r] = core_W[k * 3 + i][r]
                    W_total[(k + 3) * 3 + i][r] = core_W[k * 3 + i][r]
                    
        return {"U": U_total, "V": V_total, "W": W_total}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
