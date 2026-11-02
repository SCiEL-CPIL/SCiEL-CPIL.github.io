"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    1. Build the rank-35 Strassen-based decomposition (5 blocks x 7 Strassen products).
    2. Use Alternating Least Squares (ALS) to attempt rank reduction from 34 down to a lower bound.
    3. Verify any found decomposition and return the best (lowest rank) valid one.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Build the rank-35 decomposition
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(35):
        u = np.array(U35)[:, r]
        v = np.array(V35)[:, r]
        w = np.array(W35)[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Try ranks from 34 down to 20 (lower bound)
    # We prioritize higher ranks first to find a solution quickly, then try to go lower.
    # However, if 34 fails, 33 might succeed.
    # Given time constraints, we try a few ranks.
    
    # Lower bound is max(n*m, m*p, p*n) = 20.
    # But tensor rank is usually higher.
    # Let's try 34, 33, 32, 31, 30.
    
    for target_rank in range(34, 19, -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a decomposition of target_rank
        # Use multiple random initializations
        found = False
        for trial in range(10):
            if time.time() > deadline - 5:
                break
                
            result = _try_rank_decomposition_als(T, n, m, p, target_rank, time_budget=10)
            if result is not None:
                U_new, V_new, W_new = result
                # Verify the decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = target_rank
                    found = True
                    break  # Found a valid lower rank, stop trying this rank
        
        if found:
            # If we found a lower rank, we could continue to try even lower, 
            # but given time, we might break early if we found a very low one.
            # For now, we break to return the improvement.
            break

    return best_result


def _build_strassen_decomposition(n, m, p):
    """Build the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # A factors (U)
    strassen_A = [
        [1, 0, 0, 1],  # P1: (a00+a11)
        [0, 0, 1, 1],  # P2: (a10+a11)
        [1, 0, 0, 0],  # P3: a00
        [0, 0, 0, 1],  # P4: a11
        [1, 1, 0, 0],  # P5: (a00+a10)
        [-1, 0, 1, 0], # P6: (a11-a00)
        [0, 1, 0, -1]  # P7: (a01-a11)
    ]
    
    # B factors (V)
    # Note: B is 2x2 in the block. Indices (j, k) where j is row (0,1), k is col (0,1)
    # V index = j*p + k. In block, p=2. So V indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    strassen_B = [
        [1, 1, 1, 1],  # P1: (b00+b11) -> b00, b11 are 1? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11*B11
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B01) -- Wait, B01?
        # Let's use the standard set:
        # P1: (A00+A11)(B00+B11)
        # P2: (A10+A11)B00
        # P3: A00(B10-B11)
        # P4: A11*B11
        # P5: (A00+A10)B11
        # P6: (A11-A00)(B00+B01) -> This is not standard.
        # Standard 7:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11*B11
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B01) -- No, usually (A00+A11) is used.
        # Let's stick to the coefficients that work.
        # Output:
        # C00 = M1 - M2 + M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M5
        # C11 = M1 - M4 + M7
        # This requires specific M definitions.
        
        # Let's use the coefficients from the previous program which worked (score 1.1429).
        # The previous program's strassen_B and w_contrib were verified to produce rank 35.
        # I will reuse them to ensure the base case is correct.
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    w_contrib[5][3] = 1
    
    w_contrib[6][0] = 1
    
    col = 0
    for b in range(5):
        for r in range(7):
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return U, V, W


def _try_rank_decomposition_als(T, n, m, p, target_rank, time_budget=10):
    """
    Try to find a rank-target_rank decomposition of tensor T using ALS.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Try multiple random initializations
    for trial in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # ALS Iterations
        for it in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Solve for U
            # T_ijk = sum_r U_ir W_kr V_jr
            # For fixed i: sum_r U_ir (W_kr V_jr) = T_ijk
            # Vectorize: for each i, we have a linear system for U_i (length R)
            # Matrix M_i of shape (dim_v * dim_w, R)
            # M_i[j, k, r] = W[k, r] * V[j, r]
            # t_i of shape (dim_v * dim_w)
            
            U_new = np.zeros_like(U)
            for i in range(dim_u):
                # Construct M_i
                # M_i is (dim_v * dim_w) x R
                # Entry (j, k, r) = W[k, r] * V[j, r]
                # We can form this as:
                # M_i = (V * W^T) reshaped?
                # V is (dim_v, R), W is (dim_w, R)
                # V[:, :, None] * W[None, :, :] -> (dim_v, dim_w, R)
                # Reshape to (dim_v * dim_w, R)
                M_i = (V[:, :, None] * W[None, :, :]).reshape(dim_v * dim_w, target_rank)
                t_i = T[i].reshape(-1)
                
                # Solve least squares
                try:
                    U_i, _, _, _ = np.linalg.lstsq(M_i, t_i, rcond=None)
                    U_new[i] = U_i
                except:
                    pass
            
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(dim_v):
                M_j = (U[:, :, None] * W[None, :, :]).reshape(dim_u * dim_w, target_rank)
                t_j = T[:, j, :].reshape(-1)
                
                try:
                    V_j, _, _, _ = np.linalg.lstsq(M_j, t_j, rcond=None)
                    V_new[j] = V_j
                except:
                    pass
            
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(dim_w):
                M_k = (U[:, :, None] * V[None, :, :]).reshape(dim_u * dim_v, target_rank)
                t_k = T[:, :, k].reshape(-1)
                
                try:
                    W_k, _, _, _ = np.linalg.lstsq(M_k, t_k, rcond=None)
                    W_new[k] = W_k
                except:
                    pass
            
            W = W_new
            
            # Check convergence
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            loss = np.sum((T - T_est)**2)
            if loss < 1e-10:
                break
        
        # Final check
        T_final = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T - T_final)) < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = len(U[0])
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute tensor from decomposition
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Compute expected tensor
    T_expected = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_expected[u, v, w] = 1
    
    return np.max(np.abs(T - T_expected)) < 1e-8
# EVOLVE-BLOCK-END
