import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)
    # c = k*n + i (k in 0..p-1, i in 0..n-1)
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor()
    
    # Generate schoolbook solution as fallback
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decomposition
    # Strategy: Target rank 18 first (as per directive), then others
    # We use a simple coordinate descent / ALS approach with random restarts
    
    np.random.seed(42)
    
    # Define ranks to try: 18 is the specific target, then a few others
    ranks_to_try = [18, 20, 22, 25, 30, 40, 50]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
            
        found = False
        # Try a few restarts
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values to avoid overflow
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                
                try:
                    # Update U
                    # T_flat[a, b*c] = sum_r U[a,r] V[r,b] W[r,c]
                    # For fixed V, W:
                    # P[r, b*(p*n) + c] = V[r,b] * W[r,c]
                    # T_flat[a, :] = U[a, :] @ P
                    # U[a, :] = T_flat[a, :] @ P^+
                    
                    # Construct P efficiently
                    # V is (m*p, R), W is (p*n, R)
                    # We need P of shape (R, m*p * p*n)
                    # P[r, b*(p*n) + c] = V[r, b] * W[r, c]
                    
                    # This is an outer product for each r
                    # We can do this with broadcasting
                    # V[:, :, None] * W[:, None, :] -> (R, m*p, p*n)
                    # Reshape to (R, m*p * p*n)
                    P = (V[:, :, None] * W[:, None, :]).reshape(target_rank, -1)
                    
                    T_flat = T.reshape(n*m, -1)
                    
                    # Solve U = T_flat @ P.T (P P.T)^-1
                    # Use lstsq for stability
                    U_new, _, _, _ = np.linalg.lstsq(P.T, T_flat.T, rcond=None)
                    U = U_new.T
                    
                    # Update V
                    # For fixed U, W:
                    # T[a, b, c] = sum_r U[a,r] V[r,b] W[r,c]
                    # For fixed b, T[:, b, :] (n*m, p*n) = sum_r (U[:, r] * W[r, :]) * V[r, b]
                    # Let's reshape T to (n*m, m*p, p*n)
                    # For each b, we have a matrix S_b (n*m, p*n)
                    # S_b = V.T[b, :] @ (U * W) ... no.
                    # S_b[a, c] = sum_r U[a,r] V[r,b] W[r,c]
                    # Let M_r[a, c] = U[a,r] * W[r,c]
                    # S_b = sum_r V[r,b] M_r
                    # This is a linear combination of M_r with coefficients V[:, b]
                    # We want to find V[:, b] such that S_b is approximated.
                    # This is equivalent to:
                    # For each (a, c), T[a, :, c] (vector of length m*p) = V.T @ (U[:, :] * W[:, c])
                    # Let Y_c = U * W[:, c] (element-wise, shape n*m, R)
                    # Then T[:, :, c] = V.T @ Y_c
                    # So V.T = T_all @ Y_all^+
                    # We can stack all c's.
                    # Y_all shape: (p*n, n*m, R) ? No.
                    # For each c, Y_c is (n*m, R).
                    # T_c is (n*m, m*p).
                    # T_c = V.T @ Y_c
                    # So [T_0, T_1, ..., T_{p*n-1}] = V.T @ [Y_0, Y_1, ..., Y_{p*n-1}]
                    # Left side shape: (n*m, m*p * p*n)
                    # Right side: (m*p, R) @ (R, p*n * n*m) -> No, Y_c is (n*m, R).
                    # We need to form a matrix Y_stack of shape (R, n*m * p*n)
                    # Y_stack[r, a*(p*n) + c] = U[a,r] * W[r,c]
                    
                    Y_stack = (U[:, :, None] * W[:, None, :]).T.reshape(target_rank, -1)
                    # Wait, U is (n*m, R), W is (p*n, R).
                    # U[:, :, None] is (n*m, R, 1).
                    # W[:, None, :] is (p*n, 1, R).
                    # Broadcasting fails directly.
                    # We need U[a,r] * W[c,r].
                    # Shape (n*m, p*n, R).
                    # U_exp = U[:, None, :] # (n*m, 1, R)
                    # W_exp = W[None, :, :] # (1, p*n, R)
                    # Prod = U_exp * W_exp # (n*m, p*n, R)
                    # Reshape to (R, n*m * p*n)
                    Y_stack = (U[:, None, :] * W[None, :, :]).transpose(2, 0, 1).reshape(target_rank, -1)
                    
                    T_flat_for_V = T.reshape(n*m, m*p * p*n)
                    # T_flat_for_V[a, b*(p*n) + c] = T[a,b,c]
                    # We want V.T (m*p, R) such that V.T @ Y_stack.T = T_flat_for_V
                    # V.T = T_flat_for_V @ Y_stack^+
                    V_T_new, _, _, _ = np.linalg.lstsq(Y_stack.T, T_flat_for_V.T, rcond=None)
                    V = V_T_new.T
                    
                    # Update W
                    # Similar to V.
                    # T[a, b, c] = sum_r U[a,r] V[r,b] W[r,c]
                    # For fixed c, T[:, :, c] (n*m, m*p) = W.T[c, :] @ (U * V[:, c])?
                    # Let's derive:
                    # T[a, b, c] = sum_r U[a,r] V[r,b] W[r,c]
                    # For fixed c, let Z_c = U * V[:, c] (element-wise, shape n*m, R)
                    # Then T[:, :, c] = W.T[c, :] @ Z_c
                    # Stack all c.
                    # Z_stack shape: (R, n*m * m*p)
                    # Z_stack[r, a*(m*p) + b] = U[a,r] * V[r,b]
                    
                    Z_stack = (U[:, :, None] * V[:, None, :]).transpose(2, 0, 1).reshape(target_rank, -1)
                    # U (n*m, R), V (m*p, R)
                    # U[:, :, None] (n*m, R, 1)
                    # V[:, None, :] (m*p, 1, R)
                    # Prod (n*m, m*p, R)
                    # Transpose to (R, n*m, m*p)
                    # Reshape to (R, n*m * m*p)
                    
                    T_flat_for_W = T.reshape(n*m, m*p * p*n)
                    # T_flat_for_W[a, b*(p*n) + c]
                    # We need to reshape T to group by c.
                    # T shape (n*m, m*p, p*n).
                    # We want T_c shape (n*m, m*p) for each c.
                    # T_reshaped = T.transpose(0, 2, 1) -> (n*m, p*n, m*p)
                    # Flatten last two dims: (n*m, p*n * m*p)
                    # This matches T_flat_for_W if we reorder indices?
                    # Original T_flat_for_W index: a, b*(p*n) + c.
                    # New index: a, c*(m*p) + b.
                    # So we need to permute T_flat_for_W.
                    # T_perm = T.reshape(n*m, m*p, p*n).transpose(0, 2, 1).reshape(n*m, -1)
                    
                    T_perm = T.reshape(n*m, m*p, p*n).transpose(0, 2, 1).reshape(n*m, -1)
                    
                    W_T_new, _, _, _ = np.linalg.lstsq(Z_stack.T, T_perm.T, rcond=None)
                    W = W_T_new.T
                    
                except Exception:
                    break
            
            # Check if current U,V,W gives exact decomposition
            M = np.einsum('ar,rb,rc->abc', U, V, W)
            if np.allclose(M, T, atol=1e-5):
                # Check if we can round to integers/half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                M_round = np.einsum('ar,rb,rc->abc', U_round, V_round, W_round)
                if np.allclose(M_round, T, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    found = True
                    break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
