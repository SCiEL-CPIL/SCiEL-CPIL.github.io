import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    # Note: The problem statement says "exactly decompose the matrix multiplication tensor".
    # The tensor has 1s at specific indices and 0s elsewhere.
    
    num_a = n * m
    num_b = m * p
    num_c = p * n
    
    T = np.zeros((num_a, num_b, num_c), dtype=float)
    
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                triples.append((a, b, c))
                
    # We also need to enforce that all other entries are 0.
    # The ALS approach minimizes the squared error over the whole tensor.
    # This is a large tensor: 9 * 15 * 15 = 2025 elements. Manageable.
    
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((num_a, rank))
        V = np.zeros((num_b, rank))
        W = np.zeros((num_c, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W):
        # U, V, W are lists of lists
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check all 1s
        for (a, b, c) in triples:
            val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c]) # This is wrong, it's element-wise product sum
            # Correct: sum_r U[a,r] * V[b,r] * W[c,r]
            val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
            if not np.isclose(val, 1.0, atol=1e-9):
                return False
        
        # Check all 0s
        # Iterate over all indices, skip the 1s
        # To be efficient, we can reconstruct the tensor and check
        # But for small dims, explicit check is fine
        # Actually, we can just check a subset or the whole tensor if fast enough
        # 9*15*15 = 2025 checks. Each check is a dot product of length R.
        # If R=30, 2025*30 = 60k ops. Very fast.
        
        for a in range(num_a):
            for b in range(num_b):
                for c in range(num_c):
                    if (a, b, c) in set(triples):
                        continue
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if not np.isclose(val, 0.0, atol=1e-9):
                        return False
        return True
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from a lower bound to 44
    # Lower bound: Strassen-like? For 3x3x3 it's 23. For 3x3x5, maybe 25-30?
    # Let's try 20 to 44.
    for R in range(20, 45):
        if time.time() > deadline:
            break
            
        # Multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use small integers to help with snapping
            U = np.random.randint(-1, 2, (num_a, R)).astype(float)
            V = np.random.randint(-1, 2, (num_b, R)).astype(float)
            W = np.random.randint(-1, 2, (num_c, R)).astype(float)
            
            # ALS Optimization
            # Minimize || T - sum_r U_r V_r W_r ||_F^2
            # We can reshape T into a matrix for each factor.
            # For U: Fix V, W.
            # T_reshaped (num_a, num_b * num_c)
            # M (num_b * num_c, R) where M[ (b*num_c+c), r ] = V[b, r] * W[c, r]
            # Then T_reshaped = U @ M^T
            # Solve U = T_reshaped @ M @ (M^T M)^-1
            
            # Precompute V*W matrix for each R? No, V and W change.
            
            for _ in range(20): # ALS iterations
                # Solve for U
                # M_VW shape (num_b * num_c, R)
                M_VW = np.zeros((num_b * num_c, R))
                for r in range(R):
                    M_VW[:, r] = (V[:, r] * W[:, r]).flatten()
                
                T_mat = T.reshape(num_a, num_b * num_c)
                # U = T_mat @ M_VW @ pinv(M_VW.T @ M_VW)
                # Use lstsq for stability
                try:
                    U, _, _, _ = np.linalg.lstsq(M_VW.T, T_mat.T, rcond=None)
                    U = U.T
                except:
                    pass
                
                # Solve for V
                # M_UW shape (num_a * num_c, R)
                M_UW = np.zeros((num_a * num_c, R))
                for r in range(R):
                    M_UW[:, r] = (U[:, r] * W[:, r]).flatten()
                
                # T reshaped to (num_b, num_a * num_c)
                # T[b, a, c] -> T_mat_V[b, a*num_c+c]
                T_mat_V = np.zeros((num_b, num_a * num_c))
                for b in range(num_b):
                    for a in range(num_a):
                        for c in range(num_c):
                            T_mat_V[b, a * num_c + c] = T[a, b, c]
                
                try:
                    V, _, _, _ = np.linalg.lstsq(M_UW.T, T_mat_V.T, rcond=None)
                    V = V.T
                except:
                    pass
                
                # Solve for W
                # M_UV shape (num_a * num_b, R)
                M_UV = np.zeros((num_a * num_b, R))
                for r in range(R):
                    M_UV[:, r] = (U[:, r] * V[:, r]).flatten()
                
                # T reshaped to (num_c, num_a * num_b)
                T_mat_W = np.zeros((num_c, num_a * num_b))
                for c in range(num_c):
                    for a in range(num_a):
                        for b in range(num_b):
                            T_mat_W[c, a * num_b + b] = T[a, b, c]
                
                try:
                    W, _, _, _ = np.linalg.lstsq(M_UV.T, T_mat_W.T, rcond=None)
                    W = W.T
                except:
                    pass
                
            # After ALS, check if close to half-integers
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness
            if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                best_rank = R
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
