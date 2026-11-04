import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    
    # Build the tensor T[a,b,g] where a=i*m+j, b=j*p+k, g=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                g = k*n + i
                T[a, b, g] = 1.0
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Strategy: Try structured block decompositions first, then random search
    # 
    # Block decomposition idea:
    # Partition A (3x3) and B (3x4) into blocks and use Strassen for 2x2 parts
    #
    # Let's try a direct construction with rank 27:
    # Use Strassen (rank 7) for the top-left 2x2 x 2x2 block
    # Plus additional terms for cross-block interactions
    
    # First, try a known structured decomposition
    # Rank 27 construction:
    # 
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    #
    # C11 = A11*B11 + A12*B21  (2x2 result)
    # C12 = A11*B12 + A12*B22  (2x2 result)
    # C21 = A21*B11 + A22*B21  (1x2 result)
    # C22 = A21*B12 + A22*B22  (1x2 result)
    
    # Let me try to construct this explicitly
    # Strassen for 2x2x2:
    # M1 = (A21+A22)*(B11+B12)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*B22
    # M5 = (A11+A22)*(B11+B22)
    # M6 = (A11-A21)*(B12+B22)
    # M7 = (A21-A22)*(B11-B21)
    
    # For our 3x3 x 3x4 case, let me try a different approach
    # Use the fact that we can share some computations
    
    # Let me try rank 28 with explicit block structure
    # Then fall back to random search for lower ranks
    
    # Try random search for ranks 20-27
    for r in range(20, 28):
        if time.time() > deadline:
            break
        
        max_attempts = 5000 if r <= 22 else 2000 if r <= 24 else 1000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Use structured random: mix of {-1, 0, 1} with some bias
            # Try different sparsity patterns
            if attempt % 3 == 0:
                # Dense
                U = np.random.choice([-1, 0, 1], size=(n*m, r))
                V = np.random.choice([-1, 0, 1], size=(m*p, r))
            elif attempt % 3 == 1:
                # Sparse (mostly zeros)
                U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[0.3, 0.4, 0.3])
                V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[0.3, 0.4, 0.3])
            else:
                # Binary (no negatives)
                U = np.random.choice([0, 1], size=(n*m, r))
                V = np.random.choice([0, 1], size=(m*p, r))
            
            # Precompute M (108 x r)
            M = np.zeros((n*m * m*p, r), dtype=np.float64)
            for a in range(n*m):
                for b in range(m*p):
                    M[a*(m*p)+b, :] = U[a, :] * V[b, :]
            
            # Solve for W
            W = np.zeros((p*n, r), dtype=np.float64)
            valid = True
            
            for g in range(p*n):
                b_g = T[:, :, g].flatten()
                
                try:
                    w_g, residuals, rank_val, sv = np.linalg.lstsq(M, b_g, rcond=None)
                except:
                    valid = False
                    break
                    
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        valid = False
                        break
                else:
                    res = np.linalg.norm(M @ w_g - b_g)
                    if res > 1e-8:
                        valid = False
                        break
                        
                W[g, :] = w_g
                
            if not valid:
                continue
                
            # Check if W is close to integers or half-integers
            W_int = np.round(W)
            T_approx = np.einsum('ac,bc,gc->abg', U, V, W_int)
            if np.allclose(T_approx, T, atol=1e-7):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
            
            W_half = np.round(W * 2) / 2.0
            T_approx = np.einsum('ac,bc,gc->abg', U, V, W_half)
            if np.allclose(T_approx, T, atol=1e-7):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_half.tolist()}
            
            # Also try rounding U and V if they're close
            U_int = np.round(U)
            V_int = np.round(V)
            if np.allclose(U_int, U) and np.allclose(V_int, V):
                T_approx = np.einsum('ac,bc,gc->abg', U_int, V_int, W_int)
                if np.allclose(T_approx, T, atol=1e-7):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to schoolbook rank 36
    rank = 36
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
