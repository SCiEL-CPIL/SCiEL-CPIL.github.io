import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for a in range(dim_u):
        i_a = a // m
        j_a = a % m
        for b in range(dim_v):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(dim_w):
                k_c = c // n
                i_c = c % n
                if i_a == i_c and k_b == k_c:
                    T[a, b, c] = 1.0

    # Precompute the linear system structure for each c
    # M_c is (dim_u * dim_v) x R, where M_c[a*dim_v+b, r] = U[a, r] * V[b, r]
    # We will iterate over R from 27 to 30 (trying to beat schoolbook 30)
    
    # Strategy: 
    # 1. For each R, generate random sparse integer U, V
    # 2. Solve for W using least squares
    # 3. Check if W is close to integer/half-integer
    # 4. If not, try "hard thresholding" / refinement
    
    # To improve success rate, we use a structured initialization based on
    # the idea that rank-28 Strassen-like constructions exist for 2x3x5?
    # Actually, the lower bound is 26 (from literature).
    # Let's try R = 27, 28, 29, 30.
    
    # We will use random restarts with different sparsity levels.
    
    for R in range(27, 31):
        for trial in range(50):
            # Random initialization with sparsity
            # Each entry is 0 with prob 0.5, else +/-1
            U = np.zeros((dim_u, R))
            V = np.zeros((dim_v, R))
            
            # Ensure each column has at least a few non-zeros
            for r in range(R):
                # Pick random non-zero positions
                nz_u = np.random.choice(dim_u, size=max(1, dim_u // 3), replace=False)
                nz_v = np.random.choice(dim_v, size=max(1, dim_v // 3), replace=False)
                
                U[nz_u, r] = np.random.choice([-1, 1], size=len(nz_u))
                V[nz_v, r] = np.random.choice([-1, 1], size=len(nz_v))
            
            # Solve for W
            valid = True
            W = np.zeros((dim_w, R))
            
            for c in range(dim_w):
                # Build M_c
                # M_c has shape (dim_u * dim_v, R)
                # M_c[a * dim_v + b, r] = U[a, r] * V[b, r]
                
                # Vectorized construction
                # U is (dim_u, R), V is (dim_v, R)
                # We need outer product per column r
                # M_c[:, r] = (U[:, r] * V[:, r]).flatten() ? No.
                # M_c[a * dim_v + b, r] = U[a, r] * V[b, r]
                # This is the Kronecker product of U[:, r] and V[:, r] flattened?
                # No, it's just element-wise product broadcast.
                
                # M_c[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                M_c = np.zeros((dim_u * dim_v, R))
                for r in range(R):
                    M_c[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                T_c_flat = T[:, :, c].flatten()
                
                try:
                    W_c, residuals, rank, s = np.linalg.lstsq(M_c, T_c_flat, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if W_c is close to integer or half-integer
                # The task allows integer/half-integer entries.
                # Let's check if 2*W_c is close to integer.
                W_c_2 = 2 * W_c
                W_c_2_int = np.round(W_c_2)
                
                if not np.allclose(W_c_2, W_c_2_int, atol=1e-7):
                    valid = False
                    break
                
                W[c, :] = W_c_2_int / 2.0
            
            if valid:
                # Verify the solution
                # T_prime[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                T_prime = np.zeros((dim_u, dim_v, dim_w))
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            T_prime[a, b, c] = np.dot(U[a, :], V[b, :] * W[c, :])
                
                if np.allclose(T_prime, T, atol=1e-7):
                    # Convert to lists
                    # Need to ensure they are integers or half-integers
                    # The return type expects integer/half-integer entries.
                    # If we use half-integers, we should represent them as floats?
                    # The prompt says "integer/half-integer entries".
                    # The current seed returns integers.
                    # If W has half-integers, we can return them as floats.
                    # But the seed returns lists of ints.
                    # Let's check if all are integers first.
                    if np.allclose(W, np.round(W), atol=1e-9):
                        W_out = W.astype(int).tolist()
                    else:
                        W_out = W.tolist()
                    
                    U_out = U.astype(int).tolist()
                    V_out = V.astype(int).tolist()
                    
                    return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
