import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the tensor T[i, j, k] = 1 if it corresponds to a multiplication term
    # T is of shape (n, m, p)
    # Actually, the tensor T_{ijk} for matrix multiplication C = A B is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Wait, the standard tensor is T_{i, j, k} where i is row of A, j is col of A/row of B, k is col of B.
    # The multiplication tensor T has T[i, j, k] = 1 for all i,j,k? No.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k such that it represents the bilinear map.
    # For matrix multiplication, the tensor is T_{i, j, k} = 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
    # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The rank of this tensor is the rank of the matrix multiplication.
    
    # Let's create the tensor T of shape (n, m, p)
    T = np.ones((n, m, p))
    
    # We want to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Note the index mapping:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # This is a 3-way tensor decomposition problem.
    # We will use Alternating Least Squares (ALS).
    
    def run_als(rank, max_iters=500, tol=1e-9, seed=42):
        np.random.seed(seed)
        # Initialize U, V, W
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            # Update U
            # For each row i*m+j of U, we need to solve for U[i*m+j, :]
            # The equation is: T[i, j, :] = U[i*m+j, :] @ (V[j, :] * W[:, i])^T ? No.
            # Let's derive the linear system for U.
            # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Let u = U[i*m+j, :].
            # We want to minimize sum_k (T[i, j, k] - sum_r u_r * V[j*p+k, r] * W[k*n+i, r])^2
            # This is a linear least squares problem for u.
            # Construct matrix M_u of shape (p, rank)
            # M_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then M_u @ u = T[i, j, :]
            
            for i in range(n):
                for j in range(m):
                    # Build M_u
                    M_u = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            M_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve M_u @ u = T[i, j, :]
                    # T[i, j, :] is a vector of length p
                    target = T[i, j, :]
                    # Use np.linalg.lstsq
                    u_new, residuals, rank_u, s = np.linalg.lstsq(M_u, target, rcond=None)
                    U[i*m+j, :] = u_new
                    
            # Update V
            # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For fixed j, k, we update V[j*p+k, :]
            # Let v = V[j*p+k, :].
            # Minimize sum_i (T[i, j, k] - sum_r U[i*m+j, r] * v_r * W[k*n+i, r])^2
            # Matrix M_v of shape (n, rank)
            # M_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # M_v @ v = T[:, j, k]
            
            for j in range(m):
                for k in range(p):
                    M_v = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            M_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    target = T[:, j, k]
                    v_new, residuals, rank_v, s = np.linalg.lstsq(M_v, target, rcond=None)
                    V[j*p+k, :] = v_new
                    
            # Update W
            # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For fixed k, i, we update W[k*n+i, :]
            # Let w = W[k*n+i, :].
            # Minimize sum_j (T[i, j, k] - sum_r U[i*m+j, r] * V[j*p+k, r] * w_r)^2
            # Matrix M_w of shape (m, rank)
            # M_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # M_w @ w = T[i, :, k]
            
            for k in range(p):
                for i in range(n):
                    M_w = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            M_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    target = T[i, :, k]
                    w_new, residuals, rank_w, s = np.linalg.lstsq(M_w, target, rcond=None)
                    W[k*n+i, :] = w_new
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (T[i, j, k] - val) ** 2
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        # Final error check
        final_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    final_err += (T[i, j, k] - val) ** 2
        
        return U, V, W, final_err

    # Try to find a low rank decomposition
    # Start with a rank slightly lower than schoolbook (72)
    # If we find one, great. If not, return schoolbook.
    
    # We will try ranks from 71 down to 60.
    # To save time, we will limit iterations.
    
    start_time = time.time()
    time_limit = 110 # Leave 10s for overhead
    
    for rank in range(71, 59, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random seeds for each rank
        for seed in range(5):
            if time.time() - start_time > time_limit:
                break
            U, V, W, err = run_als(rank, max_iters=100, tol=1e-8, seed=seed)
            
            if err < 1e-6:
                # We have a solution. Now check if it's rational/integer.
                # The directive mentions "Smith Normal Form" or "rationalization".
                # For now, check if values are close to integers.
                # If they are, round them and verify.
                
                # Check if entries are close to integers
                U_int = np.rint(U)
                V_int = np.rint(V)
                W_int = np.rint(W)
                
                # Check error with integers
                int_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                            int_err += (T[i, j, k] - val) ** 2
                
                if int_err < 1e-6:
                    # Valid integer decomposition
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                
                # Check if entries are close to halves (x.5)
                U_half = np.rint(U * 2) / 2
                V_half = np.rint(V * 2) / 2
                W_half = np.rint(W * 2) / 2
                
                half_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_half[i*m+j, r] * V_half[j*p+k, r] * W_half[k*n+i, r]
                            half_err += (T[i, j, k] - val) ** 2
                
                if half_err < 1e-6:
                    # Convert to integers by multiplying by 2? No, the output must be the factors.
                    # The task allows "integer/half-integer entries".
                    # So we can return the half-integers directly?
                    # "integer/half-integer entries" means values like 0.5, 1.0, 1.5.
                    # We can return them as floats.
                    return {
                        "U": U_half.tolist(),
                        "V": V_half.tolist(),
                        "W": W_half.tolist()
                    }

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
# EVOLVE-BLOCK-END
