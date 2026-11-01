import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill (evaluator) or 150s (task)

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    def verify(U, V, W, num_tests=3):
        # Efficient verification using tensor contraction
        # We need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
        # This is equivalent to checking if the tensor T_{i,j,k} defined by the decomposition matches the identity tensor.
        
        # Let's construct the tensor T from U, V, W
        # T is shape (n, m, p)
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # We can compute this efficiently using einsum or loops.
        # Given small dimensions (2,4,7), even loops are fast.
        # But let's use vectorized ops if possible.
        
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # This is a bit tricky for einsum directly because of the index matching (i,j,k vs j,k,i).
        # Let's just do the loop over i, j, k and check against 1.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U_r[i, j, :], V_r[j, k, :]), W_r[k, i, :])
                    if not np.isclose(val, 1.0, atol=1e-6):
                        return False
        return True

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Known lower bounds and known ranks for small matrices
    # For 2x4 * 4x7, the schoolbook is 56.
    # Strassen-like reductions might exist.
    # Let's try to find a decomposition with rank < 56.
    
    np.random.seed(42)
    
    # Try ranks from 55 down to 40
    # We will use an iterative refinement approach:
    # 1. Random initialize U, V, W
    # 2. Fix one matrix, solve for the others via least squares?
    #    The equation is trilinear.
    #    Fix W. Then for each (i,j,k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    #    This is still bilinear in U and V.
    #    
    # Alternative: Use the "tensor train" or "CP decomposition" optimization.
    # Minimize || T(U,V,W) - T_target ||_F^2
    # where T_target[i,j,k] = 1.
    
    def optimize(U, V, W, iterations=50):
        # Alternate minimization
        # T_target is all ones.
        
        # Precompute W_r as (p, n, R)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        for it in range(iterations):
            if time.time() > deadline - 10:
                break
            
            # Update U
            # For fixed V, W, we want to minimize sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
            # This is linear in U.
            # For a specific U[i*m+j, :], let u be the vector.
            # Term is (sum_r u_r * (V[j*p+k,r] W[k*n+i,r]) - 1)^2
            # Let A[i,j,k] = V[j*p+k,r] W[k*n+i,r] ... wait, A depends on r.
            # Let's define for each (i,j), the vector b_{i,j} of length R.
            # No, let's look at the system for U.
            # We have n*m*R unknowns.
            # The loss is sum_{i,j,k} ( dot(U[i*m+j,:], C_{i,j,k}) - 1 )^2
            # where C_{i,j,k}[r] = V[j*p+k,r] * W[k*n+i,r]
            
            # We can solve for each U[i*m+j, :] independently?
            # No, the terms are coupled by the sum over k.
            # Loss = sum_k ( dot(u, c_k) - 1 )^2 for each (i,j).
            # This is a standard least squares problem for each (i,j).
            # Let M_{i,j} be a matrix of shape (p, R) where M_{i,j}[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # We want to find u such that M_{i,j} u approximates ones(p).
            
            for i in range(n):
                for j in range(m):
                    M_ij = np.zeros((p, R))
                    for k in range(p):
                        M_ij[k, :] = V_r[j, k, :] * W_r[k, i, :]
                    # Solve M_ij u = ones
                    u, residuals, rank, sv = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                    U[i * m + j, :] = u
            
            # Update V
            # Similarly for V[j*p+k, :]
            # Loss = sum_i ( dot(v, d_i) - 1 )^2 for each (j,k).
            # d_i[r] = U[i*m+j, r] * W[k*n+i, r]
            # Matrix N_{j,k} of shape (n, R) where N_{j,k}[i, r] = U_r[i, j, r] * W_r[k, i, r]
            # Solve N_{j,k} v = ones(n)
            
            for j in range(m):
                for k in range(p):
                    N_jk = np.zeros((n, R))
                    for i in range(n):
                        N_jk[i, :] = U_r[i, j, :] * W_r[k, i, :]
                    v, residuals, rank, sv = np.linalg.lstsq(N_jk, np.ones(n), rcond=None)
                    V[j * p + k, :] = v
            
            # Update W
            # Similarly for W[k*n+i, :]
            # Loss = sum_j ( dot(w, e_j) - 1 )^2 for each (k,i).
            # e_j[r] = U[i*m+j, r] * V[j*p+k, r]
            # Matrix O_{k,i} of shape (m, R) where O_{k,i}[j, r] = U_r[i, j, r] * V_r[j, k, r]
            # Solve O_{k,i} w = ones(m)
            
            for k in range(p):
                for i in range(n):
                    O_ki = np.zeros((m, R))
                    for j in range(m):
                        O_ki[j, :] = U_r[i, j, :] * V_r[j, k, :]
                    w, residuals, rank, sv = np.linalg.lstsq(O_ki, np.ones(m), rcond=None)
                    W[k * n + i, :] = w
            
            # Update views
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            # Check convergence / validity
            # We can check the max error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U_r[i, j, :], V_r[j, k, :]), W_r[k, i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-5:
                return True, max_err
        return False, max_err

    # Try different ranks
    # Start from 55 down to 30 (optimistic)
    # Use multiple random restarts for lower ranks
    
    for R in range(rank_sb - 1, 29, -1):
        if time.time() > deadline - 10:
            break
        
        attempts = 5 if R > 40 else 20
        if R < 45:
            attempts = 50
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            success, err = optimize(U, V, W, iterations=100)
            
            if success:
                # Verify with the strict function
                if verify(U, V, W):
                    best_rank = R
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    # Try to go lower immediately? The loop handles that.
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
