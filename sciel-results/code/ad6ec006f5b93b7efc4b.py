import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek U (n*m, R), V (m*p, R), W (p*n, R) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-8):
        """Check if sum_r U[:,r] outer V[:,r] outer W[:,r] == T (all ones)"""
        R = U.shape[1]
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this efficiently using matrix multiplications
        # For each r, the contribution is u_r (n*m) outer v_r (m*p) outer w_r (p*n)
        # Reshape and sum
        
        # More efficient: for fixed i,k, T[i,j,k] for all j should be 1
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's compute the full tensor and check
        # Tensor shape: (n, m, p)
        T_comp = np.zeros((n, m, p))
        for r in range(R):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # For each i,j,k: u_r[i,j] * v_r[j,k] * w_r[k,i]
            # This is: for each i, (u_r[i,:] * w_r[:,i]) @ v_r ... no
            # T[i,j,k] = sum_j' ... no, it's just u_r[i,j]*v_r[j,k]*w_r[k,i]
            # For fixed i,k: sum over j? No, T[i,j,k] is a single value for each (i,j,k)
            # T[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # We can compute: for each i, define a_r[i] = w_r[:, i] (length p)
            # Then T[i,j,k] = u_r[i,j] * v_r[j,k] * a_r[i][k]
            # For fixed i: T[i,:,:] = u_r[i,:] (m,) outer v_r (m,p) * a_r[i] (p,) broadcast
            # Actually: T[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # = (u_r[i,:] * 1) * v_r * (1 * w_r[:,i])
            # For fixed i: T[i,:,:] = diag(u_r[i,:]) @ v_r @ diag(w_r[:,i]) ... no
            # T[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # Let's just do it with einsum or loops since dimensions are small
            for i in range(n):
                for k in range(p):
                    wk_i = w_r[k, i]
                    if wk_i == 0:
                        continue
                    # T[i,j,k] += u_r[i,j] * v_r[j,k] * wk_i
                    T_comp[i, :, k] += u_r[i, :] * v_r[:, k] * wk_i
        
        # Check if T_comp is all ones
        err = np.max(np.abs(T_comp - 1.0))
        return err < tol, err
    
    # ALS for CP decomposition of the all-ones tensor
    def als_decompose(R, max_iters=200, tol=1e-10, seed=42):
        """
        Find U (n*m, R), V (m*p, R), W (p*n, R) such that
        sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, R) * 0.5
        V = rng.randn(m * p, R) * 0.5
        W = rng.randn(p * n, R) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each i*m+j, solve least squares
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # This is: U[i*m+j, :] @ M_{ij} = 1_vector (length p)
            # where M_{ij}[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    # Build M_{ij}: (p, R)
                    # M_{ij}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M_ij = np.zeros((p, R))
                    for k in range(p):
                        M_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j, :] @ M_ij = ones(p)
                    # i.e., M_ij.T @ U[i*m+j, :].T = ones(p)
                    # U[i*m+j, :] = pinv(M_ij) @ ones(p)
                    # Use least squares
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each j*p+k, solve least squares
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed (j,k): sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
            # V[j*p+k, :] @ N_{jk} = ones(n)
            # where N_{jk}[i, r] = U[i*m+j, r] * W[k*n+i, r]
            
            for j in range(m):
                for k in range(p):
                    N_jk = np.zeros((n, R))
                    for i in range(n):
                        N_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(N_jk, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each k*n+i, solve least squares
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed (k,i): sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
            # W[k*n+i, :] @ P_{ki} = ones(m)
            # where P_{ki}[j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            for k in range(p):
                for i in range(n):
                    P_ki = np.zeros((m, R))
                    for j in range(m):
                        P_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(P_ki, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            ok, err = verify(U, V, W)
            if ok:
                return U, V, W, True
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        ok, err = verify(U, V, W)
        return U, V, W, ok
    
    # Try various ranks from low to high
    # Known: schoolbook is 56. Literature suggests rank ~26-30 for 2x4x7.
    # Let's try ranks from 24 to 40
    
    best_result = None
    best_rank = n * m * p
    
    for R in range(24, 41):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            U, V, W, ok = als_decompose(R, max_iters=100, seed=seed + R * 100)
            if ok:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = R
                found = True
                break
        if found:
            break
    
    if best_result is not None:
        # Convert to lists
        return {
            "U": best_result["U"].tolist(),
            "V": best_result["V"].tolist(),
            "W": best_result["W"].tolist()
        }
    
    # Fallback to schoolbook
    return schoolbook()
