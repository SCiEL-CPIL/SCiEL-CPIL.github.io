"""Matrix multiplication 2,3,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_m = n * m  # 6
    m_p = m * p  # 18
    p_n = p * n  # 12

    # Build the target tensor T as a 3D array for verification
    # T[x, y, z] where x = i*m+j, y = j*p+k, z = k*n+i
    T = np.zeros((n_m, m_p, p_n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Verification function: check if U, V, W decompose T exactly
    def verify(U, V, W, tol=1e-9):
        try:
            U = np.asarray(U, dtype=float)
            V = np.asarray(V, dtype=float)
            W = np.asarray(W, dtype=float)
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            return np.max(np.abs(T_recon - T)) < tol
        except Exception:
            return False

    # Convert to list of lists with integer/half-integer values
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result

    # Schoolbook decomposition (rank 36) as fallback
    def schoolbook():
        sr = n * m * p
        Us = np.zeros((n_m, sr))
        Vs = np.zeros((m_p, sr))
        Ws = np.zeros((p_n, sr))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, col] = 1
                    Vs[j*p+k, col] = 1
                    Ws[k*n+i, col] = 1
                    col += 1
        return Us, Vs, Ws

    # Try to find a rank-18 decomposition via alternating least squares
    def try_rank18():
        rank = 18
        U = np.random.randn(n_m, rank)
        V = np.random.randn(m_p, rank)
        W = np.random.randn(p_n, rank)
        
        # Precompute for efficiency
        # For fixed U, V: solve for W
        # For fixed V, W: solve for U
        # For fixed U, W: solve for V
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update W: for each (x,y), the equation is sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
            # This is linear in W. We can set up a least squares problem.
            # For each z, we have equations: for all (x,y): sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
            # Let A[z] be the matrix where A[z][idx, r] = U[x,r]*V[y,r] for (x,y) = idx
            # Then A[z] @ W[z,:] = T[:,:,z].flatten()
            
            # Build the system for W
            # Number of (x,y) pairs: n_m * m_p = 6*18 = 108
            # For each z (0..11), we have 108 equations in 18 unknowns (W[z,0..17])
            
            for z in range(p_n):
                # Build coefficient matrix
                coeffs = np.zeros((n_m * m_p, rank))
                rhs = np.zeros(n_m * m_p)
                idx = 0
                for x in range(n_m):
                    for y in range(m_p):
                        coeffs[idx, :] = U[x, :] * V[y, :]
                        rhs[idx] = T[x, y, z]
                        idx += 1
                # Solve least squares
                W[z, :], residuals, rank_w, sv = np.linalg.lstsq(coeffs, rhs, rcond=None)
            
            # Update U: for each (y,z), the equation is sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
            # Linear in U. For each x, we have 18*12=216 equations in 18 unknowns.
            for x in range(n_m):
                coeffs = np.zeros((m_p * p_n, rank))
                rhs = np.zeros(m_p * p_n)
                idx = 0
                for y in range(m_p):
                    for z in range(p_n):
                        coeffs[idx, :] = V[y, :] * W[z, :]
                        rhs[idx] = T[x, y, z]
                        idx += 1
                U[x, :], residuals, rank_u, sv = np.linalg.lstsq(coeffs, rhs, rcond=None)
            
            # Update V: for each (x,z), the equation is sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
            # Linear in V. For each y, we have 6*12=72 equations in 18 unknowns.
            for y in range(m_p):
                coeffs = np.zeros((n_m * p_n, rank))
                rhs = np.zeros(n_m * p_n)
                idx = 0
                for x in range(n_m):
                    for z in range(p_n):
                        coeffs[idx, :] = U[x, :] * W[z, :]
                        rhs[idx] = T[x, y, z]
                        idx += 1
                V[y, :], residuals, rank_v, sv = np.linalg.lstsq(coeffs, rhs, rcond=None)
            
            # Check convergence
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-10:
                return U, V, W
        
        return None

    # Try rank-18 decomposition with random restarts
    best_result = None
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(attempt * 12345 + 67890)
        result = try_rank18()
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_result = (U, V, W)
                break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
# EVOLVE-BLOCK-END
