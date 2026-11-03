"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    deadline = time.time() + 140
    
    best_rank = n * m * p  # schoolbook rank 36
    best_result = None
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half_integers(M):
        """Snap matrix entries to nearest half-integer."""
        return np.round(M * 2.0) / 2.0
    
    # Try ranks from low to high
    for rank in range(15, n*m*p + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 7919 + rank * 104729)
            
            # Initialize
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            # Alternating least squares
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k in [0,p)
                        # Let M[k,c] = V[j*p+k,c] * W[k*n+i,c], shape (p, rank)
                        # Then U[i*m+j,:] @ M.T = ones(p)
                        M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])  # (p, rank)
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # Need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,n)
                        # Let N[i,c] = U[i*m+j,c] * W[k*n+i,c], shape (n, rank)
                        # Then V[j*p+k,:] @ N.T = ones(n)
                        N = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])  # (n, rank)
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(N.T, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # Need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j in [0,m)
                        # Let P[j,c] = U[i*m+j,c] * V[j*p+k,c], shape (m, rank)
                        # Then W[k*n+i,:] @ P.T = ones(m)
                        P = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])  # (m, rank)
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(P.T, np.ones(m), rcond=None)
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    # Try to snap
                    U_s = snap_to_half_integers(U)
                    V_s = snap_to_half_integers(V)
                    W_s = snap_to_half_integers(W)
                    
                    if verify(U_s, V_s, W_s, tol=1e-6):
                        best_rank = rank
                        best_result = {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }
                        found = True
                        break
                    elif verify(U, V, W, tol=1e-10):
                        # Use non-snapped if snapped doesn't work
                        best_rank = rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
