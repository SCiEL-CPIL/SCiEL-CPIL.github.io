"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]=1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, j, k] = s
    return np.allclose(result, 1.0, atol=1e-9)

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-6

def _try_round(U, V, W):
    """Try to round entries to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    n, m, p = DIMENSIONS
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-8):
    """Alternating least squares for tensor decomposition."""
    rng = np.random.default_rng(42)
    # Initialize
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    T = _build_tensor(n, m, p)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed (i,j), this is a linear system in U[i*m+j,:]
                # Build matrix A[c,k] = V[j*p+k,c] * W[k*n+i,c]
                # Then T[i,j,:] = U[i*m+j,:] @ A.T  (approximately)
                A = np.zeros((rank, p))
                for k in range(p):
                    for c in range(rank):
                        A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                # Solve U[i*m+j,:] @ A.T = T[i,j,:]
                # i.e., A @ U[i*m+j,:].T = T[i,j,:].T
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((rank, n))
                for i in range(n):
                    for c in range(rank):
                        A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, T[:,j,k].T, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((rank, m))
                for j in range(m):
                    for c in range(rank):
                        A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                except:
                    pass
        
        # Compute residual
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i,j,k] = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
        res = np.linalg.norm(recon - T)
        if res < tol:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try lower ranks
    # Known: for [2,4,7], try ranks from some lower bound up
    # The tensor rank is at least max(n*m, m*p, p*n) = max(8,28,14) = 28
    # Also at least n*m*p / min(n,m,p) ... not a tight bound
    # Let's try ranks from 28 to 55
    
    for rank in range(28, rank_school):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            rng = np.random.default_rng(rank * 100 + trial)
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
            
            T = _build_tensor(n, m, p)
            
            # ALS iterations
            for it in range(100):
                if time.time() - _START_TIME > _TIME_LIMIT - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((rank, p))
                        for k in range(p):
                            for c in range(rank):
                                A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            for c in range(rank):
                                A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, T[:,j,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((rank, m))
                        for j in range(m):
                            for c in range(rank):
                                A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i,j,k] = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                res = np.linalg.norm(recon - T)
                if res < 1e-7:
                    break
            
            # Try to round
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            rounded = _try_round(U_list, V_list, W_list)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                # Found a valid decomposition at this rank, can stop
                return best_result
            
            # Also check if already close to half-integers
            if _is_half_integer(np.max(np.abs(U))) and _is_half_integer(np.max(np.abs(V))) and _is_half_integer(np.max(np.abs(W))):
                pass  # unlikely
    
    return best_result
# EVOLVE-BLOCK-END
