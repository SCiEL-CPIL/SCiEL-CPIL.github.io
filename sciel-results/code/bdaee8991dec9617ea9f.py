"""Matrix multiplication 4,5,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1 for all i,j,k
    """
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _round_to_halfint(mat):
    """Round a matrix to nearest half-integer values."""
    return np.round(mat * 2.0) / 2.0

def _try_rank(rank, n, m, p, max_iters=500, n_restarts=20, deadline=None):
    """Try to find an exact decomposition of given rank."""
    best = None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to avoid divergence
        for c in range(rank):
            norm = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-10)
            if norm > 0:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Alternating least squares
            # Fix V and W, solve for U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # U[u_idx, :] should satisfy: sum_k W[k*n+i, :] * V[j*p+k, :] * U[u_idx, :] = 1
                    # This is: U[u_idx, :] dot (sum_k W[k*n+i, :] * V[j*p+k, :]) = 1
                    # But we want to minimize ||U[u_idx, :] dot M - 1||^2 where M = sum_k ...
                    # Actually, for each (i,j), the constraint is:
                    # sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] * W[k*n+i, c]) = 1
                    # So let M_u[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                    # Then U[u_idx, :] dot M_u = 1
                    # To minimize ||U[u_idx, :]*M_u - 1||^2, the solution is U[u_idx, :] = M_u / (M_u dot M_u)
                    M_u = np.zeros(rank)
                    for k in range(p):
                        M_u += V[j * p + k, :] * W[k * n + i, :]
                    denom = np.dot(M_u, M_u)
                    if denom > 1e-12:
                        U[u_idx, :] = M_u / denom
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    M_v = np.zeros(rank)
                    for i in range(n):
                        M_v += U[i * m + j, :] * W[k * n + i, :]
                    denom = np.dot(M_v, M_v)
                    if denom > 1e-12:
                        V[v_idx, :] = M_v / denom
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    M_w = np.zeros(rank)
                    for j in range(m):
                        M_w += U[i * m + j, :] * V[j * p + k, :]
                    denom = np.dot(M_w, M_w)
                    if denom > 1e-12:
                        W[w_idx, :] = M_w / denom
            
            # Check convergence and exactness
            if it % 10 == 0:
                # Round to half-integers and verify
                U_r = _round_to_halfint(U)
                V_r = _round_to_halfint(V)
                W_r = _round_to_halfint(W)
                if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                    return (U_r, V_r, W_r)
                
                # Also try without rounding (check if already close to integer/half-integer)
                if _verify_decomposition(U, V, W, n, m, p):
                    return (U, V, W)
        
        # Final attempt with rounding
        U_r = _round_to_halfint(U)
        V_r = _round_to_halfint(V)
        W_r = _round_to_halfint(W)
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return (U_r, V_r, W_r)
    
    return None

def construct():
    """Constructs a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Try to find lower rank decompositions
    # Based on known results, for 4x4x4, rank can be as low as 20-25
    # For 4x5x6, let's try ranks from low to high
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try various ranks in increasing order
    # Start from a reasonable lower bound
    for rank in range(30, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, max_iters=200, n_restarts=10, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify once more
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
