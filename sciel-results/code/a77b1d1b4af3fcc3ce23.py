"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    # A is n x m (2x2), B is m x p (2x7), C is n x p (2x7)
    # U: (n*m) x rank = 4 x r
    # V: (m*p) x rank = 14 x r  
    # W: (p*n) x rank = 14 x r
    # C_flat[k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... 
    # Actually the tensor: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # C[k*n + i] = sum_j A[i*m+j] * B[j*p+k]
    
    deadline = time.time() + 140  # 10s safety margin
    
    # Try ranks from low to high
    for r in range(14, 29):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p)
        if result is not None:
            return result
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(r, n, m, p):
    """Try to find a rank-r decomposition using optimization."""
    np.random.seed(42)
    
    # Build the target tensor T[i, j, k] = 1 for all i,j,k
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] = 1 for all valid i, j, k
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This is a 3-tensor T of shape (n*m, m*p, p*n) but with specific structure.
    # Actually T is sparse: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0.
    
    # Let's use a different approach: parametrize and use least squares
    # For fixed U and V, W can be solved linearly.
    
    # Strategy: alternate optimization
    # Fix U, V -> solve for W (linear)
    # Fix V, W -> solve for U (linear)  
    # Fix U, W -> solve for V (linear)
    
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    # Build the linear system structure
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For fixed U,V: this is linear in W.
    # Group by (i,j,k) -> W index is k*n+i
    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    
    # Let's build the matrix for W: for each output index w_idx = k*n+i,
    # the equation is: sum_c M[w_idx, c] * W[w_idx, c] ... no wait.
    
    # Actually, for each triple (i,j,k), we have one equation:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # The unknowns are W[k*n+i,c] for all c.
    # But W[k*n+i,c] appears in multiple equations (for different j).
    
    # Let me restructure. For a fixed w_idx = k*n+i, the equations involving W[w_idx, :] are:
    # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[w_idx,c] = 1
    # This gives m equations for the r unknowns W[w_idx, 0..r-1]
    
    # So we can set up a least squares problem for W given U,V.
    
    def solve_W(U, V):
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                # For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[w_idx, c] = 1
                # Build m x r matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # Then A @ W[w_idx, :] = ones(m)
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve A_mat @ W[w_idx,:] = 1 (least squares)
                if r >= m:
                    W[w_idx, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                else:
                    W[w_idx, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
        return W
    
    def solve_U(U, V, W):
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                # For each k: sum_c U[u_idx, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Build p x r matrix B where B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then B @ U[u_idx, :] = ones(p)
                B_mat = np.zeros((p, r))
                for k in range(p):
                    B_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                U_new[u_idx, :] = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)[0]
        return U_new
    
    def solve_V(U, V, W):
        V_new = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                # For each i: sum_c U[i*m+j, c] * V[v_idx, c] * W[k*n+i, c] = 1
                # Build n x r matrix C where C[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # Then C @ V[v_idx, :] = ones(n)
                C_mat = np.zeros((n, r))
                for i in range(n):
                    C_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                V_new[v_idx, :] = np.linalg.lstsq(C_mat, np.ones(n), rcond=None)[0]
        return V_new
    
    def residual(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    # Alternating minimization
    best_res = float('inf')
    for iteration in range(500):
        W = solve_W(U, V)
        U = solve_U(U, V, W)
        V = solve_V(U, V, W)
        
        res = residual(U, V, W)
        if res < best_res:
            best_res = res
        
        if res < 1e-10:
            break
        if time.time() > deadline:
            break
    
    if best_res < 1e-8:
        # Convert to list of lists with rounded values
        U_list = [[float(round(x, 10)) for x in row] for row in U]
        V_list = [[float(round(x, 10)) for x in row] for row in V]
        W_list = [[float(round(x, 10)) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    return None
# EVOLVE-BLOCK-END

