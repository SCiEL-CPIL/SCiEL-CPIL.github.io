"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1;  U[3, 0] = 1
    V[0, 0] = 1;  V[3, 0] = 1
    W[0, 0] = 1;  W[3, 0] = 1
    
    U[2, 1] = 1;  U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1;  W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1;  V[3, 2] = -1
    W[2, 2] = 1;  W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1;  V[0, 3] = -1
    W[0, 3] = 1;  W[1, 3] = 1
    
    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1;  W[2, 4] = 1
    
    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_223_rank10(timeout=30):
    """Try to find a rank-10 decomposition for 2x2x3 using optimization."""
    n, m, p = 2, 2, 3
    rank = 10
    
    # Target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    def objective(params):
        U = params[0:rank*4].reshape(rank, 4).T
        V = params[rank*4:rank*10].reshape(rank, 6).T
        W = params[rank*10:rank*16].reshape(rank, 6).T
        
        loss = 0.0
        for (ui, vi, wi) in target_indices:
            val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
            loss += (val - 1.0) ** 2
        
        # Also penalize non-target positions
        all_positions = set()
        for (ui, vi, wi) in target_indices:
            all_positions.add((ui, vi, wi))
        
        for ui in range(n*m):
            for vi in range(m*p):
                for wi in range(p*n):
                    if (ui, vi, wi) not in all_positions:
                        val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
                        loss += val ** 2
        
        return loss
    
    # Start with Strassen + 3 schoolbook terms
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U0 = np.zeros((4, rank))
    V0 = np.zeros((6, rank))
    W0 = np.zeros((6, rank))
    
    # First 7 terms: Strassen for columns 0,1
    U0[:, 0:7] = U_str
    V0[0:2, 0:7] = V_str[0:2, :]  # j=0
    V0[2:4, 0:7] = V_str[2:4, :]  # j=1, k=0,1
    # For 2x2x3, V has shape (6,7)
    # V_str has shape (4,7) for 2x2x2
    # Need to map: for 2x2x3, V[j*p+k, r] where p=3
    # For columns 0,1: V[0*3+0]=V[0], V[0*3+1]=V[1], V[1*3+0]=V[3], V[1*3+1]=V[4]
    V0 = np.zeros((6, rank))
    V0[0, 0:7] = V_str[0, :]  # j=0,k=0
    V0[1, 0:7] = V_str[1, :]  # j=0,k=1
    V0[3, 0:7] = V_str[2, :]  # j=1,k=0
    V0[4, 0:7] = V_str[3, :]  # j=1,k=1
    
    W0 = np.zeros((6, rank))
    # W_str has shape (4,7) for 2x2x2: W[k*n+i, r]
    # For 2x2x3: W[k*n+i, r] where n=2
    # k=0: W[0], W[1]; k=1: W[2], W[3]
    W0[0, 0:7] = W_str[0, :]  # k=0,i=0
    W0[1, 0:7] = W_str[1, :]  # k=0,i=1
    W0[2, 0:7] = W_str[2, :]  # k=1,i=0
    W0[3, 0:7] = W_str[3, :]  # k=1,i=1
    
    # Terms 7-9: schoolbook for column 2
    U0[0, 7] = 1; V0[0, 7] = 1; W0[4, 7] = 1  # A[0,0]*B[0,2]
    U0[1, 8] = 1; V0[5, 8] = 1; W0[4, 8] = 1  # A[0,1]*B[1,2]
    U0[2, 9] = 1; V0[3, 9] = 1; W0[5, 9] = 1  # A[1,0]*B[0,2]
    # Missing: A[1,1]*B[1,2] - we'll let optimization handle this
    
    params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    start_time = time.time()
    result = minimize(objective, params0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return None
    
    if result.fun < 1e-10:
        U = result.x[0:rank*4].reshape(rank, 4).T
        V = result.x[rank*4:rank*10].reshape(rank, 6).T
        W = result.x[rank*10:rank*16].reshape(rank, 6).T
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
    
    return None

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _build_rank52():
    """Build rank-52 decomposition: 6 Strassen blocks + 1 rank-10 triple block."""
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_223, V_223, W_223 = _try_find_223_rank10(timeout=30)
    
    if U_223 is None:
        return None
    
    total_rank = 6 * 7 + 10  # 52
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(6):
        start_k = b * 2
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7
    
    # Last 3 columns: rank-10 block
    start_k = 12
    U[:, col_offset:col_offset+10] = U_223
    for j in range(m):
        for k_local in range(3):
            k_global = start_k + k_local
            V[j * p + k_global, col_offset:col_offset+10] = V_223[j * 3 + k_local, :]
    for k_local in range(3):
        for i in range(n):
            W[k_local * n + i, col_offset:col_offset+10] = W_223[k_local * 2 + i, :]
    col_offset += 10
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
        best_U, best_V, best_W = U, V, W
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 40:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try rank-52
    result_52 = _build_rank52()
    if result_52 is not None:
        U52, V52, W52 = result_52
        if _verify_decomposition(U52, V52, W52, n, m, p):
            best_rank = 52
            best_U, best_V, best_W = U52, V52, W52
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
