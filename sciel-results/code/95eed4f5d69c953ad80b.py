"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7) indices i*2+j for A
    V: (4, 7) indices j*2+k for B (local 2x2)
    W: (4, 7) indices k*2+i for C (local 2x2)
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00 + a11) * (b00 + b11)
    # C00 += M1, C11 += M1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (a10 + a11) * b00
    # C10 += M2, C11 -= M2
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3 = a00 * (b10 - b11)
    # C01 += M3, C11 += M3
    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = a11 * (b01 - b00)
    # C00 += M4, C10 += M4
    # Note: Standard Strassen C00 = M1 - M4 - M5 + M7 ? No.
    # Let's stick to one standard set.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M7
    # So M4 contributes to C00 (-) and C10 (+)?
    # Wait, M4 = a11(b01 - b00).
    # In C00: -M4. In C10: +M4.
    U[3, 3] = 1.0
    V[1, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = -1.0; W[2, 3] = 1.0
    
    # M5 = (a00 + a01) * b11
    # C00 += M5, C01 += M5
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0; W[1, 4] = 1.0
    
    # M6 = (a10 - a00) * (b00 + b10)
    # C10 += M6
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[2, 5] = 1.0
    W[2, 5] = 1.0
    
    # M7 = (a01 - a11) * (b01 + b11)
    # C00 += M7, C11 += M7
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[1, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0; W[3, 6] = 1.0
    
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Construct a block decomposition for M_{2,2,p}.
    Uses Strassen for 2x2 blocks and Schoolbook for 1x1 blocks.
    """
    if n != 2 or m != 2:
        return None
        
    # Determine block sizes
    # p=15: 7 blocks of 2, 1 block of 1
    block_sizes = []
    remaining = p
    while remaining > 0:
        if remaining >= 2:
            block_sizes.append(2)
            remaining -= 2
        else:
            block_sizes.append(1)
            remaining -= 1
            
    # Precompute Strassen and Schoolbook 1x1 decompositions
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Schoolbook for 2x2x1
    # U shape (4, 4), V shape (2, 4), W shape (2, 4)
    # Note: V and W local indices are 0..1 for j and k respectively.
    U_1 = np.zeros((4, 4), dtype=np.float64)
    V_1 = np.zeros((2, 4), dtype=np.float64)
    W_1 = np.zeros((2, 4), dtype=np.float64)
    col = 0
    for i in range(2):
        for j in range(2):
            U_1[i * 2 + j, col] = 1.0
            V_1[j, col] = 1.0
            W_1[i, col] = 1.0
            col += 1
            
    # Calculate total rank
    total_rank = 0
    for bs in block_sizes:
        if bs == 2:
            total_rank += 7
        elif bs == 1:
            total_rank += 4
            
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    for bs in block_sizes:
        if bs == 2:
            for r in range(7):
                # U: same for all k in this block
                # Global U index: i * m + j
                U[0, col_offset] = U_s[0, r]  # a00
                U[1, col_offset] = U_s[1, r]  # a01
                U[2, col_offset] = U_s[2, r]  # a10
                U[3, col_offset] = U_s[3, r]  # a11
                
                # V: for j=0,1 and k=k_offset, k_offset+1
                # Global V index: j * p + k
                # Local V indices: 0->(j=0,k=0), 1->(j=0,k=1), 2->(j=1,k=0), 3->(j=1,k=1)
                V[0 * p + k_offset, col_offset] = V_s[0, r]
                V[0 * p + k_offset + 1, col_offset] = V_s[1, r]
                V[1 * p + k_offset, col_offset] = V_s[2, r]
                V[1 * p + k_offset + 1, col_offset] = V_s[3, r]
                
                # W: for k=k_offset, k_offset+1 and i=0,1
                # Global W index: k * n + i
                # Local W indices: 0->(k=0,i=0), 1->(k=0,i=1), 2->(k=1,i=0), 3->(k=1,i=1)
                # Wait, Strassen W_s was defined for 2x2 output C.
                # C indices: 0->c00, 1->c01, 2->c10, 3->c11 ?
                # My _strassen_2x2x2 used W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11?
                # Let's check _strassen_2x2x2 W assignments:
                # W[0, 0] = 1.0 (c00)
                # W[1, 2] = 1.0 (c01) -> W[1] is c01
                # W[2, 1] = 1.0 (c10) -> W[2] is c10
                # W[3, 0] = 1.0 (c11) -> W[3] is c11
                # So local W index 0 corresponds to C[0,0] (i=0, k=0)
                # local W index 1 corresponds to C[0,1] (i=0, k=1)
                # local W index 2 corresponds to C[1,0] (i=1, k=0)
                # local W index 3 corresponds to C[1,1] (i=1, k=1)
                
                # Global W index: k * n + i
                # For k=k_offset, i=0: idx = k_offset * 2 + 0
                # For k=k_offset, i=1: idx = k_offset * 2 + 1
                # For k=k_offset+1, i=0: idx = (k_offset+1) * 2 + 0
                # For k=k_offset+1, i=1: idx = (k_offset+1) * 2 + 1
                
                # Map local W_s[0] (c00) -> Global W[k_offset*2+0]
                W[k_offset * n + 0, col_offset] = W_s[0, r]
                # Map local W_s[1] (c01) -> Global W[(k_offset+1)*2+0]
                W[(k_offset + 1) * n + 0, col_offset] = W_s[1, r]
                # Map local W_s[2] (c10) -> Global W[k_offset*2+1]
                W[k_offset * n + 1, col_offset] = W_s[2, r]
                # Map local W_s[3] (c11) -> Global W[(k_offset+1)*2+1]
                W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r]
                
                col_offset += 1
            k_offset += 2
        elif bs == 1:
            for r in range(4):
                # U: a00, a01, a10, a11
                U[0, col_offset] = U_1[0, r]
                U[1, col_offset] = U_1[1, r]
                U[2, col_offset] = U_1[2, r]
                U[3, col_offset] = U_1[3, r]
                
                # V: b0, b1
                # Global V index: j * p + k
                V[0 * p + k_offset, col_offset] = V_1[0, r]
                V[1 * p + k_offset, col_offset] = V_1[1, r]
                
                # W: c0, c1
                # Global W index: k * n + i
                W[k_offset * n + 0, col_offset] = W_1[0, r]
                W[k_offset * n + 1, col_offset] = W_1[1, r]
                
                col_offset += 1
            k_offset += 1
            
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank by attempting to eliminate one column at a time.
    This is a heuristic: for each column, check if it can be expressed as a 
    linear combination of others (in the tensor sense).
    """
    start_time = time.time()
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    while current_rank > target_rank:
        if time.time() > deadline:
            break
            
        # Try to eliminate each column
        eliminated = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline:
                break
                
            # Check if removing this column still gives a valid decomposition
            # (This is expensive, so we do a quick check first)
            U_test = np.delete(current_U, col_to_remove, axis=1)
            V_test = np.delete(current_V, col_to_remove, axis=1)
            W_test = np.delete(current_W, col_to_remove, axis=1)
            
            if _is_valid(U_test, V_test, W_test, n, m, p):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank -= 1
                eliminated = True
                break
        
        if not eliminated:
            break
    
    if current_rank < U.shape[1]:
        return current_U, current_V, current_W
    return None

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try block decomposition first
    U, V, W = _block_decomposition(n, m, p)
    
    # Verify validity
    if U is not None and _is_valid(U, V, W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
    else:
        # Fallback to schoolbook if needed
        U, V, W = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
    
    # Try to find a lower-rank decomposition using nonlinear optimization
    # The tensor constraint: for all (i,k), sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is 2*15 = 30 scalar equations.
    
    if time.time() < deadline - 30:
        # Try ranks from best_rank-1 down to a lower bound
        # Lower bound: max(4, 2*15, 2*15) = 30, but likely higher
        # Try a few candidate ranks
        candidate_ranks = [r for r in range(best_rank - 1, max(29, best_rank - 10), -1)]
        
        for target_rank in candidate_ranks:
            if time.time() > deadline - 5:
                break
            
            # Try a few random initializations
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization
                rng = np.random.RandomState(attempt * 1000 + target_rank)
                U_init = rng.randn(n * m, target_rank) * 0.1
                V_init = rng.randn(m * p, target_rank) * 0.1
                W_init = rng.randn(p * n, target_rank) * 0.1
                
                # Define the residual function
                def _residual(params):
                    U_r = params[0:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_r = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Compute residuals for each (i,k)
                    residuals = []
                    for i in range(n):
                        for k in range(p):
                            val = 0.0
                            for j in range(m):
                                for r in range(target_rank):
                                    val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                            residuals.append(val - 1.0)
                    return residuals
                
                # Try to solve
                try:
                    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                    result = minimize(_residual, x0, method='lm', options={'maxiter': 1000, 'xatol': 1e-12})
                    
                    if result.success and np.allclose(result.fun, 0, atol=1e-8):
                        U_sol = result.x[0:n*m*target_rank].reshape(n*m, target_rank)
                        V_sol = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                        W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                        
                        if _is_valid(U_sol, V_sol, W_sol, n, m, p):
                            best_U, best_V, best_W = U_sol, V_sol, W_sol
                            best_rank = target_rank
                            break  # Found valid decomposition at this rank
                except Exception:
                    pass
        
        # Also try to remove redundant columns from the block decomposition
        if time.time() < deadline - 10 and best_rank > 30:
            reduced = _try_rank_reduction(best_U, best_V, best_W, n, m, p, max(29, best_rank - 5), deadline - 5)
            if reduced is not None:
                best_U, best_V, best_W = reduced
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def construct():
    return _construct()
# EVOLVE-BLOCK-END
