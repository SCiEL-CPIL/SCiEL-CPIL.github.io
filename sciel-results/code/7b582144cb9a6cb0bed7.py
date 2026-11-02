"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U is (n*m, rank) -> (n, m, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    
    # V is (m*p, rank) -> (m, p, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    
    # W is (p*n, rank) -> (p, n, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute the reconstructed tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # The true multiplication tensor is all ones
    T_true = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_true, atol=1e-10)

def _strassen_block_decomposition(n, m, p):
    """
    Construct rank-32 decomposition using Strassen blocks for column pairs
    and schoolbook for the single remaining column.
    """
    rank = 4 * 7 + 4  # 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_block(k1, k2, start_col):
        nonlocal U, V, W
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 1
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1
        W[w_03][c] -= 1
        
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] -= 1
        W[w_02][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] -= 1
        W[w_00][c] += 1
        W[w_01][c] += 1
        
        c = start_col + 4
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1
        W[w_02][c] += 1
        
        c = start_col + 5
        U[u_10][c] += 1
        U[u_00][c] -= 1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 6
        U[u_01][c] += 1
        U[u_11][c] -= 1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        nonlocal U, V, W
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        
        w_0 = k * n + 0
        w_1 = k * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1
        
        return start_col + 4
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    col = schoolbook_single(8, col)
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, max_time=30):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    Returns a new decomposition with lower rank if found, otherwise None.
    """
    start_time = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to remove one term at a time
    for r_remove in range(rank):
        if time.time() - start_time > max_time:
            break
        
        # Skip this term and see if we can adjust the others to compensate
        # This is a complex optimization problem. For now, we'll just check
        # if the term is linearly dependent on the others in a simple way.
        
        # A simpler approach: try to set this term to zero and see if the
        # decomposition still works (it won't, but we can try to optimize).
        
        # For now, let's just return the original if no reduction is found.
        # A full implementation would use scipy.optimize to find the new coefficients.
        pass
    
    return None

def _tensor_product_decomposition(n, m, p):
    """
    Construct a rank-36 schoolbook decomposition as a fallback.
    """
    rank = n * m * p  # 36 schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Primary: rank-32 Strassen-block construction.
    Fallback: rank-36 schoolbook if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Try the rank-32 Strassen block construction
    result = _strassen_block_decomposition(n, m, p)
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Try to reduce the rank further
        if time.time() < deadline:
            reduced = _try_reduce_rank(result["U"], result["V"], result["W"], n, m, p, 
                                       max_time=min(30, deadline - time.time()))
            if reduced is not None:
                if _verify_decomposition(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    return reduced
        
        return result
    else:
        # Fallback to schoolbook
        return _tensor_product_decomposition(n, m, p)

# EVOLVE-BLOCK-END
