import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_55():
    """
    Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication.
    Based on 5 blocks of 2 columns, each using 11 ranks.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over 5 blocks of 2 columns
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _find_rank_5_decomp():
    """
    Attempts to find a rank 5 decomposition for 2x3x1 matrix multiplication.
    Returns (U, V, W) of shapes (6,5), (3,5), (2,5) or None.
    """
    # Tensor T for 2x3x1
    # T[i, j, k] where i in {0,1}, j in {0,1,2}, k in {0,1}
    # T[i, j, k] = 1 if k == i else 0? No.
    # C_i = sum_j A_ij B_j.
    # Tensor indices: (i,j) for A, j for B, i for C.
    # T[(i,j), j, i] = 1.
    
    n, m, p = 2, 3, 1
    rank = 5
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m, rank)
    W = np.random.randn(p*n, rank)
    
    # Target tensor T
    T = np.zeros((n*m, m, p*n))
    for i in range(n):
        for j in range(m):
            T[i*m+j, j, p*n//2 + i] = 1.0 # Wait, W index is k*n+i. Here k is not in T index?
            # T index is (u_idx, v_idx, w_idx).
            # u_idx = i*m+j
            # v_idx = j
            # w_idx = i (since p=1, k=0? No, W index is k*n+i. For p=1, k=0. So w_idx = i).
            # So T[i*m+j, j, i] = 1.
    
    T = np.zeros((n*m, m, p*n))
    for i in range(n):
        for j in range(m):
            T[i*m+j, j, i] = 1.0

    # Flatten T
    T_flat = T.flatten()
    
    def error(params):
        u = params[0: n*m*rank].reshape(n*m, rank)
        v = params[n*m*rank: n*m*rank + m*rank].reshape(m, rank)
        w = params[n*m*rank + m*rank:].reshape(p*n, rank)
        
        # Compute sum of rank 1 tensors
        # T_calc[u_idx, v_idx, w_idx] = sum_r u[u_idx, r] v[v_idx, r] w[w_idx, r]
        T_calc = np.zeros((n*m, m, p*n))
        for r in range(rank):
            T_calc += np.einsum('a,b,c->abc', u[:,r], v[:,r], w[:,r])
            
        return np.sum((T_calc.flatten() - T_flat)**2)

    # Use scipy minimize
    try:
        from scipy.optimize import minimize
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        res = minimize(error, x0, method='BFGS', options={'maxiter': 100})
        if res.fun < 1e-6:
            u = res.x[0: n*m*rank].reshape(n*m, rank)
            v = res.x[n*m*rank: n*m*rank + m*rank].reshape(m, rank)
            w = res.x[n*m*rank + m*rank:].reshape(p*n, rank)
            return u, v, w
    except Exception:
        pass
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    # Try to find rank 5 decomposition
    rank_5_decomp = None
    if time.time() < DEADLINE - 20:
        rank_5_decomp = _find_rank_5_decomp()
        
    if rank_5_decomp is not None:
        u, v, w = rank_5_decomp
        # Construct full decomposition
        rank = 5 * p # 5 * 10 = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for k in range(p):
            # Copy the rank 5 decomp for column k
            # U is same for all k? No, U index is (i,j). It doesn't depend on k.
            # But in the full tensor, U index is (i,j) and V index is (j,k).
            # So for each k, we add a block.
            # U[:, k*5 : (k+1)*5] = u
            # V[:, k*5 : (k+1)*5] = v (but V index is (j,k), so V[j, k] is the entry)
            # Wait, V is (m*p, rank). Index is j*p + k.
            # So V[j*p+k, r] = v[j, r]
            # W is (p*n, rank). Index is k*n + i.
            # So W[k*n+i, r] = w[i, r]
            
            U[:, k*5 : (k+1)*5] = u
            for j in range(m):
                V[j*p+k, k*5 : (k+1)*5] = v[j, :]
            for i in range(n):
                W[k*n+i, k*5 : (k+1)*5] = w[i, :]
                
        # Verify
        # T_calc[i*m+j, j*p+k, k*n+i] should be 1
        # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # = Sum_{r=0}^4 u[i*m+j, r] v[j, r] w[i, r] (for r in block k)
        # This matches the rank 5 decomp.
        
        U_int = np.rint(U).astype(int)
        V_int = np.rint(V).astype(int)
        W_int = np.rint(W).astype(int)
        
        # Check if it's exact
        # Reconstruct tensor
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_calc += np.einsum('a,b,c->abc', U_int[:,r], V_int[:,r], W_int[:,r])
            
        # Check non-zero entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T_calc[i*m+j, j*p+k, k*n+i] != 1.0:
                        valid = False
        if valid:
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
    # Fallback to rank 55
    U, V, W = _construct_rank_55()
    U, V, W = np.rint(U).astype(int), np.rint(V).astype(int), np.rint(W).astype(int)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
