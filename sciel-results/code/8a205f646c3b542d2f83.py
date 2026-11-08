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

def _verify_decomposition(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_calc += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T_calc[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, max_time=30.0):
    """
    Try to reduce rank by finding linear dependencies.
    Uses the fact that if sum_r U[:,r] V[:,r] W[:,r] = T, then
    we can try to express one column as a combination of others.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if not _verify_decomposition(U, V, W):
        return U, V, W
    
    # Build the constraint system
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Try to find if any column can be eliminated by adjusting others
    # This is a nonlinear problem, so we use a heuristic approach
    
    # First, try simple column removal
    improved_U, improved_V, improved_W = U.copy(), V.copy(), W.copy()
    current_rank = improved_U.shape[1]
    
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
        
        removed_any = False
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            
            # Try removing column r
            mask = [i for i in range(current_rank) if i != r]
            if len(mask) == 0:
                continue
                
            U_test = improved_U[:, mask]
            V_test = improved_V[:, mask]
            W_test = improved_W[:, mask]
            
            if _verify_decomposition(U_test, V_test, W_test):
                improved_U = U_test
                improved_V = V_test
                improved_W = W_test
                current_rank = len(mask)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return improved_U, improved_V, improved_W

def _find_rank_5_2x3x1():
    """
    Find a rank-5 decomposition for 2x3x1 matrix multiplication.
    C_i = sum_j A_ij * B_j for i=0,1
    """
    n, m, p = 2, 3, 1
    rank = 5
    
    # Tensor T: T[i*m+j, j, i] = 1 (W index is k*n+i = 0*2+i = i since p=1, k=0)
    T = np.zeros((n*m, m, p*n))
    for i in range(n):
        for j in range(m):
            T[i*m+j, j, i] = 1.0
    
    T_flat = T.flatten()
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for trial in range(50):
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m, rank)
        W = np.random.randn(p*n, rank)
        
        # Scale to get closer to solution
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        def error(params):
            u = params[0: n*m*rank].reshape(n*m, rank)
            v = params[n*m*rank: n*m*rank + m*rank].reshape(m, rank)
            w = params[n*m*rank + m*rank:].reshape(p*n, rank)
            
            T_calc = np.zeros((n*m, m, p*n))
            for r in range(rank):
                T_calc += np.einsum('a,b,c->abc', u[:,r], v[:,r], w[:,r])
            
            return np.sum((T_calc.flatten() - T_flat)**2)
        
        try:
            from scipy.optimize import minimize
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            res = minimize(error, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-12})
            
            if res.fun < 1e-8:
                u = res.x[0: n*m*rank].reshape(n*m, rank)
                v = res.x[n*m*rank: n*m*rank + m*rank].reshape(m, rank)
                w = res.x[n*m*rank + m*rank:].reshape(p*n, rank)
                
                # Verify
                T_calc = np.zeros((n*m, m, p*n))
                for r in range(rank):
                    T_calc += np.einsum('a,b,c->abc', u[:,r], v[:,r], w[:,r])
                
                if np.allclose(T_calc, T, atol=1e-8):
                    return u, v, w
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    # Try to find rank 5 decomposition for 2x3x1
    rank_5_decomp = None
    if time.time() < DEADLINE - 30:
        rank_5_decomp = _find_rank_5_2x3x1()
        
    if rank_5_decomp is not None:
        u, v, w = rank_5_decomp
        # Construct full decomposition
        rank = 5 * p  # 5 * 10 = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for k in range(p):
            # Copy the rank 5 decomp for column k
            U[:, k*5 : (k+1)*5] = u
            for j in range(m):
                V[j*p+k, k*5 : (k+1)*5] = v[j, :]
            for i in range(n):
                W[k*n+i, k*5 : (k+1)*5] = w[i, :]
                
        # Check if it's exact
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 55
    U, V, W = _construct_rank_55()
    
    # Try local improvement to reduce rank
    if time.time() < DEADLINE - 10:
        U, V, W = _try_reduce_rank(U, V, W, max_time=DEADLINE - time.time() - 5)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
