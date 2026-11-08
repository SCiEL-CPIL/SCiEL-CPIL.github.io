import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _get_baseline():
    """Return the known rank-34 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen Left (A11 * B1L)
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1
    
    # A12 * B2L
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1
    
    # A21 * B1L
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
    U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1
    
    # A22 * B2L
    U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1
    
    # Strassen Right (A11 * B1R)
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[6][col]=1; W[10][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[7][col]=1; W[10][col]=-1; col+=1
    U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[9][col]=1; W[10][col]=1; col+=1
    U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[7][col]=1; W[6][col]=-1; W[9][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[10][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[6][col]=1; col+=1
    
    # A12 * B2R
    U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1
    
    # A21 * B1R
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
    U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1
    
    # A22 * B2R
    U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
    U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}

def _als_search(target_rank, T, max_time, n_restarts=20):
    """
    Alternating Least Squares search for a decomposition of given rank.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 2:
            break
        
        np.random.seed(restart * 7919 + 42)
        
        # Initialize
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(500):
            if time.time() - start_time > max_time - 2:
                break
            
            # Update U
            for r in range(target_rank):
                rest = np.zeros_like(T)
                for r2 in range(target_rank):
                    if r2 != r:
                        rest += np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                residual = T - rest
                vw = V[:, r][:, None] * W[None, :]
                denom = np.sum(vw ** 2)
                if denom > 1e-10:
                    U[:, r] = np.einsum('abc,bc->a', residual, vw) / denom
            
            # Update V
            for r in range(target_rank):
                rest = np.zeros_like(T)
                for r2 in range(target_rank):
                    if r2 != r:
                        rest += np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                residual = T - rest
                uw = U[:, r][:, None] * W[None, :]
                denom = np.sum(uw ** 2)
                if denom > 1e-10:
                    V[:, r] = np.einsum('abc,ac->b', residual, uw) / denom
            
            # Update W
            for r in range(target_rank):
                rest = np.zeros_like(T)
                for r2 in range(target_rank):
                    if r2 != r:
                        rest += np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                residual = T - rest
                uv = U[:, r][:, None] * V[:, r][None, :]
                denom = np.sum(uv ** 2)
                if denom > 1e-10:
                    W[:, r] = np.einsum('abc,ab->c', residual, uv) / denom
            
            # Check error periodically
            if it % 100 == 0:
                T_recon = np.zeros_like(T)
                for r in range(target_rank):
                    T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.max(np.abs(T_recon - T))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-8:
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    if _verify_decomposition(U_round, V_round, W_round):
                        return U_round.tolist(), V_round.tolist(), W_round.tolist()
    
    return None

def construct():
    deadline = time.time() + 130
    
    baseline = _get_baseline()
    best_result = baseline
    best_rank = 34
    
    T = _build_tensor()
    
    # Try to find lower-rank decompositions
    for target_rank in range(29, best_rank):
        if time.time() > deadline - 15:
            break
        
        remaining = min(12, max(1, deadline - time.time() - 12))
        result = _als_search(target_rank, T, max_time=remaining, n_restarts=15)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
    
    return best_result
