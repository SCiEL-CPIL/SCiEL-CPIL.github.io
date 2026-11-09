"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _compute_error(U, V, W):
    """Compute sum of squared errors efficiently using einsum."""
    n, m, p = DIMENSIONS
    # Reshape for tensor contraction
    # U shape: (n*m, rank) -> (n, m, rank)
    # V shape: (m*p, rank) -> (m, p, rank)  
    # W shape: (p*n, rank) -> (p, n, rank)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.sum((T - 1.0) ** 2)
    return err

def _als_decompose(max_rank, deadline, seed=42, max_iters=300):
    """Try to find a decomposition of given rank using vectorized ALS."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    rank = max_rank
    if rank <= 0:
        return None
    
    # Initialize randomly with multiple strategies
    best_result = None
    best_err = float('inf')
    
    for init_seed in [seed, seed+1, seed+2, seed+3, seed+4]:
        if time.time() > deadline - 15:
            break
        
        rng_init = np.random.RandomState(init_seed)
        
        # Try different initializations
        for init_type in ['randn', 'rand', 'sparse']:
            if time.time() > deadline - 15:
                break
            
            if init_type == 'randn':
                U = rng_init.randn(n*m, rank)
                V = rng_init.randn(m*p, rank)
                W = rng_init.randn(p*n, rank)
            elif init_type == 'rand':
                U = rng_init.rand(n*m, rank)
                V = rng_init.rand(m*p, rank)
                W = rng_init.rand(p*n, rank)
            else:  # sparse
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                # Randomly set some entries
                for _ in range(n*m):
                    U[rng_init.randint(n*m), rng_init.randint(rank)] = rng_init.choice([-1, 1])
                for _ in range(m*p):
                    V[rng_init.randint(m*p), rng_init.randint(rank)] = rng_init.choice([-1, 1])
                for _ in range(p*n):
                    W[rng_init.randint(p*n), rng_init.randint(rank)] = rng_init.choice([-1, 1])
            
            cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
            cur_err = _compute_error(cur_U, cur_V, cur_W)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve linear system
                # A[c,k] = V[j*p+k, c] * W[k*n+i, c]
                # Solve: A @ u_row.T = ones(p)
                for i in range(n):
                    for j in range(m):
                        # Vectorized: A = (V[j*p:(j+1)*p, :] * W[::n, i].reshape(p, 1)).T
                        V_slice = V[j*p:(j+1)*p, :]  # (p, rank)
                        W_slice = W[::n, i]  # (p,)
                        A = (V_slice * W_slice[:, None]).T  # (rank, p)
                        try:
                            cur_U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve linear system
                # B[c,i] = U[i*m+j, c] * W[k*n+i, c]
                for j in range(m):
                    for k in range(p):
                        U_slice = U[:, j]  # (n,) - column j of U reshaped
                        # Actually U is (n*m, rank), so U[i*m+j, :] for all i
                        U_col = np.array([cur_U[i*m+j, :] for i in range(n)])  # (n, rank)
                        W_col = np.array([cur_W[k*n+i, :] for i in range(n)])  # (n, rank)
                        B = (U_col * W_col).T  # (rank, n)
                        try:
                            cur_V[j*p+k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve linear system
                # C[c,j] = U[i*m+j, c] * V[j*p+k, c]
                for k in range(p):
                    for i in range(n):
                        U_row = np.array([cur_U[i*m+j, :] for j in range(m)])  # (m, rank)
                        V_row = np.array([cur_V[j*p+k, :] for j in range(m)])  # (m, rank)
                        C = (U_row * V_row).T  # (rank, m)
                        try:
                            cur_W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = _compute_error(cur_U, cur_V, cur_W)
                if err < cur_err:
                    cur_err = err
                    U, V, W = cur_U.copy(), cur_V.copy(), cur_W.copy()
                
                if cur_err < 1e-12:
                    break
            
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
            
            if best_err < 1e-9:
                break
    
    # Check if we found an exact solution
    if best_err < 1e-9:
        # Try rounding to nearest half-integer
        for round_factor in [2, 1]:
            U_list = [[round(x * round_factor) / round_factor for x in row] for row in best_U]
            V_list = [[round(x * round_factor) / round_factor for x in row] for row in best_V]
            W_list = [[round(x * round_factor) / round_factor for x in row] for row in best_W]
            
            if _verify(U_list, V_list, W_list, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try integer rounding
        U_list = [[round(x) for x in row] for row in best_U]
        V_list = [[round(x) for x in row] for row in best_V]
        W_list = [[round(x) for x in row] for row in best_W]
        if _verify(U_list, V_list, W_list, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    # Start with schoolbook
    best_result = _schoolbook()
    best_rank = len(best_result['U'][0])
    
    # Try to find lower rank decompositions
    # Focus on promising ranks: known lower bounds suggest rank >= 9 for 2x3x5
    # Try ranks from best_rank-1 down to 9
    for target_rank in range(best_rank - 1, 8, -1):
        if time.time() > deadline - 20:
            break
        result = _als_decompose(target_rank, deadline, seed=target_rank * 100)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
