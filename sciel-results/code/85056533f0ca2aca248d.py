"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 135  # seconds, leave 15s margin for 150s limit


def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return n * m * p == 0
    
    def to_int2(x):
        v = int(round(x * 2))
        if abs(x * 2 - v) > 1e-9:
            return None
        return v
    
    U2 = [[to_int2(x) for x in row] for row in U]
    V2 = [[to_int2(x) for x in row] for row in V]
    W2 = [[to_int2(x) for x in row] for row in W]
    
    for mat in [U2, V2, W2]:
        for row in mat:
            if any(x is None for x in row):
                return False
    
    U3 = np.array(U2).reshape(n, m, rank)
    V3 = np.array(V2).reshape(m, p, rank)
    W3 = np.array(W2).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if s != 4:
                    return False
    return True


def _verify_float(U, V, W, n, m, p, atol=1e-9):
    """Verify using floating point."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        recon += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
    return np.allclose(recon, 1.0, atol=atol)


def _round_and_verify(U, V, W, n, m, p):
    """Try multiple rounding strategies and return valid one or None."""
    strategies = [
        lambda x: round(x * 2) / 2,  # half-integers
        lambda x: int(round(x)),     # integers
        lambda x: (1 if x > 0.5 else (-1 if x < -0.5 else 0)),  # {-1,0,1}
        lambda x: (1 if x > 0.25 else (-1 if x < -0.25 else 0)),  # aggressive {-1,0,1}
    ]
    
    for strat in strategies:
        U_r = [[strat(x) for x in row] for row in U]
        V_r = [[strat(x) for x in row] for row in V]
        W_r = [[strat(x) for x in row] for row in W]
        if _verify_exact(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    return None


def _build_schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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


def _als_decompose(n, m, p, rank, seed, max_iters=300):
    """ALS for tensor decomposition with better convergence."""
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    prev_res = np.inf
    stall = 0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_res = np.inf
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                Vj = V[j * p:(j + 1) * p, :]
                Wi = W[np.arange(p) * n + i, :]
                M = (Vj * Wi).T  # (rank, p)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                Ui = U[np.arange(n) * m + j, :]
                Wk = W[k * n:(k + 1) * n, :]
                M = (Ui * Wk).T  # (rank, n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_slice = U.reshape(n, m, rank)[i, :, :]
                Vk = V[np.arange(m) * p + k, :]
                M = (U_slice * Vk).T  # (rank, m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute residual
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            recon += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        res = np.max(np.abs(recon - 1.0))
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if res < prev_res:
            prev_res = res
            stall = 0
        else:
            stall += 1
        
        if res < 1e-12:
            return U, V, W
        
        if stall > 10:
            noise_scale = 0.3 if it % 20 == 10 else 0.1
            U += rng.standard_normal((n * m, rank)) * noise_scale
            V += rng.standard_normal((m * p, rank)) * noise_scale
            W += rng.standard_normal((p * n, rank)) * noise_scale
            stall = 0
    
    if best_res < 0.1:
        return best_U, best_V, best_W
    return None


def _als_binary_init(n, m, p, rank, seed, max_iters=200):
    """ALS with binary initialization."""
    rng = np.random.default_rng(seed)
    
    U = rng.choice([-1, 1], size=(n * m, rank))
    V = rng.choice([-1, 1], size=(m * p, rank))
    W = rng.choice([-1, 1], size=(p * n, rank))
    
    # Run a few ALS steps from binary init
    prev_res = np.inf
    stall = 0
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        for i in range(n):
            for j in range(m):
                Vj = V[j * p:(j + 1) * p, :]
                Wi = W[np.arange(p) * n + i, :]
                M = (Vj * Wi).T
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                Ui = U[np.arange(n) * m + j, :]
                Wk = W[k * n:(k + 1) * n, :]
                M = (Ui * Wk).T
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                U_slice = U.reshape(n, m, rank)[i, :, :]
                Vk = V[np.arange(m) * p + k, :]
                M = (U_slice * Vk).T
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            recon += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        res = np.max(np.abs(recon - 1.0))
        
        if res < prev_res:
            prev_res = res
            stall = 0
        else:
            stall += 1
        
        if res < 1e-12:
            return U, V, W
        
        if stall > 8:
            U += rng.standard_normal((n * m, rank)) * 0.2
            V += rng.standard_normal((m * p, rank)) * 0.2
            W += rng.standard_normal((p * n, rank)) * 0.2
            stall = 0
    
    if res < 0.1:
        return U, V, W
    return None


def construct():
    n, m, p = 2, 4, 7
    
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 28
    for rank in range(28, best_rank):
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        
        # More trials for lower ranks
        if rank <= 30:
            num_trials = 6
        elif rank <= 35:
            num_trials = 4
        elif rank <= 42:
            num_trials = 3
        else:
            num_trials = 2
        
        for trial in range(num_trials):
            if time.time() - _START_TIME > _TIME_LIMIT - 15:
                break
            
            seed = rank * 1000 + trial * 137
            
            # Alternate between different initialization strategies
            if trial % 3 == 0:
                res = _als_decompose(n, m, p, rank, seed)
            elif trial % 3 == 1:
                res = _als_binary_init(n, m, p, rank, seed)
            else:
                res = _als_decompose(n, m, p, rank, seed + 999)
            
            if res is not None:
                U, V, W = res
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                rounded = _round_and_verify(U_list, V_list, W_list, n, m, p)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r, "V": V_r, "W": W_r}
    
    return best
