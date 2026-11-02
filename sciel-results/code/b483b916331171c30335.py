import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 140.0  # safety margin from 150s kill

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. U,V,W are lists of lists of ints/halves."""
    rank = len(U[0]) if U else 0
    # Build tensor from factors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            expected = 1 if (j == j2 and i == i2 and k == k2) else 0
                            actual = 0
                            for c in range(rank):
                                actual += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i2][c]
                            if actual != expected:
                                return False
    return True

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2 block (terms 0-6)
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[0][1] = 1; U[3][1] = 1; V[3][1] = 1; W[1][1] = 1
    U[1][2] = 1; V[0][2] = 1; V[1][2] = 1; W[2][2] = 1
    U[4][3] = 1; V[3][3] = 1; V[4][3] = 1; W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1; V[4][4] = 1; V[3][4] = -1; W[0][4] = -1; W[3][4] = 1
    U[3][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
    U[1][6] = 1; U[4][6] = -1; V[3][6] = 1; V[4][6] = 1; W[0][6] = 1; W[3][6] = -1
    
    # Right column schoolbook (terms 7-10)
    U[0][7] = 1; V[2][7] = 1; W[4][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[4][8] = 1
    U[3][9] = 1; V[2][9] = 1; W[5][9] = 1
    U[4][10] = 1; V[5][10] = 1; W[5][10] = 1
    
    # Outer product row 2 (terms 11-16)
    U[2][11] = 1; V[6][11] = 1; W[0][11] = 1
    U[2][12] = 1; V[7][12] = 1; W[2][12] = 1
    U[2][13] = 1; V[8][13] = 1; W[4][13] = 1
    U[5][14] = 1; V[6][14] = 1; W[1][14] = 1
    U[5][15] = 1; V[7][15] = 1; W[3][15] = 1
    U[5][16] = 1; V[8][16] = 1; W[5][16] = 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-18 decomposition."""
    n, m, p = 2, 3, 3
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
    return U, V, W

def _round_to_half(x, tol=1e-6):
    """Round x to nearest half-integer. Return None if too far."""
    h = round(2 * x) / 2.0
    if abs(x - h) < tol:
        return h
    return None

def _try_als_search(rank, n, m, p, max_iters=200, seed=42, time_budget=30.0):
    """Try ALS optimization to find rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Target tensor: T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j', i=i', k=k'
    # Shape: (n*m, m*p, p*n) = (6, 9, 6)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_loss = np.inf
    best_factors = None
    
    for attempt in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        # Scale to match tensor magnitude
        U *= (n * m * p) ** 0.1
        V *= (m * p * p * n) ** 0.1
        W *= (p * n * n * m) ** 0.1
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each row a, solve least squares
            for a in range(n * m):
                # T[a, :, :] = U[a,:] * (V * W) -> U[a,:] = T[a] / (V*W)
                # More precisely: sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
                # Fix V,W: minimize sum_{b,c_idx} (sum_c U[a,c] V[b,c] W[c_idx,c] - T[a,b,c_idx])^2
                # This is: U[a,:] @ (V[:,c] * W[:,c]) = T[a,:,:] reshaped
                # Let M[b, c_idx, c] = V[b,c] * W[c_idx,c]
                # Then U[a,:] @ M[b,c_idx,:] = T[a,b,c_idx]
                # Equivalent: reshape T[a] to (m*p*p*n) and M to (m*p*p*n, rank)
                M = np.einsum('bc,dc->bdc', V, W)  # (m*p, p*n, rank)
                M_flat = M.reshape(-1, rank)  # (m*p*p*n, rank)
                T_flat = T[a].reshape(-1)  # (m*p*p*n,)
                # Solve: U[a,:] @ M_flat.T = T_flat
                if rank <= M_flat.shape[0]:
                    sol, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat, rcond=None)
                    U[a] = sol
            
            # Update V
            for b in range(m * p):
                M = np.einsum('ac,dc->adc', U, W)  # (n*m, p*n, rank)
                M_flat = M.reshape(-1, rank)
                T_flat = T[:, b, :].reshape(-1)
                if rank <= M_flat.shape[0]:
                    sol, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat, rcond=None)
                    V[b] = sol
            
            # Update W
            for c_idx in range(p * n):
                M = np.einsum('ac,bc->abc', U, V)  # (n*m, m*p, rank)
                M_flat = M.reshape(-1, rank)
                T_flat = T[:, :, c_idx].reshape(-1)
                if rank <= M_flat.shape[0]:
                    sol, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat, rcond=None)
                    W[c_idx] = sol
            
            # Compute loss
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            loss = np.sum((T_recon - T) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_factors = (U.copy(), V.copy(), W.copy())
            
            # Check if we've converged to exact solution
            if loss < 1e-10:
                break
        
        # Try to round best_factors to half-integers
        if best_factors is not None:
            Uf, Vf, Wf = best_factors
            U_rounded = np.zeros_like(Uf)
            V_rounded = np.zeros_like(Vf)
            W_rounded = np.zeros_like(Wf)
            valid = True
            for i in range(Uf.shape[0]):
                for j in range(Uf.shape[1]):
                    r = _round_to_half(Uf[i, j])
                    if r is None:
                        valid = False
                        break
                    U_rounded[i, j] = r
                if not valid:
                    break
            if valid:
                for i in range(Vf.shape[0]):
                    for j in range(Vf.shape[1]):
                        r = _round_to_half(Vf[i, j])
                        if r is None:
                            valid = False
                            break
                        V_rounded[i, j] = r
                    if not valid:
                        break
            if valid:
                for i in range(Wf.shape[0]):
                    for j in range(Wf.shape[1]):
                        r = _round_to_half(Wf[i, j])
                        if r is None:
                            valid = False
                            break
                        W_rounded[i, j] = r
                    if not valid:
                        break
            if valid:
                # Verify exactness
                U_list = [[int(U_rounded[i][j] * 2) // 2 if U_rounded[i][j] % 1 == 0 else U_rounded[i][j] for j in range(rank)] for i in range(n*m)]
                # Simpler: just check with floats
                T_recon = np.einsum('ac,bc,dc->abd', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_recon, T, atol=1e-9):
                    # Convert to list format
                    U_out = [[U_rounded[i, j] for j in range(rank)] for i in range(n * m)]
                    V_out = [[V_rounded[i, j] for j in range(rank)] for i in range(m * p)]
                    W_out = [[W_rounded[i, j] for j in range(rank)] for i in range(p * n)]
                    # Verify with exact arithmetic
                    if _verify(U_out, V_out, W_out, n, m, p):
                        return U_out, V_out, W_out
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the known rank-17 solution
    U17, V17, W17 = _build_rank17()
    if _verify(U17, V17, W17, n, m, p):
        best_result = {"U": U17, "V": V17, "W": W17}
        best_rank = 17
    else:
        U18, V18, W18 = _build_schoolbook()
        best_result = {"U": U18, "V": V18, "W": W18}
        best_rank = 18
    
    # Try to find lower rank with ALS search
    for target_rank in range(16, max(16, best_rank - 2), -1):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        result = _try_als_search(target_rank, n, m, p, max_iters=150, seed=target_rank * 100, time_budget=min(25.0, TIME_LIMIT - 10 - (time.time() - start_time)))
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = target_rank
                break
    
    return best_result
