import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s for cleanup/return

    n, m, p = DIMENSIONS
    
    # Fallback: Rank 28 solution from current program
    def get_fallback():
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        blocks = [
            ([0, 1], [0, 1], [0, 1], [0, 1]),
            ([2, 3], [2, 3], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [2, 3], [2, 3]),
            ([2, 3], [2, 3], [2, 3], [2, 3]),
        ]
        
        for blk in range(4):
            ac0, ac1 = blocks[blk][0]
            br0, br1 = blocks[blk][1]
            bc0, bc1 = blocks[blk][2]
            cc0, cc1 = blocks[blk][3]
            base = blk * 7
            
            aa00 = 0 * m + ac0
            aa01 = 0 * m + ac1
            aa10 = 1 * m + ac0
            aa11 = 1 * m + ac1
            
            bb00 = br0 * p + bc0
            bb01 = br0 * p + bc1
            bb10 = br1 * p + bc0
            bb11 = br1 * p + bc1
            
            cc00 = cc0 * n + 0
            cc01 = cc1 * n + 0
            cc10 = cc0 * n + 1
            cc11 = cc1 * n + 1
            
            # M1
            U[aa00][base + 0] += 1; U[aa11][base + 0] += 1
            V[bb00][base + 0] += 1; V[bb11][base + 0] += 1
            W[cc00][base + 0] += 1; W[cc11][base + 0] += 1
            # M2
            U[aa10][base + 1] += 1; U[aa11][base + 1] += 1
            V[bb00][base + 1] += 1
            W[cc10][base + 1] += 1; W[cc11][base + 1] -= 1
            # M3
            U[aa00][base + 2] += 1
            V[bb01][base + 2] += 1; V[bb11][base + 2] -= 1
            W[cc01][base + 2] += 1; W[cc11][base + 2] += 1
            # M4
            U[aa11][base + 3] += 1
            V[bb10][base + 3] += 1; V[bb00][base + 3] -= 1
            W[cc00][base + 3] += 1; W[cc10][base + 3] += 1
            # M5
            U[aa00][base + 4] += 1; U[aa01][base + 4] += 1
            V[bb11][base + 4] += 1
            W[cc00][base + 4] -= 1; W[cc01][base + 4] += 1
            # M6
            U[aa10][base + 5] += 1; U[aa00][base + 5] -= 1
            V[bb00][base + 5] += 1; V[bb01][base + 5] += 1
            W[cc11][base + 5] += 1
            # M7
            U[aa01][base + 6] += 1; U[aa11][base + 6] -= 1
            V[bb10][base + 6] += 1; V[bb11][base + 6] += 1
            W[cc00][base + 6] += 1
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Target Tensor T
    # T[i, j, k] = 1 if A[i, k] * B[k, j] contributes to C[i, j]
    # Indices:
    # i in [0, n)
    # j in [0, m)  (Wait, A is n x m, B is m x p, C is n x p)
    # Task says: A (2x4), B (4x4).
    # n=2, m=4, p=4.
    # U shape (n*m, rank) -> (8, R)
    # V shape (m*p, rank) -> (16, R)
    # W shape (p*n, rank) -> (8, R)
    # W index convention: W[k*n+i, r] where k is col of C (0..p-1), i is row of C (0..n-1).
    # So W index is $k \cdot n + i$.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # A[i, k] is at index i*m + k in U
                # B[k, j] is at index k*p + j in V
                # C[i, j] is at index j*n + i in W
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Try to find rank 27
    target_rank = 27
    if time.time() < deadline - 10:
        # Random initialization
        np.random.seed(42)
        U_init = np.random.randn(n * m, target_rank)
        V_init = np.random.randn(m * p, target_rank)
        W_init = np.random.randn(p * n, target_rank)
        
        # Flatten
        def flatten_params(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            return U, V, W

        def residuals(params):
            U, V, W = flatten_params(params)
            # Compute sum U_r \otimes V_r \otimes W_r
            # Reshape to (8, R, 16, R, 8, R) -> sum over R
            # Efficient way: np.einsum('ir,jr,kr->ijk', U, V, W)
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            return (T_approx - T).flatten()

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use least squares
        # This might be slow or fail to converge to 0 if rank 27 is impossible
        try:
            # Limit iterations/time
            # scipy.optimize.least_squares doesn't have a direct time limit, 
            # but we can check time in the callback? No, callback is for constraints.
            # We'll just run it and hope it finishes or we wrap it in a thread?
            # Simpler: just run it. If it takes too long, we might hit the deadline.
            # To be safe, we can limit the number of iterations.
            res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            
            if res.success or res.cost < 1e-6:
                U_opt, V_opt, W_opt = flatten_params(res.x)
                
                # Check if we can snap to integers/half-integers
                # Try rounding to nearest 0.5
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                T_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_snap, T, atol=1e-9):
                    # Convert to list
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        except Exception:
            pass

    return get_fallback()
