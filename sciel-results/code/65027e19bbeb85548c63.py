import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def try_reduce_rank(U_init, V_init, W_init, target_rank, max_time=20):
        """
        Uses scipy.optimize.least_squares to try to find a decomposition of target_rank.
        Returns (U, V, W) if successful, else (None, None, None).
        """
        R = target_rank
        # Flatten initial matrices to target rank (pad with zeros if necessary, though we expect init to be larger)
        # We take the first R columns of the init matrices
        U0 = U_init[:, :R].copy()
        V0 = V_init[:, :R].copy()
        W0 = W_init[:, :R].copy()
        
        # Add small noise to escape symmetry
        np.random.seed(42)
        U0 += np.random.randn(*U0.shape) * 0.01
        V0 += np.random.randn(*V0.shape) * 0.01
        W0 += np.random.randn(*W0.shape) * 0.01

        params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def residuals(params):
            U = params[:U_dim * R].reshape(U_dim, R)
            V = params[U_dim * R : U_dim * R + V_dim * R].reshape(V_dim, R)
            W = params[U_dim * R + V_dim * R :].reshape(W_dim, R)
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            return (prod - T).flatten()

        try:
            result = least_squares(residuals, params, method='lm', max_nfev=10000)
            if time.time() > deadline - 5:
                return None, None, None
            
            if result.success or result.cost < 1e-8:
                U_opt = result.x[:U_dim * R].reshape(U_dim, R)
                V_opt = result.x[U_dim * R : U_dim * R + V_dim * R].reshape(V_dim, R)
                W_opt = result.x[U_dim * R + V_dim * R :].reshape(W_dim, R)
                
                # Snap to half-integers
                U_s = snap_to_half(U_opt)
                V_s = snap_to_half(V_opt)
                W_s = snap_to_half(W_opt)
                
                if validate(U_s, V_s, W_s):
                    return U_s, V_s, W_s
                else:
                    # Try integers
                    U_i = np.round(U_opt)
                    V_i = np.round(V_opt)
                    W_i = np.round(W_opt)
                    if validate(U_i, V_i, W_i):
                        return U_i, V_i, W_i
        except Exception:
            pass
        
        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None

    # Try to reduce to rank 44
    if time.time() < deadline - 30:
        U44, V44, W44 = try_reduce_rank(U46, V46, W46, 44, max_time=20)
        if U44 is not None:
            best_result = (U44, V44, W44)

    # Try to reduce to rank 45
    if best_result is None and time.time() < deadline - 30:
        U45, V45, W45 = try_reduce_rank(U46, V46, W46, 45, max_time=20)
        if U45 is not None:
            best_result = (U45, V45, W45)

    if best_result is not None:
        U_r, V_r, W_r = best_result
        return {
            "U": [[float(x) for x in row] for row in U_r],
            "V": [[float(x) for x in row] for row in V_r],
            "W": [[float(x) for x in row] for row in W_r]
        }

    # Fallback to rank 46
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
