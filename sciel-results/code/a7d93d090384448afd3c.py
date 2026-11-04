"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def _find_rank7_decomp():
    """Try to find a rank-7 decomposition for 2x3 x 3x5 multiplication."""
    n, m, p = 2, 3, 5
    R = 7
    
    def compute_residuals(params):
        U = params[0:42].reshape(6, 7)
        V = params[42:147].reshape(15, 7)
        W = params[147:217].reshape(10, 7)
        
        residuals = np.zeros(30)
        idx = 0
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    val = 0.0
                    for r in range(7):
                        val += U[i*3+j, r] * V[j*5+k, r] * W[k*2+i, r]
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals
    
    best_params = None
    best_cost = np.inf
    
    for seed in range(10):
        np.random.seed(seed)
        params0 = np.random.randn(217) * 0.5
        
        try:
            result = least_squares(compute_residuals, params0, method='lm', max_nfev=10000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x
        except Exception:
            continue
    
    if best_cost < 1e-8 and best_params is not None:
        U = best_params[0:42].reshape(6, 7)
        V = best_params[42:147].reshape(15, 7)
        W = best_params[147:217].reshape(10, 7)
        
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        ok = True
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    val = sum(U_r[i*3+j, r] * V_r[j*5+k, r] * W_r[k*2+i, r] for r in range(7))
                    if abs(val - 1) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    block_p = 5
    num_blocks = p // block_p  # 2
    
    # Try to find rank-7 decomposition for 2x3 x 3x5
    decomp = _find_rank7_decomp()
    
    if decomp is not None:
        U_block, V_block, W_block = decomp
        total_rank = 7 * num_blocks  # 14
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for b in range(num_blocks):
            for r in range(7):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][b * 7 + r] = U_block[i * m + j][r]
                    for k in range(block_p):
                        V[j * p + (b * block_p + k)][b * 7 + r] = V_block[j * block_p + k][r]
                        W[(b * block_p + k) * n + i][b * 7 + r] = W_block[k * n + i][r]
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    total_rank = n * m * p
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
