import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 60)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # Target tensor for 3x2x5
    n2, m2, p2 = 3, 2, 5
    T2 = np.zeros((n2, m2, p2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i, j, k] = 1.0

    def check_decomp_2(U, V, W, tol=1e-6):
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def solve_325(rank):
        # Solve for 3x2x5
        U_init = np.random.randn(n2*m2, rank)
        V_init = np.random.randn(m2*p2, rank)
        W_init = np.random.randn(p2*n2, rank)
        
        def residuals(params):
            U = params[:n2*m2*rank].reshape(n2*m2, rank)
            V = params[n2*m2*rank : n2*m2*rank + m2*p2*rank].reshape(m2*p2, rank)
            W = params[n2*m2*rank + m2*p2*rank :].reshape(p2*n2, rank)
            res = []
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000)
            if result.cost < 1e-10:
                U = result.x[:n2*m2*rank].reshape(n2*m2, rank)
                V = result.x[n2*m2*rank : n2*m2*rank + m2*p2*rank].reshape(m2*p2, rank)
                W = result.x[n2*m2*rank + m2*p2*rank :].reshape(p2*n2, rank)
                if check_decomp_2(U, V, W):
                    return U, V, W
        except Exception:
            pass
        return None

    # Try to find decomposition for 3x2x5 with rank 18
    found = False
    for rank_try in range(15, 31):
        if time.time() > deadline - 20:
            break
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            np.random.seed(seed * 100 + rank_try)
            sol = solve_325(rank_try)
            if sol:
                U2, V2, W2 = sol
                # Construct 3x4x5 solution
                # Split j into 0,1 and 2,3
                # U shape (12, 2*rank_try)
                # V shape (20, 2*rank_try)
                # W shape (15, 2*rank_try)
                
                total_rank = 2 * rank_try
                U_full = np.zeros((n*m, total_rank))
                V_full = np.zeros((m*p, total_rank))
                W_full = np.zeros((p*n, total_rank))
                
                # Part 1: j=0,1
                # U indices: i*4 + j. For j=0,1: i*4, i*4+1
                # V indices: j*5 + k. For j=0,1: 0..4, 5..9
                # W indices: k*3 + i. Same for both parts.
                
                for i in range(n):
                    for j in range(2):
                        U_full[i*m + j, :rank_try] = U2[i*m2 + j, :]
                        
                for j in range(2):
                    for k in range(p):
                        V_full[j*p + k, :rank_try] = V2[j*p2 + k, :]
                        
                for k in range(p):
                    for i in range(n):
                        W_full[k*n + i, :rank_try] = W2[k*n2 + i, :]
                        
                # Part 2: j=2,3
                # Map j=2 -> 0, j=3 -> 1 in the 3x2x5 solution
                for i in range(n):
                    for j in range(2, 4):
                        j_local = j - 2
                        U_full[i*m + j, rank_try:total_rank] = U2[i*m2 + j_local, :]
                        
                for j in range(2, 4):
                    j_local = j - 2
                    for k in range(p):
                        V_full[j*p + k, rank_try:total_rank] = V2[j_local*p2 + k, :]
                        
                for k in range(p):
                    for i in range(n):
                        W_full[k*n + i, rank_try:total_rank] = W2[k*n2 + i, :]
                
                # Check full decomposition
                T_full = np.zeros((n, m, p))
                T_full[:] = 1.0
                
                def check_full(U, V, W, tol=1e-6):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                if abs(val - 1.0) > tol:
                                    return False
                    return True
                
                if check_full(U_full, V_full, W_full):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = total_rank
                    found = True
                    break
        if found:
            break

    # Try to snap to half-integers
    def snap_half_int(M):
        M_snapped = np.round(M * 2) / 2
        return M_snapped

    if found:
        best_U = snap_half_int(best_U)
        best_V = snap_half_int(best_V)
        best_W = snap_half_int(best_W)
        
        # Verify snapped
        T_full = np.ones((n, m, p))
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(best_U[i*m+j, :] * best_V[j*p+k, :] * best_W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
            if not valid: break
        
        if not valid:
            # If snapping broke it, try to recover by not snapping?
            # But the contract requires half-integers.
            # If we can't satisfy both, we must fall back to schoolbook which is valid.
            best_U, best_V, best_W = U_sb, V_sb, W_sb
            best_rank = rank_schoolbook

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
