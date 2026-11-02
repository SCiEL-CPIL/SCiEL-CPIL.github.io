import time
import numpy as np
from scipy.optimize import least_squares

def verify_decomposition(U, V, W, n=2, m=2, p=7, tol=1e-9):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    u_val = U[i*m+j][r] if isinstance(U, list) else U[i*m+j, r]
                    v_val = V[j*p+k][r] if isinstance(V, list) else V[j*p+k, r]
                    w_val = W[k*n+i][r] if isinstance(W, list) else W[k*n+i, r]
                    s += u_val * v_val * w_val
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 135.0  # Leave 15s margin before 150s kill

    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28

    # Schoolbook construction (rank 28)
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Rank 25 construction using Strassen blocks for pairs of columns
    def make_rank25():
        rank = 25
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k_start):
            """
            Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
            """
            # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            # M2 = (A10 + A11) * B0,ks
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            # M3 = A00 * (B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            # M4 = A11 * (B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # M5 = (A00 + A01) * B1,ks+1
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Apply Strassen to pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)
        
        # Apply schoolbook to column 6
        col_start = 21
        k = 6
        
        # C0,6 = A00*B0,6 + A01*B1,6
        U[0, col_start+0] += 1
        V[7*0 + k, col_start+0] += 1
        W[2*k + 0, col_start+0] += 1
        
        U[1, col_start+1] += 1
        V[7*1 + k, col_start+1] += 1
        W[2*k + 0, col_start+1] += 1
        
        # C1,6 = A10*B0,6 + A11*B1,6
        U[2, col_start+2] += 1
        V[7*0 + k, col_start+2] += 1
        W[2*k + 1, col_start+2] += 1
        
        U[3, col_start+3] += 1
        V[7*1 + k, col_start+3] += 1
        W[2*k + 1, col_start+3] += 1

        return U, V, W

    # Build rank-25 solution
    U25, V25, W25 = make_rank25()
    best_U, best_V, best_W = U25.tolist(), V25.tolist(), W25.tolist()
    best_rank = 25

    # Attempt rank-24 via optimization: start from rank-25, try to nullify one term
    def try_rank24_via_optimization():
        if time.time() - start_time > deadline - 20:
            return None
        
        R = 24
        # Initialize from rank-25 solution, dropping the last column (schoolbook col 6, term 3)
        # U25 shape (4,25), V25 (14,25), W25 (14,25)
        # Drop column 24 (last schoolbook term for C1,6)
        U_init = np.zeros((4, R))
        V_init = np.zeros((14, R))
        W_init = np.zeros((14, R))
        
        # Copy first 24 columns from rank-25
        U_init[:, :24] = U25[:, :24]
        V_init[:, :24] = V25[:, :24]
        W_init[:, :24] = W25[:, :24]
        
        # The dropped term was: U[3,24]=1, V[13,24]=1, W[13,24]=1
        # We need to redistribute its contribution into the remaining 24 terms
        # Add small perturbation to help optimizer
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(x):
            U = x[:4*R].reshape(4, R)
            V = x[4*R:4*R+14*R].reshape(14, R)
            W = x[4*R+14*R:].reshape(14, R)
            
            res = np.zeros(28)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(7):
                        s = 0.0
                        for r in range(R):
                            s += U[i, r] * V[j*7+k, r] * W[k*2+i, r]
                        res[idx] = s - 1.0
                        idx += 1
            return res
        
        best_x = None
        best_cost = float('inf')
        
        num_attempts = 3
        for attempt in range(num_attempts):
            if time.time() - start_time > deadline - 15:
                break
            
            # Different initializations
            if attempt == 0:
                x = x0.copy()
            elif attempt == 1:
                # Perturb the last few columns
                x = x0.copy()
                rng = np.random.RandomState(attempt + 100)
                x[4*R:] += rng.randn(28*R) * 0.1
            else:
                # More aggressive perturbation
                x = x0.copy()
                rng = np.random.RandomState(attempt + 200)
                x += rng.randn(32*R) * 0.05
            
            try:
                sol = least_squares(residuals, x, method='lm', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
            except Exception:
                pass
        
        if best_x is not None and best_cost < 1e-10:
            U = best_x[:4*R].reshape(4, R)
            V = best_x[4*R:4*R+14*R].reshape(14, R)
            W = best_x[4*R+14*R:].reshape(14, R)
            
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(7):
                        s = 0.0
                        for r in range(R):
                            s += U_rounded[i, r] * V_rounded[j*7+k, r] * W_rounded[k*2+i, r]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
            # Also try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            valid_int = True
            for i in range(2):
                for j in range(2):
                    for k in range(7):
                        s = 0.0
                        for r in range(R):
                            s += U_int[i, r] * V_int[j*7+k, r] * W_int[k*2+i, r]
                        if abs(s - 1.0) > 1e-6:
                            valid_int = False
                            break
                    if not valid_int: break
                if not valid_int: break
            
            if valid_int:
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None

    result24 = try_rank24_via_optimization()
    if result24 is not None:
        best_U, best_V, best_W = result24
        best_rank = 24

    return {"U": best_U, "V": best_V, "W": best_W}
