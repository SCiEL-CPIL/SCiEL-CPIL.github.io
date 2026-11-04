import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a decomposition for M_{2,3,3}
    # Then use it for M_{2,3,9}
    
    def solve_mp(n_sub, m_sub, p_sub, R, time_limit=10.0):
        """
        Solve for M_{n_sub, m_sub, p_sub} of rank R.
        Returns U, V, W or None.
        """
        nm = n_sub * m_sub
        mp = m_sub * p_sub
        np_ = p_sub * n_sub
        
        # Residuals
        # Indices: i, k, j, j'
        # i in 0..n_sub-1
        # k in 0..p_sub-1
        # j in 0..m_sub-1
        # j' in 0..m_sub-1
        # Equation: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta(j, j')
        
        def residuals(params):
            U = params[:nm*R].reshape(nm, R)
            V = params[nm*R:nm*R+mp*R].reshape(mp, R)
            W = params[nm*R+mp*R:].reshape(np_, R)
            
            res = []
            for i in range(n_sub):
                for k in range(p_sub):
                    for j in range(m_sub):
                        for jp in range(m_sub):
                            val = 0.0
                            for r in range(R):
                                val += U[i*m_sub + j, r] * V[jp*p_sub + k, r] * W[k*n_sub + i, r]
                            target = 1.0 if j == jp else 0.0
                            res.append(val - target)
            return res
        
        # Initial guess
        best_res = None
        best_params = None
        best_err = float('inf')
        
        num_restarts = 5
        for _ in range(num_restarts):
            if time.time() > time_limit:
                break
            # Random initialization
            params = np.random.randn(nm*R + mp*R + np_*R) * 0.1
            try:
                res = least_squares(residuals, params, method='lm', max_nfev=1000)
                err = res.cost
                if err < best_err:
                    best_err = err
                    best_params = res.x
                    if err < 1e-8:
                        break
            except Exception:
                pass
        
        if best_params is None:
            return None
            
        U = best_params[:nm*R].reshape(nm, R)
        V = best_params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = best_params[nm*R+mp*R:].reshape(np_, R)
        
        # Try to snap to 0.5 increments
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Check error of snapped solution
        def check_error(U_s, V_s, W_s):
            err = 0.0
            for i in range(n_sub):
                for k in range(p_sub):
                    for j in range(m_sub):
                        for jp in range(m_sub):
                            val = 0.0
                            for r in range(R):
                                val += U_s[i*m_sub + j, r] * V_s[jp*p_sub + k, r] * W_s[k*n_sub + i, r]
                            target = 1.0 if j == jp else 0.0
                            err += (val - target)**2
            return np.sqrt(err)
        
        err_snapped = check_error(U_snapped, V_snapped, W_snapped)
        
        if err_snapped < 1e-4:
            return U_snapped, V_snapped, W_snapped
        else:
            # Check if unsnapped is good enough? No, must be integer/half-integer.
            # Maybe try rounding to integers only?
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            err_int = check_error(U_int, V_int, W_int)
            if err_int < 1e-4:
                return U_int, V_int, W_int
            else:
                return None

    # Try ranks for M_{2,3,3}
    # Lower bound is 9. Try 9, 10, 11, 12.
    U_33 = None
    V_33 = None
    W_33 = None
    
    for R in [9, 10, 11, 12, 13]:
        if time.time() > deadline:
            break
        print(f"Trying rank {R} for M_233")
        sol = solve_mp(2, 3, 3, R, time_limit=15.0)
        if sol is not None:
            U_33, V_33, W_33 = sol
            print(f"Found rank {R} for M_233")
            break
    else:
        print("Failed to find low rank for M_233")

    if U_33 is not None:
        # Construct M_{2,3,9} using 3 blocks
        # n=2, m=3, p=9
        # B is 3x9. Split into B1, B2, B3 each 3x3.
        # C = A B1 + A B2 + A B3
        # Rank is 3 * R_33
        R_total = 3 * len(U_33[1]) # Wait, U_33 shape is (nm, R). So R is U_33.shape[1]
        
        U = np.zeros((2*3, R_total))
        V = np.zeros((3*9, R_total))
        W = np.zeros((9*2, R_total))
        
        for block in range(3):
            # U is same for all blocks?
            # A is 2x3. It is used in all 3 blocks.
            # So U columns are the same for each block?
            # Yes, U depends on A. A is the same.
            # So U[:, block*R_33:(block+1)*R_33] = U_33
            
            # V depends on B. B is split.
            # V_33 is for B_block (3x3).
            # V for full B (3x9).
            # V[j*p+k, r].
            # For block 0: k in 0..2. V_33[j*3+k, r].
            # For block 1: k in 3..5. V_33[j*3+(k-3), r].
            # For block 2: k in 6..8. V_33[j*3+(k-6), r].
            
            # W depends on C. C is 2x9.
            # W_33 is for C_block (2x3).
            # W[k*n+i, r].
            # For block 0: k in 0..2.
            # For block 1: k in 3..5.
            # For block 2: k in 6..8.
            
            U[:, block*R_total//3:(block+1)*R_total//3] = U_33
            
            for j in range(3):
                for k_local in range(3):
                    k_global = block*3 + k_local
                    V[j*9 + k_global, block*R_total//3:(block+1)*R_total//3] = V_33[j*3 + k_local]
            
            for k_local in range(3):
                k_global = block*3 + k_local
                for i in range(2):
                    W[k_global*2 + i, block*R_total//3:(block+1)*R_total//3] = W_33[k_local*2 + i]
            
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = 2 * 3 * 9
        U = [[0] * rank for _ in range(2 * 3)]
        V = [[0] * rank for _ in range(3 * 9)]
        W = [[0] * rank for _ in range(9 * 2)]
        column = 0
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    U[i * 3 + j][column] = 1
                    V[j * 9 + k][column] = 1
                    W[k * 2 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
