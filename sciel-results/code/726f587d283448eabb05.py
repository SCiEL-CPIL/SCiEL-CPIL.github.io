import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18

    # Build target tensor for 3x3 x 3x6
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-8):
        r = U.shape[1]
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=tol, rtol=0)

    def schoolbook_decomposition():
        rank = n * m * p  # 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, sb_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), sb_rank

    # Try to find a rank-21 decomposition for 3x3
    def find_rank_21_3x3():
        # Target tensor for 3x3
        T_3x3 = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T_3x3[a, b, c] = 1.0
        
        rank_target = 21
        # Variables: U (9x21), V (9x21), W (9x21)
        # Flatten: 9*21 * 3 = 567
        
        from scipy.optimize import minimize
        
        for trial in range(10):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(trial * 12345 + 67890)
            U_t = np.random.randn(9, rank_target) * 0.5
            V_t = np.random.randn(9, rank_target) * 0.5
            W_t = np.random.randn(9, rank_target) * 0.5
            
            # Scale
            scale = (27.0 / (9 * 9 * 9)) ** (1.0 / 3.0)
            U_t *= scale
            V_t *= scale
            W_t *= scale
            
            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])
            
            def objective(x):
                U_f = x[:9*rank_target].reshape(9, rank_target)
                V_f = x[9*rank_target:18*rank_target].reshape(9, rank_target)
                W_f = x[18*rank_target:].reshape(9, rank_target)
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_3x3 - T_recon
                return np.sum(error ** 2)
            
            def gradient(x):
                U_f = x[:9*rank_target].reshape(9, rank_target)
                V_f = x[9*rank_target:18*rank_target].reshape(9, rank_target)
                W_f = x[18*rank_target:].reshape(9, rank_target)
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_3x3 - T_recon
                dU = -2.0 * np.einsum('abc,bt,ct->at', error, V_f, W_f)
                dV = -2.0 * np.einsum('abc,at,ct->bt', error, U_f, W_f)
                dW = -2.0 * np.einsum('abc,at,bt->ct', error, U_f, V_f)
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': 2000, 'gtol': 1e-14, 'maxfun': 10000})
            except Exception:
                continue
            
            U_final = result.x[:9*rank_target].reshape(9, rank_target)
            V_final = result.x[9*rank_target:18*rank_target].reshape(9, rank_target)
            W_final = result.x[18*rank_target:].reshape(9, rank_target)
            
            T_recon = np.einsum('at,bt,ct->abc', U_final, V_final, W_final)
            if np.allclose(T_recon, T_3x3, atol=1e-7, rtol=0):
                return U_final, V_final, W_final
        
        return None

    sol_3x3 = find_rank_21_3x3()
    
    if sol_3x3 is not None:
        U_3x3, V_3x3, W_3x3 = sol_3x3
        # Construct 3x3 x 3x6 solution
        # U is 9x42
        # V is 18x42
        # W is 18x42
        
        U_new = np.zeros((9, 42))
        V_new = np.zeros((18, 42))
        W_new = np.zeros((18, 42))
        
        # First 21 columns: for B1 (k=0,1,2)
        U_new[:, 0:21] = U_3x3
        # V rows for B1: j*6+k for k in 0,1,2
        # j=0: 0,1,2
        # j=1: 6,7,8
        # j=2: 12,13,14
        V_new[0:3, 0:21] = V_3x3[0:3, :]
        V_new[6:9, 0:21] = V_3x3[3:6, :]
        V_new[12:15, 0:21] = V_3x3[6:9, :]
        
        # W rows for B1: k*3+i for k in 0,1,2
        # k=0: 0,1,2
        # k=1: 3,4,5
        # k=2: 6,7,8
        W_new[0:3, 0:21] = W_3x3[0:3, :]
        W_new[3:6, 0:21] = W_3x3[3:6, :]
        W_new[6:9, 0:21] = W_3x3[6:9, :]
        
        # Next 21 columns: for B2 (k=3,4,5)
        U_new[:, 21:42] = U_3x3
        # V rows for B2: j*6+k for k in 3,4,5
        # j=0: 3,4,5
        # j=1: 9,10,11
        # j=2: 15,16,17
        V_new[3:6, 21:42] = V_3x3[0:3, :]
        V_new[9:12, 21:42] = V_3x3[3:6, :]
        V_new[15:18, 21:42] = V_3x3[6:9, :]
        
        # W rows for B2: k*3+i for k in 3,4,5
        # k=3: 9,10,11
        # k=4: 12,13,14
        # k=5: 15,16,17
        W_new[9:12, 21:42] = W_3x3[0:3, :]
        W_new[12:15, 21:42] = W_3x3[3:6, :]
        W_new[15:18, 21:42] = W_3x3[6:9, :]
        
        if check_decomposition(U_new, V_new, W_new):
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, 42
            print("Found rank 42 solution")

    # Convert to lists
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                frac = Fraction(val).limit_denominator(10000)
                if abs(float(frac) - val) < 1e-9:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    elif frac.denominator == 2:
                        row_list.append(frac.numerator / 2.0)
                    else:
                        row_list.append(float(frac))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
