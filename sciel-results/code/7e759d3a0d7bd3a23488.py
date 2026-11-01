import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Target tensor T where T[i*m+j, j*p+k, k*n+i] = 1
    # U index: i*m + j (0..3)
    # V index: j*p + k (0..5)
    # W index: k*n + i (0..5)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Fallback: Rank 11 (Strassen + Schoolbook)
    # This is the same as the current program's solution, ensuring we don't score worse.
    def get_rank_11_fallback():
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Strassen Part (Cols 0-6)
        # m1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # m3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # m4 = d(g-e)
        U[3, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1
        
        # Schoolbook Part for C2 (Cols 7-10)
        # C2 = A * B2
        # Term 7: a * B02
        U[0, 7] = 1
        V[2, 7] = 1
        W[4, 7] = 1
        
        # Term 8: b * B12
        U[1, 8] = 1
        V[5, 8] = 1
        W[4, 8] = 1
        
        # Term 9: c * B02
        U[2, 9] = 1
        V[2, 9] = 1
        W[5, 9] = 1
        
        # Term 10: d * B12
        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1
        
        return U, V, W

    U_fallback, V_fallback, W_fallback = get_rank_11_fallback()
    
    # Try to find Rank 10 decomposition
    # We search for U, V, W of shape (4, 10), (6, 10), (6, 10)
    # such that sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] == T_target
    
    def try_rank_10():
        # Flatten the variables: 4*10 + 6*10 + 6*10 = 160 variables
        # We use multiple random restarts to escape local minima
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            # Scaling is important for convergence. 
            # Typical entries in matrix mult tensors are 1, so init around 1 is good.
            x0 = np.random.randn(160) * 1.0
            
            def residuals(x):
                U = x[0:40].reshape(4, 10)
                V = x[40:100].reshape(6, 10)
                W = x[100:160].reshape(6, 10)
                
                # Compute tensor sum
                # T_calc[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
                T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
                
                # Residuals are the differences
                return (T_calc - T_target).flatten()
            
            try:
                # Use least_squares to minimize sum of squares of residuals
                # 'lm' (Levenberg-Marquardt) is often faster for small problems with exact solutions
                # 'trf' (Trust Region Reflective) is more robust
                res = least_squares(residuals, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
            if res.success and np.max(np.abs(res.fun)) < 1e-6:
                # We found a real-valued solution. Now snap to half-integers.
                U_sol = res.x[0:40].reshape(4, 10)
                V_sol = res.x[40:100].reshape(6, 10)
                W_sol = res.x[100:160].reshape(6, 10)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify exactness with snapped values
                T_check = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_check, T_target, atol=1e-9):
                    return U_snap, V_snap, W_snap
        
        return None

    U10, V10, W10 = try_rank_10()
    
    if U10 is not None:
        U, V, W = U10, V10, W10
    else:
        U, V, W = U_fallback, V_fallback, W_fallback

    # Final verification and time check
    T_final = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_final, T_target, atol=1e-9):
        # Should not happen if fallback is correct, but just in case
        U, V, W = U_fallback, V_fallback, W_fallback

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
