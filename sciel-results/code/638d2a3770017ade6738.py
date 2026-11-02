import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def make_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def set_u(i, j, r, val): U[i * m + j][r] = val
        def set_v(j, k, r, val): V[j * p + k][r] = val
        def set_w(k, i, r, val): W[k * n + i][r] = val

        # Term 1: (A00 + A11) * (B00 + B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # Term 2: (A10 + A11) * B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # Term 3: A00 * (B01 - B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # Term 4: A11 * (B10 - B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # Term 5: (A00 + A01) * B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # Term 6: (A10 - A00) * (B00 + B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # Term 7: (A01 - A11) * (B10 + B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # Term 8: A00 * B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # Term 9: A01 * B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # Term 10: A10 * B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # Term 11: A11 * B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def verify_solution(sol, n_trials=10):
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            M = np.einsum('ijr,ij->r', U3, A)
            N = np.einsum('jkr,jk->r', V3, B)
            MN = M * N
            C_computed = np.einsum('kir,r->ik', W3, MN)
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True

    sol11 = make_rank11()
    
    # Attempt to find rank 10
    def try_rank10():
        if time_remaining() < 30:
            return None
            
        rank = 10
        best_error = float('inf')
        best_params = None
        
        n_vars = n*m*rank + m*p*rank + p*n*rank
        
        def objective(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            # Compute tensor T_computed[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # We want T_computed to match the multiplication tensor T_target
            # T_target[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k]?
            # Actually, the tensor of multiplication is defined by:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # This is equivalent to:
            # sum_r (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k]) W[k,i,r] = C[i,k]
            # This must hold for all A, B.
            # This implies that the tensor formed by U, V, W must equal the multiplication tensor.
            # The multiplication tensor T has entries T[i,j,k] = 1 if the indices (i,j,k) correspond to a valid multiplication term?
            # No. The standard tensor T for matmul (n,m,p) has T[i,j,k] = 1 if the term A[i,j] B[j,k] is present.
            # In our case, every term A[i,j] B[j,k] is present in the sum for C[i,k].
            # So T_target[i,j,k] = 1 for all i,j,k? 
            # Wait. C[i,k] = sum_j A[i,j] B[j,k].
            # The tensor T is such that <T, A \otimes B> = C.
            # T[i,j,k] = 1 if the indices i,j,k are valid for the multiplication.
            # Here, i in 0..1, j in 0..1, k in 0..2.
            # Every combination (i,j,k) is a valid term in the sum for C[i,k].
            # So T_target[i,j,k] = 1 for all i,j,k.
            
            T_target = np.ones((n, m, p))
            
            T_computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            
            error = np.sum((T_computed - T_target)**2)
            return error

        # Strategy 1: Warm start from rank 11 by dropping each term
        sol11 = make_rank11()
        U11 = np.array(sol11["U"], dtype=float).reshape(n, m, 11)
        V11 = np.array(sol11["V"], dtype=float).reshape(m, p, 11)
        W11 = np.array(sol11["W"], dtype=float).reshape(p, n, 11)
        
        for drop_idx in range(11):
            if time_remaining() < 10:
                break
                
            mask = [i for i in range(11) if i != drop_idx]
            U0 = U11[:,:,mask]
            V0 = V11[:,:,mask]
            W0 = W11[:,:,mask]
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use L-BFGS-B
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
            if best_error < 1e-10:
                break

        # Strategy 2: Random restarts
        if best_error > 1e-8 and time_remaining() > 20:
            for i in range(20):
                if time_remaining() < 10:
                    break
                params0 = np.random.randn(n_vars) * 0.5
                res = minimize(objective, params0, method='Powell', options={'maxiter': 1000, 'ftol': 1e-12})
                if res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x.copy()
                if best_error < 1e-10:
                    break

        if best_error < 1e-8:
            U_flat = best_params[:n*m*rank]
            V_flat = best_params[n*m*rank : n*m*rank + m*p*rank]
            W_flat = best_params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n, m, rank)
            V = V_flat.reshape(m, p, rank)
            W = W_flat.reshape(p, n, rank)
            
            U_list = [[0]*rank for _ in range(n*m)]
            V_list = [[0]*rank for _ in range(m*p)]
            W_list = [[0]*rank for _ in range(p*n)]
            
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        U_list[i*m+j][r] = U[i,j,r]
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        V_list[j*p+k][r] = V[j,k,r]
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        W_list[k*n+i][r] = W[k,i,r]
                        
            sol10 = {"U": U_list, "V": V_list, "W": W_list}
            
            if verify_solution(sol10, n_trials=10):
                return sol10
                
        return None

    sol10 = try_rank10()
    if sol10 is not None:
        return sol10
    else:
        return sol11

# EVOLVE-BLOCK-END
