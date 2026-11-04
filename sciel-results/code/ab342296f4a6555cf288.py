import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if not np.allclose(T_calc, Target, atol=tol):
            return False
        return True

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def is_half_integer(x, tol=1e-5):
        return np.allclose(x, snap_to_half_int(x), atol=tol)

    def solve_for_U(V, W):
        """
        Solve for U given V and W.
        U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / (V[b,r]^2 + W[c,r]^2 ...)?
        Actually, we minimize || T - sum_r U_r V_r W_r ||_F^2.
        For a fixed r, U[:, r] is a vector.
        The term is U[:, r] * (V[:, r] * W[:, r])? No.
        T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        For fixed r, let M_r[b,c] = V[b,r] W[c,r].
        Then T[a,:,:] = sum_r U[a,r] M_r.
        This is not a simple linear system for U[:, r] because M_r depends on r.
        However, if we fix V and W, the equation for U is:
        sum_r U[a,r] (V[:, r] W[:, r]^T) = T[a, :, :].
        This is a linear system for the vector U[:, r] for each r? No, it's coupled.
        Wait, the decomposition is a sum of rank-1 tensors.
        The tensor is $T = \sum_{r=0}^{R-1} u_r \otimes v_r \otimes w_r$.
        This is NOT a linear system for U if V and W are fixed, because the "basis" tensors $v_r \otimes w_r$ are different for each r.
        We are projecting T onto the span of $\{v_r \otimes w_r\}$? No.
        We are finding $u_r$ such that $\sum u_r \otimes v_r \otimes w_r = T$.
        This is a linear system for the unknowns $u_{a,r}$?
        Yes.
        The equations are:
        $\sum_{r=0}^{R-1} u_{a,r} v_{b,r} w_{c,r} = T_{a,b,c}$ for all $a,b,c$.
        This is a linear system in the variables $u_{a,r}$.
        Number of unknowns: $dim_U \times R$.
        Number of equations: $dim_U \times dim_V \times dim_W$.
        We can solve this using least squares.
        """
        # Construct the matrix A for the linear system.
        # Variables are U_flat (size dim_U * R).
        # Equation index (a, b, c).
        # Coefficient for U[a', r] in equation (a, b, c) is:
        # 1 if a' == a, else 0.
        # And the term is v[b, r] * w[c, r].
        # So A[(a,b,c), (a', r)] = delta_{a,a'} * v[b,r] * w[c,r].
        
        # This matrix is huge. dim_U*dim_V*dim_W x dim_U*R.
        # 75 * 25 * 15 = 28125 equations.
        # dim_U * R = 15 * R.
        # This is feasible.
        
        # But we can exploit the structure.
        # For each a, the equations involving U[a, :] are:
        # sum_r U[a, r] v[b, r] w[c, r] = T[a, b, c] for all b, c.
        # This is a system of dim_V * dim_W equations for R unknowns (U[a, 0..R-1]).
        # We can solve this for each a independently!
        
        U = np.zeros((dim_U, V.shape[1]))
        R = V.shape[1]
        
        for a in range(dim_U):
            # Build matrix M of size (dim_V * dim_V, R)
            # M[b, c, r] = V[b, r] * W[c, r]
            M = np.einsum('br,cr->bcr', V, W).reshape(dim_V * dim_W, R)
            b = Target[a, :, :].reshape(dim_V * dim_W)
            
            x, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
            U[a, :] = x
            
        return U

    def solve_for_V(U, W):
        V = np.zeros((dim_V, U.shape[1]))
        R = U.shape[1]
        
        for b in range(dim_V):
            # M[a, c, r] = U[a, r] * W[c, r]
            M = np.einsum('ar,cr->acr', U, W).reshape(dim_U * dim_W, R)
            b_vec = Target[:, b, :].reshape(dim_U * dim_W)
            
            x, residuals, rank, s = np.linalg.lstsq(M, b_vec, rcond=None)
            V[b, :] = x
            
        return V

    def solve_for_W(U, V):
        W = np.zeros((dim_W, U.shape[1]))
        R = U.shape[1]
        
        for c in range(dim_W):
            # M[a, b, r] = U[a, r] * V[b, r]
            M = np.einsum('ar,br->abr', U, V).reshape(dim_U * dim_V, R)
            b_vec = Target[:, :, c].reshape(dim_U * dim_V)
            
            x, residuals, rank, s = np.linalg.lstsq(M, b_vec, rcond=None)
            W[c, :] = x
            
        return W

    def calculate_residual(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_calc - Target)**2)

    start_time = time.time()
    time_limit = 110.0

    best_result = None
    best_rank = n * m * p
    
    # Schoolbook
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for lower ranks using ALS with snapping
    # Try ranks from 40 upwards
    for R in range(40, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Multiple restarts
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W randomly with half-integers
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_U, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_V, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_W, R))
            
            # ALS Loop
            for it in range(20):
                if time.time() - start_time > time_limit:
                    break
                    
                # Solve for U
                U_new = solve_for_U(V, W)
                U_new = snap_to_half_int(U_new)
                
                # Solve for V
                V_new = solve_for_V(U_new, W)
                V_new = snap_to_half_int(V_new)
                
                # Solve for W
                W_new = solve_for_W(U_new, V_new)
                W_new = snap_to_half_int(W_new)
                
                # Check convergence
                res = calculate_residual(U_new, V_new, W_new)
                
                if res < 1e-9:
                    if verify(U_new, V_new, W_new):
                        best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                        best_rank = R
                        found = True
                        break
                
                U, V, W = U_new, V_new, W_new
                
            if found:
                break

        if found:
            break

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
