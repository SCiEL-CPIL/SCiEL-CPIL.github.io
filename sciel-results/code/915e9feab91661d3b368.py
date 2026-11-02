import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct Target Tensor T[a, b, c]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = 145.0 # Safety margin for 150s limit

    # Helper to verify a decomposition
    def is_valid(U, V, W):
        try:
            M = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            return np.allclose(M, T, atol=1e-9)
        except:
            return False

    # Fallback: Schoolbook Solution (Rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt 1: Lader (Rank 23) for 3x3x3 + Trivial (Rank 18) for 3x3x2
    # Total Rank 41
    if time.time() < deadline - 40:
        # Construct 3x3x3 Tensor
        T_333 = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T_333[a, b, c] = 1.0
        
        # Solve for Rank 23 decomposition of T_333
        U_333, V_333, W_333 = solve_tensor_low_rank(T_333, 9, 9, 9, 23, time_limit=35)
        
        if U_333 is not None:
            R_total = 41
            U = np.zeros((9, R_total))
            V = np.zeros((15, R_total))
            W = np.zeros((15, R_total))
            
            # Fill Lader part (cols 0-22)
            U[:, 0:23] = U_333
            for j in range(3):
                for k in range(3):
                    v_idx_333 = j * 3 + k
                    v_idx_full = j * 5 + k
                    V[v_idx_full, 0:23] = V_333[v_idx_333, :]
            W[0:9, 0:23] = W_333
            
            # Fill Trivial part for 3x3x2 (cols 23-40)
            col_offset = 23
            for i in range(3):
                for j in range(3):
                    for k in range(3, 5):
                        a = i * 3 + j
                        b = j * 5 + k
                        c = k * 3 + i
                        U[a, col_offset] = 1.0
                        V[b, col_offset] = 1.0
                        W[c, col_offset] = 1.0
                        col_offset += 1
                        
            if is_valid(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt 2: Numerical Search for Rank 44 on Full Tensor
    if time.time() < deadline - 10:
        U, V, W = solve_tensor_low_rank(T, 9, 15, 15, 44, time_limit=30)
        if U is not None and is_valid(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    return get_schoolbook()

def solve_tensor_low_rank(T, dim_u, dim_v, dim_w, rank, time_limit):
    start = time.time()
    best_err = np.inf
    best_params = None
    
    # Use multiple restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start > time_limit:
            break
            
        # Random initialization
        U = np.random.uniform(-1, 1, (dim_u, rank))
        V = np.random.uniform(-1, 1, (dim_v, rank))
        W = np.random.uniform(-1, 1, (dim_w, rank))
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_ = params[:dim_u*rank].reshape(dim_u, rank)
            V_ = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
            W_ = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
            M = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
            return np.sum((M - T)**2)
            
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                # If we found a very good solution, we can break early
                if best_err < 1e-8:
                    break
        except Exception:
            continue
            
    if best_err < 1e-6 and best_params is not None:
        U = best_params[:dim_u*rank].reshape(dim_u, rank)
        V = best_params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Final verification
        M = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if np.allclose(M, T, atol=1e-9):
            return U, V, W
            
    return None, None, None

if __name__ == "__main__":
    construct()
