import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate_strong(U, V, W, tol=1e-10):
        """Check strong condition: sum_c U[i,j1,c] V[j2,k,c] W[k,i,c] == delta(j1,j2)"""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Reshape to 3D tensors
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # Check for all i, k, j1, j2
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        # Sum over c
                        s = np.sum(U_t[i, j1, :] * V_t[j2, k, :] * W_t[k, i, :])
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    # Try to find a rank 10 decomposition for 3x3x2
    def find_3x3x2_decomp(target_rank=10, max_iters=10000):
        # Variables: U (9 x R), V (6 x R), W (6 x R)
        # Flatten to 1D
        # Reshape back to 3D for error calc
        n_local, m_local, p_local = 3, 3, 2
        nm_l, mp_l, pn_l = n_local*m_local, m_local*p_local, p_local*n_local
        
        def error_func(x):
            U_flat, V_flat, W_flat = x[:nm_l*target_rank], x[nm_l*target_rank:nm_l*target_rank+mp_l*target_rank], x[nm_l*target_rank+mp_l*target_rank:]
            U_t = U_flat.reshape(n_local, m_local, target_rank)
            V_t = V_flat.reshape(m_local, p_local, target_rank)
            W_t = W_flat.reshape(p_local, n_local, target_rank)
            
            errors = []
            for i in range(n_local):
                for k in range(p_local):
                    for j1 in range(m_local):
                        for j2 in range(m_local):
                            s = np.sum(U_t[i, j1, :] * V_t[j2, k, :] * W_t[k, i, :])
                            expected = 1.0 if j1 == j2 else 0.0
                            errors.append(s - expected)
            return np.array(errors)
        
        # Initialize
        # Try a few random initializations
        best_x = None
        best_err = np.inf
        for init in range(10):
            x0 = np.random.randn(nm_l*target_rank + mp_l*target_rank + pn_l*target_rank) * 0.1
            # Scale might be important
            try:
                res = least_squares(error_func, x0, max_nfev=5000)
                if res.cost < best_err:
                    best_err = res.cost
                    best_x = res.x
            except:
                pass
            if time_left() < 20:
                break
        
        if best_x is not None and best_err < 1e-8:
            U_flat, V_flat, W_flat = best_x[:nm_l*target_rank], best_x[nm_l*target_rank:nm_l*target_rank+mp_l*target_rank], best_x[nm_l*target_rank+mp_l*target_rank:]
            U = U_flat.reshape(nm_l, target_rank)
            V = V_flat.reshape(mp_l, target_rank)
            W = W_flat.reshape(pn_l, target_rank)
            # Snap to half-integers?
            # Check if snapping helps or breaks
            # For now, return floats
            return U, V, W
        return None

    # Construct rank 68 decomposition (fallback)
    def construct_68():
        rank = 68
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        U_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1}, {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1}, {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1}, {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]
        
        for g in range(4):
            k0 = 2 * g
            k1 = 2 * g + 1
            for l in range(7):
                for li, coeff in U_coeffs[l].items():
                    i_global = li // 2
                    j_global = li % 2
                    U[i_global*m + j_global, col] += coeff
                for lj, coeff in V_coeffs[l].items():
                    j_global = lj // 2
                    k_global = k0 + lj % 2
                    V[j_global*p + k_global, col] += coeff
                for lc, coeff in W_coeffs[l].items():
                    i_global = lc // 2
                    k_global = k0 + lc % 2
                    W[k_global*n + i_global, col] += coeff
                col += 1
            
            for i in range(2):
                for k in [k0, k1]:
                    U[i*m + 2, col] = 1.0
                    V[2*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
            for j in range(2):
                for k in [k0, k1]:
                    U[2*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + 2, col] = 1.0
                    col += 1
            for k in [k0, k1]:
                U[2*m + 2, col] = 1.0
                V[2*p + k, col] = 1.0
                W[k*n + 2, col] = 1.0
                col += 1
        
        return U, V, W

    # Try to find better decomposition
    best_result = None
    best_rank = 72
    
    # Try rank 10 for 3x3x2
    if time_left() > 50:
        U_332, V_332, W_332 = find_3x3x2_decomp(10)
        if U_332 is not None:
            # Build full 3x3x8
            rank = 4 * 10
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            for g in range(4):
                k0 = 2 * g
                k1 = 2 * g + 1
                # Embed 3x3x2 decomp
                for c in range(10):
                    # U: i, j (0..2, 0..2)
                    for i in range(3):
                        for j in range(3):
                            U[i*m + j, g*10 + c] = U_332[i*3 + j, c]
                    # V: j, k_local (0..2, 0..1)
                    for j in range(3):
                        for kl in range(2):
                            k_global = k0 + kl
                            V[j*p + k_global, g*10 + c] = V_332[j*2 + kl, c]
                    # W: k_local, i (0..1, 0..2)
                    for kl in range(2):
                        k_global = k0 + kl
                        for i in range(3):
                            W[k_global*n + i, g*10 + c] = W_332[kl*3 + i, c]
            
            if validate_strong(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break # Found optimal? Or just good?
    
    if best_result is None:
        # Fall back to 68
        U, V, W = construct_68()
        if validate_strong(U, V, W):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = 68
        else:
            # Fallback to schoolbook
            best_result = schoolbook()
            best_rank = 72
            
    return best_result
