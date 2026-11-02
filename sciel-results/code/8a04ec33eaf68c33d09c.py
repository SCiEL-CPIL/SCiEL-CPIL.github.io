import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target rank
    target_rank = 17
    
    # Define the indices for the non-zero tensor entries
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # Value is 1
    
    # We need to satisfy:
    # sum_r U[a,r] V[b,r] W[c,r] = 1 for valid (a,b,c)
    # sum_r U[a,r] V[b,r] W[c,r] = 0 for invalid (a,b,c)
    
    # To speed up, we can just check the 18 valid equations and a subset of invalid ones?
    # No, we must satisfy all to be a valid decomposition.
    # However, if we satisfy the 18 valid ones and the rank is minimal, 
    # usually the structure enforces the zeros? No.
    # We must include the zero constraints.
    
    # Let's generate all (a,b,c) triples.
    # There are 324 triples.
    
    # Residual function for least_squares
    # x is a flattened vector of size R * (nm + mp + pn)
    
    def residuals(x):
        R = target_rank
        U = x[:R*nm].reshape(nm, R)
        V = x[R*nm:R*nm+R*mp].reshape(mp, R)
        W = x[R*nm+R*mp:].reshape(pn, R)
        
        # Compute T = sum_r U_r outer V_r outer W_r
        # This is a tensor of size nm x mp x pn
        # We can compute it efficiently using einsum or loops
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Efficiently:
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # But this creates a large array. 6x9x6 is small (324 elements).
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residuals for valid entries (should be 1)
        res_valid = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    res_valid[idx] = T[a, b, c] - 1.0
                    idx += 1
        
        # Residuals for invalid entries (should be 0)
        # We can't iterate over all 324 easily in a vectorized way without a mask.
        # Let's create a mask of valid entries.
        # Or just compute T_flat and subtract the target vector.
        
        T_flat = T.flatten()
        
        # Create target vector
        target = np.zeros(nm * mp * pn)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Flatten index: a * (mp * pn) + b * pn + c
                    flat_idx = a * (mp * pn) + b * pn + c
                    target[flat_idx] = 1.0
        
        return T_flat - target

    # Initial guess: Random small values
    np.random.seed(42)
    R = target_rank
    x0 = np.random.randn(R * (nm + mp + pn)) * 0.1
    
    # Solve
    try:
        sol = opt.least_squares(residuals, x0, method='trf', max_nfev=10000)
        if sol.success or sol.cost < 1e-10:
            x_sol = sol.x
            U = x_sol[:R*nm].reshape(nm, R)
            V = x_sol[R*nm:R*nm+R*mp].reshape(mp, R)
            W = x_sol[R*nm+R*mp:].reshape(pn, R)
            
            # Try to round to half-integers
            # Multiply by 2, round, divide by 2
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness
            T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
            T_flat_check = T_check.flatten()
            
            # Check valid entries
            valid_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        flat_idx = a * (mp * pn) + b * pn + c
                        if abs(T_flat_check[flat_idx] - 1.0) > 1e-9:
                            valid_ok = False
                            break
                if not valid_ok: break
            if not valid_ok: break
            
            # Check invalid entries
            invalid_ok = True
            # Iterate over all entries
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        flat_idx = a * (mp * pn) + b * pn + c
                        # Check if this is a valid entry
                        is_valid = False
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if a == i*m+j and b == j*p+k and c == k*n+i:
                                        is_valid = True
                        if not is_valid:
                            if abs(T_flat_check[flat_idx]) > 1e-9:
                                invalid_ok = False
                                break
                if not invalid_ok: break
            if not invalid_ok: break
            
            if valid_ok and invalid_ok:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    except Exception:
        pass
    
    # Fallback: Rank 18 Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
