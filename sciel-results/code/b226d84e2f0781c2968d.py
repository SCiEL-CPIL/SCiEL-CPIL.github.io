import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute indices for non-zero and zero constraints
    # I in [0, nm), J in [0, mp), K in [0, pn)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # List of (I, J, K, target)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                constraints.append((I, J, K, 1.0))
                
    # We also need to enforce zeros for incompatible indices
    # To save time, we might only check a subset of zeros during optimization
    # and verify all at the end.
    # But least_squares needs the full residual.
    # 324 residuals is fine.
    
    all_indices = []
    for I in range(nm):
        for J in range(mp):
            for K in range(pn):
                # Check if (I, J, K) is compatible
                # I = i*m + j, J = j'*p + k, K = k'*n + i'
                i = I // m
                j = I % m
                jp = J // p
                k = J % p
                kp = K // n
                ip = K % n
                
                if i == ip and j == jp and k == kp:
                    # This is a non-zero entry
                    # It's already in constraints with target 1.0
                    # We need to avoid duplicates or just build the full list
                    pass
                else:
                    all_indices.append((I, J, K, 0.0))
                    
    # Combine
    # Actually, it's easier to just generate all 324 and set targets
    targets = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                idx = (I * mp + J) * pn + K
                targets[idx] = 1.0
                
    # Residual function
    def residual(params):
        R = len(params) // (nm + mp + pn)
        if R == 0:
            return np.zeros(len(targets))
            
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        # Compute tensor product sum
        # T[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
        # This is equivalent to einsum('ir, jr, kr -> IJK', U, V, W)
        T = np.einsum('ir,jr,kr->IJK', U, V, W)
        T_flat = T.flatten()
        
        return T_flat - targets

    # Try ranks from 10 to 18
    for R in range(10, 19):
        if time.time() > deadline - 10:
            break
            
        # Number of variables
        num_vars = R * (nm + mp + pn)
        
        # Random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Use schoolbook-like initialization?
            # Or random?
            # Random is better for escaping local minima.
            np.random.seed(restart)
            x0 = np.random.randn(num_vars) * 0.1
            
            # Fix scaling: U[0,0] = 1
            # This reduces the solution space.
            # We can just let it be, and check scaling later.
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if sol.cost < 1e-12:
                # Check for half-integer property
                U = sol.x[:nm*R].reshape(nm, R)
                V = sol.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W = sol.x[nm*R+mp*R:].reshape(pn, R)
                
                # Check if U, V, W can be scaled to half-integers
                # This is tricky.
                # Simple check: are entries close to half-integers?
                # If not, try to find a scaling factor?
                # The system is homogeneous of degree 3? No, degree 1 in each matrix.
                # If (U, V, W) is a solution, (cU, V/c, W) is not.
                # (cU, cV, W/c^2) is.
                
                # Let's just check if the entries are "nice"
                # Or try to round and verify
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify rounded solution
                T_round = np.einsum('ir,jr,kr->IJK', U_round, V_round, W_round)
                if np.allclose(T_round, targets.reshape(nm, mp, pn), atol=1e-9):
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
                # If not, maybe a different scaling works?
                # Try a few random scalings
                for _ in range(10):
                    c = np.random.uniform(0.5, 2.0)
                    U_s = U * c
                    V_s = V / c
                    W_s = W
                    U_r = np.round(U_s * 2) / 2
                    V_r = np.round(V_s * 2) / 2
                    W_r = np.round(W_s * 2) / 2
                    T_r = np.einsum('ir,jr,kr->IJK', U_r, V_r, W_r)
                    if np.allclose(T_r, targets.reshape(nm, mp, pn), atol=1e-9):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
