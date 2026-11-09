import time
import numpy as np
import scipy.optimize as so

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr, tol=1e-6):
        """Snap values to nearest half-integer if close, else return None."""
        snapped = []
        for row in arr:
            new_row = []
            for x in row:
                # Check if x is close to a half-integer
                closest = round(x * 2) / 2
                if abs(x - closest) < tol:
                    new_row.append(closest)
                else:
                    return None
            snapped.append(new_row)
        return snapped

    def try_rank(r, max_iters=10):
        """Attempt to find a rank r decomposition using optimization."""
        if time.time() > deadline - 5:
            return None
            
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Flatten U, V, W into a single vector
        # U shape: (nm, r)
        # V shape: (mp, r)
        # W shape: (pn, r)
        
        def residual(x):
            U_flat = x[:nm*r]
            V_flat = x[nm*r:nm*r+mp*r]
            W_flat = x[nm*r+mp*r:]
            
            U = U_flat.reshape(nm, r)
            V = V_flat.reshape(mp, r)
            W = W_flat.reshape(pn, r)
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over r
                        # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                        val = np.dot(U[i*m+j], np.dot(V[j*p+k], W[k*n+i]))
                        res[idx] = val - 1
                        idx += 1
            return res

        # Try multiple random initializations
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random half-integers in [-1, 1] or similar
            # Using random half-integers helps with the final snap
            scale = 1.0
            init_vals = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=nm*r + mp*r + pn*r)
            
            # Use least_squares to minimize sum of squares of residuals
            try:
                result = so.least_squares(residual, init_vals, method='lm', max_nfev=500)
            except Exception:
                continue
                
            if result.success or result.cost < 1e-8:
                x_opt = result.x
                U_flat = x_opt[:nm*r]
                V_flat = x_opt[nm*r:nm*r+mp*r]
                W_flat = x_opt[nm*r+mp*r:]
                
                U = U_flat.reshape(nm, r)
                V = V_flat.reshape(mp, r)
                W = W_flat.reshape(pn, r)
                
                # Snap to half-integers
                U_snapped = snap_to_half_int(U.tolist())
                V_snapped = snap_to_half_int(V.tolist())
                W_snapped = snap_to_half_int(W.tolist())
                
                if U_snapped and V_snapped and W_snapped:
                    if verify(U_snapped, V_snapped, W_snapped):
                        return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                
                # If snapping failed, maybe the solution is valid but not half-integer?
                # The problem requires integer/half-integer.
                # If we can't snap, this solution is invalid for the strict requirement.
                # However, if the residual is extremely small, maybe we can force snap?
                # No, verify will fail if it's not exact.
                
        return None

    # Try ranks from 35 down to 30
    # Schoolbook is 36.
    for r in range(35, 29, -1):
        res = try_rank(r)
        if res:
            return res
            
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
