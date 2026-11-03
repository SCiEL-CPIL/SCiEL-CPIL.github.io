import time
import random
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T is defined such that T[i, j, k] = 1 if the indices match the multiplication pattern.
    # Specifically, we want to decompose the multiplication tensor.
    # The equation is: sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all valid i, j, k.
    # Note: The indices for U, V, W are flattened.
    # U has shape (n*m, R)
    # V has shape (m*p, R)
    # W has shape (p*n, R)
    
    # Precompute the indices for the non-zero entries of the tensor.
    # The tensor has 1s at positions corresponding to the multiplication.
    # Actually, the tensor is sparse. It has exactly n*m*p non-zero entries (all 1s) in the "multiplication" sense?
    # No, the tensor $T \in \mathbb{R}^{nm \times mp \times pn}$ has $T[u, v, w] = 1$ if $u=i m+j, v=j p+k, w=k n+i$ for some $i,j,k$.
    # There are $n \times m \times p$ such triplets.
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
                
    num_entries = len(indices)
    
    def residual(params, R):
        # params is a flat vector of length R * (nm + mp + pn)
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        res = np.zeros(num_entries)
        for idx, (u, v, w) in enumerate(indices):
            # Sum over r
            res[idx] = np.dot(U[u, :], V[v, :] * W[w, :])
            
        return (res - 1.0).flatten()

    def verify_solution(U, V, W, R, tol=1e-8):
        for u, v, w in indices:
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            if abs(val - 1.0) > tol:
                return False
        return True

    def snap_to_half_integers(arr):
        # Snap values to nearest half-integer
        return np.round(arr * 2) / 2

    def check_half_integer_solution(U, V, W, R, tol=1e-6):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Verify the snapped solution
        for u, v, w in indices:
            val = np.dot(U_snap[u, :], V_snap[v, :] * W_snap[w, :])
            if abs(val - 1.0) > tol:
                return False
        return True

    best_rank = n * m * p
    best_solution = None
    
    start_time = time.time()
    timeout = 140.0 # Leave 10s margin for 150s limit
    
    # We suspect the rank is 28 based on Strassen-like insights.
    # We will search from a lower bound up to 29.
    # Lower bound is max(nm, mp, pn) = 15.
    # To save time, we can start from 28 downwards or 15 upwards.
    # Starting from 28 is safer if 28 is the answer.
    # Let's try 28, 27, 26... down to 15.
    
    # Random seed for reproducibility in search
    random.seed(42)
    np.random.seed(42)
    
    for R in range(28, 14, -1):
        if time.time() - start_time > timeout - 5:
            break
            
        # Try multiple random initializations
        # Number of attempts depends on time left
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 2:
                break
                
            # Initialize with small random values
            # Scaling: The product of 3 terms should sum to 1.
            # If R is ~28, each term is ~1/28.
            # Let's initialize with uniform [-1, 1] which is often effective for tensor decomposition.
            x0 = np.random.uniform(-1, 1, R * (nm + mp + pn))
            
            try:
                # Use least_squares
                # method='lm' is Levenberg-Marquardt, good for non-linear least squares
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=2000)
                
                if result.cost < 1e-10:
                    # Extract solution
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    U = U_flat.reshape(nm, R)
                    V = V_flat.reshape(mp, R)
                    W = W_flat.reshape(pn, R)
                    
                    # Check if it's a valid solution
                    if verify_solution(U, V, W, R):
                        # Try to snap to half-integers
                        if check_half_integer_solution(U, V, W, R):
                            U_out = snap_to_half_integers(U).tolist()
                            V_out = snap_to_half_integers(V).tolist()
                            W_out = snap_to_half_integers(W).tolist()
                            return {"U": U_out, "V": V_out, "W": W_out}
                        else:
                            # If snapping fails, return the float solution?
                            # The prompt requires integer/half-integer.
                            # If the float solution is exact but not half-integer, it's invalid per contract?
                            # "integer/half-integer entries"
                            # If we can't snap, we might have to fail this rank and try lower?
                            # Or maybe the solution IS half-integer but the solver drifted.
                            # Let's try a more aggressive snap or check if values are close to integers/half-integers.
                            pass
                    
            except Exception:
                continue
                
    # Fallback: Schoolbook decomposition
    # Rank 30
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
