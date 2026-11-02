import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares
import itertools

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def compute_residual(U, V, W):
        # U, V, W are flat arrays of length R * nm, etc.
        R = U.shape[0] // nm
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # einsum: 'ijr,jkr,kin->ijk'
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        return T - 1.0

    def residual(params, R):
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        res = compute_residual(U_flat, V_flat, W_flat)
        return res.flatten()

    def verify_half_integer(U, V, W):
        """Check if U, V, W are exactly half-integers and decompose correctly."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2)):
            return False
        if not np.allclose(V * 2, np.round(V * 2)):
            return False
        if not np.allclose(W * 2, np.round(W * 2)):
            return False
        
        # Verify decomposition
        res = compute_residual(U, V, W)
        return np.allclose(res, 0.0, atol=1e-10)

    def snap_to_half_integer(arr):
        """Snap array to nearest half-integers."""
        return np.round(arr * 2) / 2

    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer for safety
    
    # Known lower bounds suggest rank >= 18 for 2x3x5
    # Try ranks from 18 to 30
    for R in range(18, 31):
        if time.time() - start_time > timeout:
            break
            
        # Multiple initialization strategies
        n_attempts = 30 if R < 22 else 15
        
        for attempt in range(n_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Strategy 1: Random initialization
            if attempt % 3 == 0:
                x0 = np.random.uniform(-1, 1, R * (nm + mp + pn))
            # Strategy 2: Sparse initialization (mostly zeros with some ones)
            elif attempt % 3 == 1:
                x0 = np.zeros(R * (nm + mp + pn))
                # Add random non-zero entries
                n_nonzero = max(10, R * (nm + mp + pn) // 4)
                indices = np.random.choice(R * (nm + mp + pn), n_nonzero, replace=False)
                x0[indices] = np.random.choice([-1, 1], n_nonzero)
            # Strategy 3: Structured initialization based on tensor factors
            else:
                x0 = np.zeros(R * (nm + mp + pn))
                # Initialize U, V, W with structured patterns
                for r in range(R):
                    # U pattern
                    i = r % n
                    j = r % m
                    x0[r * nm + i * m + j] = 1.0
                    # V pattern
                    j = r % m
                    k = r % p
                    x0[nm * R + r * mp + j * p + k] = 1.0
                    # W pattern
                    k = r % p
                    i = r % n
                    x0[nm * R + mp * R + r * pn + k * n + i] = 1.0
            
            try:
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=300)
                
                if result.cost < 1e-4:
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    # Snap to half-integers
                    U_rounded = snap_to_half_integer(U_flat)
                    V_rounded = snap_to_half_integer(V_flat)
                    W_rounded = snap_to_half_integer(W_flat)
                    
                    # Verify rounded solution
                    if verify_half_integer(U_rounded, V_rounded, W_rounded):
                        return {
                            "U": U_rounded.reshape(nm, R).tolist(),
                            "V": V_rounded.reshape(mp, R).tolist(),
                            "W": W_rounded.reshape(pn, R).tolist()
                        }
                        
                    # Try a polish pass with tighter tolerance
                    if time.time() - start_time < timeout - 10:
                        try:
                            result2 = least_squares(residual, result.x, args=(R,), method='lm', max_nfev=500)
                            U_flat2 = result2.x[:nm*R]
                            V_flat2 = result2.x[nm*R:nm*R+mp*R]
                            W_flat2 = result2.x[nm*R+mp*R:]
                            
                            U_rounded2 = snap_to_half_integer(U_flat2)
                            V_rounded2 = snap_to_half_integer(V_flat2)
                            W_rounded2 = snap_to_half_integer(W_flat2)
                            
                            if verify_half_integer(U_rounded2, V_rounded2, W_rounded2):
                                return {
                                    "U": U_rounded2.reshape(nm, R).tolist(),
                                    "V": V_rounded2.reshape(mp, R).tolist(),
                                    "W": W_rounded2.reshape(pn, R).tolist()
                                }
                        except:
                            pass
                        
            except Exception:
                continue
                
    # If no better rank found, return schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
