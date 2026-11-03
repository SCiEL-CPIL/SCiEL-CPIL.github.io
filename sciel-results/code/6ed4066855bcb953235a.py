import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    # Schoolbook fallback
    def schoolbook():
        U = np.zeros((nm, n * m * p))
        V = np.zeros((mp, n * m * p))
        W = np.zeros((pn, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}
    
    # Try to find a decomposition of rank r using alternating least squares
    def try_rank_als(r, max_time=40, seed=42):
        """Try to find a decomposition of rank r using ALS."""
        start = time.time()
        rng = np.random.RandomState(seed)
        
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 1.0
        W = rng.randn(pn, r) * 1.0
        
        # Precompute indices
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    indices.append((u_idx, v_idx, w_idx))
        
        for it in range(100000):
            if time.time() - start > max_time:
                break
            
            # Update U
            for u_idx, v_idx, w_idx in indices:
                # We want: sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] = 1
                # Fix V and W, solve for U[u_idx, :]
                # This is: U[u_idx, :] @ (V[v_idx, :] * W[w_idx, :]) = 1
                # Let z_c = V[v_idx, c] * W[w_idx, c]
                # Then U[u_idx, :] @ z = 1
                # Minimum norm solution: U[u_idx, c] = z_c / (z @ z)
                z = V[v_idx, :] * W[w_idx, :]
                z_sq = np.dot(z, z)
                if z_sq > 1e-12:
                    U[u_idx, :] = z / z_sq
                else:
                    U[u_idx, :] = z
            
            # Update V
            for u_idx, v_idx, w_idx in indices:
                z = U[u_idx, :] * W[w_idx, :]
                z_sq = np.dot(z, z)
                if z_sq > 1e-12:
                    V[v_idx, :] = z / z_sq
                else:
                    V[v_idx, :] = z
            
            # Update W
            for u_idx, v_idx, w_idx in indices:
                z = U[u_idx, :] * V[v_idx, :]
                z_sq = np.dot(z, z)
                if z_sq > 1e-12:
                    W[w_idx, :] = z / z_sq
                else:
                    W[w_idx, :] = z
        
        # Verify
        for u_idx, v_idx, w_idx in indices:
            s = 0
            for c in range(r):
                s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            if abs(s - 1) > 1e-6:
                return None
        return {'U': U, 'V': V, 'W': W}
    
    # Try ranks from 25 to 29
    for r in [25, 26, 27, 28, 29]:
        result = try_rank_als(r, max_time=20, seed=42)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
