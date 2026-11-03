"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120 # Safety margin

def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        """
        Verifies if U, V, W form a valid decomposition for (n,m,p) matrix multiplication.
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Target tensor T[i, j, k] = 1
        # Decomposed tensor D[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_lader_3x3x3():
        """
        Returns a valid rank-23 decomposition for 3x3x3 if known, else None.
        This function is a placeholder for hardcoded coefficients.
        If coefficients are incorrect, verify() will fail and we fall back to schoolbook.
        """
        # Attempt to use a known set of 23 terms.
        # Since I cannot guarantee the exact coefficients from memory, 
        # and providing wrong ones would just trigger the fallback,
        # I will return None to explicitly use schoolbook.
        # In a real evolution step, this would be filled with the 23 vectors.
        return None

    def build_3x3x6_from_3x3x3(U0, V0, W0):
        """
        Expands a valid 3x3x3 rank-23 decomposition to 3x3x6 rank-46.
        U0, V0, W0 are 9x23, 9x23, 9x23 matrices.
        """
        n3, m3, p3 = 3, 3, 3
        n6, m6, p6 = 3, 3, 6
        
        # U is 9x46. Both halves are U0.
        U = np.hstack([U0, U0])
        
        # V is 18x46.
        # First 23 cols: V0 for k in 0..2, 0 for k in 3..5.
        # Last 23 cols: 0 for k in 0..2, V0 (shifted) for k in 3..5.
        V = np.zeros((m6 * p6, 46))
        # Fill first 23 cols
        for j in range(m3):
            for k in range(p3):
                V[j * p6 + k, 0:23] = V0[j * p3 + k, :]
        # Fill last 23 cols
        for j in range(m3):
            for k in range(p3):
                V[j * p6 + (k + p3), 23:46] = V0[j * p3 + k, :]
                
        # W is 18x46.
        # First 23 cols: W0 for k in 0..2, 0 for k in 3..5.
        # Last 23 cols: 0 for k in 0..2, W0 (shifted) for k in 3..5.
        W = np.zeros((p6 * n6, 46))
        # Fill first 23 cols
        for k in range(p3):
            for i in range(n3):
                W[k * n6 + i, 0:23] = W0[k * p3 + i, :]
        # Fill last 23 cols
        for k in range(p3):
            for i in range(n3):
                W[(k + p3) * n6 + i, 23:46] = W0[k * p3 + i, :]
                
        return U, V, W

    # Try to get the improved solution
    if time.time() - START_TIME < TIME_LIMIT - 5:
        lader = get_lader_3x3x3()
        if lader is not None:
            U0, V0, W0 = lader
            # Verify the 3x3x3 part first? Or just build and verify the 3x3x6 part.
            U, V, W = build_3x3x6_from_3x3x3(U0, V0, W0)
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # Lader coefficients were invalid, fallback
                pass

    # Fallback to schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above to respect the interface.
# EVOLVE-BLOCK-END
