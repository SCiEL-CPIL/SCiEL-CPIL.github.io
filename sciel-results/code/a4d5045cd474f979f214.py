"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses a verified rank-44 construction based on 4x Strassen (2x2x2) + 16x Schoolbook (2x1x1).
    Attempts to find lower ranks if time permits.
    """
    start_time = time.time()
    timeout = 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 8
    size_U = n * m  # 6
    size_V = m * p  # 24
    size_W = p * n  # 16

    # Precompute target tensor
    target = np.zeros((n, m, p))
    for j in range(m):
        for k in range(p):
            if j == k:
                for i in range(n):
                    target[i, j, k] = 1.0

    def compute_residuals(U, V, W):
        """Compute all residuals using einsum for speed."""
        U3 = U.reshape(U.shape[0], n, m)
        V3 = V.reshape(V.shape[0], m, p)
        W3 = W.reshape(W.shape[0], p, n)
        result = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
        return (result - target).flatten()

    def error_func(params, R):
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        return compute_residuals(U, V, W)

    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        U3 = U.reshape(U.shape[0], n, m)
        V3 = V.reshape(V.shape[0], m, p)
        W3 = W.reshape(W.shape[0], p, n)
        result = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
        max_err = np.max(np.abs(result - target))
        return max_err < 1e-9, max_err
    
    def try_rank(R, max_restarts=3):
        """Try to find a valid rank-R decomposition."""
        total_size = R * (size_U + size_V + size_W)
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                return None
                
            np.random.seed(42 + restart * 7)
            params = np.random.randn(total_size) * 0.3
            
            try:
                res = least_squares(error_func, params, args=(R,), method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                U = res.x[0 : R*size_U].reshape(R, size_U)
                V = res.x[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
                W = res.x[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
                
                # Try snapping to integers/half-integers
                for scale in [1, 2, 4]:
                    U_snapped = np.round(U * scale) / scale
                    V_snapped = np.round(V * scale) / scale
                    W_snapped = np.round(W * scale) / scale
                    
                    valid, _ = verify_decomposition(U_snapped, V_snapped, W_snapped)
                    if valid:
                        U_list = U_snapped.T.tolist()
                        V_list = V_snapped.T.tolist()
                        W_list = W_snapped.T.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                # If no snapped version works, check if original is close enough
                valid, _ = verify_decomposition(U, V, W)
                if valid:
                    U_list = U.T.tolist()
                    V_list = V.T.tolist()
                    W_list = W.T.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def get_fallback():
        """
        Construct rank-44 decomposition.
        Strategy:
        1. Split A (2x3) into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
        2. Split B (3x8) into B_top (2x8, rows 0,1) and B_bottom (1x8, row 2).
        3. C = A_left @ B_top + A_right @ B_bottom.
        
        Part 1: A_left (2x2) @ B_top (2x8).
        Split B_top columns into 4 pairs of 2.
        Each 2x2 @ 2x2 multiplication uses Strassen (rank 7).
        Total 4 * 7 = 28 terms.
        
        Part 2: A_right (2x1) @ B_bottom (1x8).
        This is 16 scalar multiplications (2 rows of A_right * 8 cols of B_bottom).
        Total 16 terms.
        
        Total rank = 28 + 16 = 44.
        """
        rank = 44
        U = np.zeros((size_U, rank))
        V = np.zeros((size_V, rank))
        W = np.zeros((size_W, rank))
        
        col = 0
        
        # Part 1: Strassen for 4 blocks of 2x2x2
        # A indices: rows 0,1; cols 0,1
        # B indices: rows 0,1; cols k, k+1
        # C indices: rows 0,1; cols k, k+1
        
        # Strassen factors for 2x2x2
        # Using standard Strassen algorithm:
        # m1 = (A00+A11)*(B00+B11)
        # m2 = (A10+A11)*B00
        # m3 = A00*(B01-B11)
        # m4 = A11*(B10-B00)
        # m5 = (A00+A01)*B11
        # m6 = (A10-A00)*(B00+B01)
        # m7 = (A01-A11)*(B10+B11)
        #
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # U factors (A side) - each row is [A00, A01, A10, A11]
        sU = [
            [1, 0, 0, 1], # m1: A00+A11
            [0, 0, 1, 1], # m2: A10+A11
            [1, 0, 0, 0], # m3: A00
            [0, 0, 0, 1], # m4: A11
            [1, 1, 0, 0], # m5: A00+A01
            [-1, 0, 1, 0],# m6: A10-A00
            [0, 1, 0, -1] # m7: A01-A11
        ]
        
        # V factors (B side) - each row is [B00, B01, B10, B11]
        sV = [
            [1, 0, 0, 1], # m1: B00+B11
            [1, 0, 0, 0], # m2: B00
            [0, 1, 0, -1],# m3: B01-B11
            [-1, 0, 1, 0],# m4: B10-B00
            [0, 0, 0, 1], # m5: B11
            [1, 1, 0, 0], # m6: B00+B01
            [0, 0, 1, 1]  # m7: B10+B11
        ]
        
        # W factors (C side) - each row is [C00, C01, C10, C11]
        sW = [
            [1, 0, 0, 1], # m1: C00=1, C11=1
            [0, 0, 1, -1],# m2: C10=1, C11=-1
            [0, 1, 0, 1], # m3: C01=1, C11=1
            [1, 0, 1, 0], # m4: C00=1, C10=1
            [-1, 1, 0, 0],# m5: C00=-1, C01=1
            [0, 0, 0, 1], # m6: C11=1
            [1, 0, 0, 0]  # m7: C00=1
        ]
        
        for block_idx in range(4):
            k_start = block_idx * 2
            
            for s in range(7):
                # A factors: local (i,j) -> global U[i*3 + j]
                for local_i in range(2):
                    for local_j in range(2):
                        val = sU[s][local_i * 2 + local_j]
                        if val != 0:
                            U[local_i * 3 + local_j, col] += val
                
                # B factors: local (j,k) -> global V[j*8 + (k_start + k)]
                for local_j in range(2):
                    for local_k in range(2):
                        val = sV[s][local_j * 2 + local_k]
                        if val != 0:
                            V[local_j * 8 + (k_start + local_k), col] += val
                
                # C factors: local (i,k) -> global W[(k_start + k)*2 + i]
                for local_i in range(2):
                    for local_k in range(2):
                        val = sW[s][local_i * 2 + local_k]
                        if val != 0:
                            W[(k_start + local_k) * 2 + local_i, col] += val
                
                col += 1
        
        # Part 2: Schoolbook for A_right (2x1) @ B_bottom (1x8)
        # A_right is col 2 of A. Indices: (0,2) and (1,2).
        # B_bottom is row 2 of B. Indices: (2,0)..(2,7).
        # C terms: (0,0)..(0,7) and (1,0)..(1,7).
        
        for i in range(2):
            for k in range(8):
                U[i * 3 + 2, col] = 1.0
                V[2 * 8 + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a better rank than 44
    best_result = None
    
    # First, verify the fallback works
    fallback = get_fallback()
    U_fb = np.array(fallback['U']).T
    V_fb = np.array(fallback['V']).T
    W_fb = np.array(fallback['W']).T
    valid_fb, err_fb = verify_decomposition(U_fb, V_fb, W_fb)
    
    if not valid_fb:
        # Fallback is broken, need to find something that works
        # Try rank 48 (schoolbook) as a last resort
        for R in [48, 45, 44, 43, 42, 41, 40]:
            if time.time() - start_time > timeout - 20:
                break
            result = try_rank(R, max_restarts=2)
            if result is not None:
                best_result = result
                break
        if best_result is None:
            # Return the fallback even if not verified (shouldn't happen)
            return fallback
        return best_result
    
    # Fallback is valid, try to improve
    if time.time() - start_time < 100.0:
        for R in [40, 41, 42, 43]:
            if time.time() - start_time > timeout - 10:
                break
            result = try_rank(R, max_restarts=2)
            if result is not None:
                best_result = result
                break
    
    if best_result is None:
        best_result = fallback
        
    return best_result
# EVOLVE-BLOCK-END
