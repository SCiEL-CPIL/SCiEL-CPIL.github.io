"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, with safety margin from 150s limit


def _check_deadline():
    """Return True if we should stop and return best result."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def _build_rank_56():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen decompositions.
    
    Dimensions: n=2, m=4, p=8
    A: (n, m) = (2, 4)
    B: (m, p) = (4, 8)
    C: (n, p) = (2, 8)
    
    The W factor uses the trace convention W[k*n+i, r] where
    C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    summed over j.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    rank = 56
    U = np.zeros((n * m, rank))  # shape (8, 56)
    V = np.zeros((m * p, rank))  # shape (32, 56)
    W = np.zeros((p * n, rank))  # shape (16, 56)
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 8 block products, we add 7 Strassen terms.
    # Block product b: d in {0,1,2,3} is B column pair index, h in {0,1} is A half
    #   h=0: A_left = A[:, 0:2], B_rows = 0:2
    #   h=1: A_right = A[:, 2:4], B_rows = 2:4
    
    for d in range(4):  # B column pair: 0->(0,1), 1->(2,3), 2->(4,5), 3->(6,7)
        for h in range(2):  # h=0: A left half, B top rows; h=1: A right half, B bottom rows
            base = (d * 2 + h) * 7
            
            if h == 0:
                # A block: A[i, 0:2] for i in {0,1}
                # B block: B[0:2, 2d:2d+2]
                # C block: C[i, 2d:2d+2] for i in {0,1}
                def a_idx(r, c):
                    return r * m + c
                def b_idx(r, c):
                    return r * p + (2 * d + c)
                def c_idx(r, c):
                    return (2 * d + c) * n + r
            else:
                # A block: A[i, 2:4] for i in {0,1}
                # B block: B[2:4, 2d:2d+2]
                # C block: C[i, 2d:2d+2] for i in {0,1}
                def a_idx(r, c):
                    return r * m + (2 + c)
                def b_idx(r, c):
                    return (2 + r) * p + (2 * d + c)
                def c_idx(r, c):
                    return (2 * d + c) * n + r
            
            # M1 = (A00 + A11) * (B00 + B11)
            col = base + 0
            U[a_idx(0, 0), col] += 1.0
            U[a_idx(1, 1), col] += 1.0
            V[b_idx(0, 0), col] += 1.0
            V[b_idx(1, 1), col] += 1.0
            W[c_idx(0, 0), col] += 1.0
            W[c_idx(1, 1), col] += 1.0
            
            # M2 = (A10 + A11) * B00
            col = base + 1
            U[a_idx(1, 0), col] += 1.0
            U[a_idx(1, 1), col] += 1.0
            V[b_idx(0, 0), col] += 1.0
            W[c_idx(1, 0), col] += 1.0
            W[c_idx(1, 1), col] -= 1.0
            
            # M3 = A00 * (B01 - B11)
            col = base + 2
            U[a_idx(0, 0), col] += 1.0
            V[b_idx(0, 1), col] += 1.0
            V[b_idx(1, 1), col] -= 1.0
            W[c_idx(0, 1), col] += 1.0
            W[c_idx(1, 1), col] += 1.0
            
            # M4 = A11 * (B10 - B00)
            col = base + 3
            U[a_idx(1, 1), col] += 1.0
            V[b_idx(1, 0), col] += 1.0
            V[b_idx(0, 0), col] -= 1.0
            W[c_idx(0, 0), col] += 1.0
            W[c_idx(1, 0), col] += 1.0
            
            # M5 = (A00 + A01) * B11
            col = base + 4
            U[a_idx(0, 0), col] += 1.0
            U[a_idx(0, 1), col] += 1.0
            V[b_idx(1, 1), col] += 1.0
            W[c_idx(0, 0), col] -= 1.0
            W[c_idx(0, 1), col] += 1.0
            
            # M6 = (A10 - A00) * (B00 + B01)
            col = base + 5
            U[a_idx(1, 0), col] += 1.0
            U[a_idx(0, 0), col] -= 1.0
            V[b_idx(0, 0), col] += 1.0
            V[b_idx(0, 1), col] += 1.0
            W[c_idx(1, 1), col] += 1.0
            
            # M7 = (A01 - A11) * (B10 + B11)
            col = base + 6
            U[a_idx(0, 1), col] += 1.0
            U[a_idx(1, 1), col] -= 1.0
            V[b_idx(1, 0), col] += 1.0
            V[b_idx(1, 1), col] += 1.0
            W[c_idx(0, 0), col] += 1.0
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct by checking on random inputs."""
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        
        # Compute via decomposition: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(U.shape[1]):
                        C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if not np.allclose(C_expected, C_computed, atol=1e-9):
            return False
    return True


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 by 4x8 matrix multiplication.
    Returns dict with U, V, W factor matrices.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    if _check_deadline():
        # Fallback: trivial rank-64 decomposition
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j, r] = 1.0
                    V_fb[j * p + k, r] = 1.0
                    W_fb[k * n + i, r] = 1.0
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Build the rank-56 Strassen-based decomposition
    U, V, W = _build_rank_56()
    
    # Verify correctness
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails (shouldn't happen), return fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_fb[i * m + j, r] = 1.0
                V_fb[j * p + k, r] = 1.0
                W_fb[k * n + i, r] = 1.0
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
