"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # C[i,k] = sum_r W[k*n+i,r] * sum_j U[i*m+j,r] * V[j*p+k,r]
    #        = sum_r W[k*n+i,r] * (U_r @ V_r)[i,k]
    # where U_r = U[:,r].reshape(n,m), V_r = V[:,r].reshape(m,p)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_mat = U[:, r].reshape(n, m)
        v_mat = V[:, r].reshape(m, p)
        w_mat = W[:, r].reshape(p, n)
        M_r = u_mat @ v_mat
        C_decomp += w_mat.T * M_r
    
    if not np.allclose(C_decomp, C_expected, atol=tol):
        return False
    
    for seed in [123, 456, 789]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            M_r = u_mat @ v_mat
            C_decomp += w_mat.T * M_r
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    
    return True


def _strassen_w_coeffs():
    """
    Return the W-coefficient patterns for the 7 Strassen products.
    Each entry is a dict mapping (i0, k0) -> coefficient.
    Strassen:
      M1 = (a+d)(e+h)
      M2 = (c+d)(e)
      M3 = a(f-h)
      M4 = d(g-e)
      M5 = (a+b)(h)
      M6 = (c-a)(e+f)
      M7 = (b-d)(g+h)
      C[0,0] = M1 + M4 - M5 + M7
      C[0,1] = M3 + M5
      C[1,0] = M2 + M4
      C[1,1] = M1 - M2 + M3 + M6
    """
    # For each product r=0..6, which (i0,k0) positions get what coefficient
    coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): -1},      # M2
        {(0,1): 1, (1,1): 1},       # M3
        {(0,0): 1, (1,0): 1},       # M4
        {(0,0): -1, (0,1): 1},      # M5
        {(1,1): 1},                  # M6
        {(0,0): 1},                  # M7
    ]
    return coeffs


def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for 4x4 * 4x8.
    
    Split indices:
      i = 2*i1 + i0, i1 in {0,1}, i0 in {0,1}
      j = 2*j1 + j0, j1 in {0,1}, j0 in {0,1}  
      k = 2*k1 + k0, k1 in {0,1,2,3}, k0 in {0,1}
    
    For each (i1, k1), the sub-problem is:
      C[i0, k0] = sum_{j1,j0} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    
    This is a (2, 4, 2) matmul: M(2x4) * N(4x2) = C(2x2)
    where M[i0, 2*j1+j0] = A[2*i1+i0, 2*j1+j0]
          N[2*j1+j0, k0] = B[2*j1+j0, 2*k1+k0]
    
    Split M = [M1 | M2] (each 2x2), N = [N1; N2] (each 2x2)
    C = M1*N1 + M2*N2
    
    Use Strassen (7 products) for each 2x2*2x2: total 14 per sub-problem.
    8 sub-problems * 14 = 112 total rank.
    """
    w_coeffs = _strassen_w_coeffs()
    
    # Strassen U/V patterns for 2x2 matmul [[a,b],[c,d]] * [[e,f],[g,h]]:
    # M1 = (a+d)(e+h): U has a,d; V has e,h
    # M2 = (c+d)(e):   U has c,d; V has e
    # M3 = a(f-h):     U has a; V has f,h (with -1 on h)
    # M4 = d(g-e):     U has d; V has g,e (with -1 on e)
    # M5 = (a+b)(h):   U has a,b; V has h
    # M6 = (c-a)(e+f): U has c,a (with -1 on a); V has e,f
    # M7 = (b-d)(g+h): U has b,d (with -1 on d); V has g,h
    
    # For a 2x2 block with A-indices (a_idx, b_idx, c_idx, d_idx) 
    # and B-indices (e_idx, f_idx, g_idx, h_idx):
    # U patterns (coefficients for a,b,c,d positions):
    strassen_u = [
        {0: 1, 2: 1},        # M1: a+d  (positions: 0=a, 1=b, 2=c, 3=d)
        {2: 1, 3: 1},        # M2: c+d
        {0: 1},              # M3: a
        {3: 1},              # M4: d
        {0: 1, 1: 1},        # M5: a+b
        {2: 1, 0: -1},       # M6: c-a
        {1: 1, 3: -1},       # M7: b-d
    ]
    # V patterns (coefficients for e,f,g,h positions):
    strassen_v = [
        {0: 1, 3: 1},        # M1: e+h  (positions: 0=e, 1=f, 2=g, 3=h)
        {0: 1},              # M2: e
        {1: 1, 3: -1},       # M3: f-h
        {2: 1, 0: -1},       # M4: g-e
        {3: 1},              # M5: h
        {0: 1, 1: 1},        # M6: e+f
        {2: 1, 3: 1},        # M7: g+h
    ]
    
    rank = 112
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i1 in range(2):
        for k1 in range(4):
            # Sub-problem: C[i0,k0] = sum_{j1,j0} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
            # M = A[2*i1:2*i1+2, 0:4]  (2x4)
            # N = B[0:4, 2*k1:2*k1+2]  (4x2)
            
            # M1 = A[2*i1:2*i1+2, 0:2] (2x2)
            #   a = A[2*i1, 0], b = A[2*i1, 1], c = A[2*i1+1, 0], d = A[2*i1+1, 1]
            # M2 = A[2*i1:2*i1+2, 2:4] (2x2)
            #   a2 = A[2*i1, 2], b2 = A[2*i1, 3], c2 = A[2*i1+1, 2], d2 = A[2*i1+1, 3]
            
            # N1 = B[0:2, 2*k1:2*k1+2] (2x2)
            #   e = B[0, 2*k1], f = B[0, 2*k1+1], g = B[1, 2*k1], h = B[1, 2*k1+1]
            # N2 = B[2:4, 2*k1:2*k1+2] (2x2)
            #   e2 = B[2, 2*k1], f2 = B[2, 2*k1+1], g2 = B[3, 2*k1], h2 = B[3, 2*k1+1]
            
            # U indices for M1 block: A[i, j] -> U[i*m + j, col]
            m1_indices = [
                (2*i1) * m + 0,      # a
                (2*i1) * m + 1,      # b
                (2*i1+1) * m + 0,    # c
                (2*i1+1) * m + 1,    # d
            ]
            # U indices for M2 block
            m2_indices = [
                (2*i1) * m + 2,      # a2
                (2*i1) * m + 3,      # b2
                (2*i1+1) * m + 2,    # c2
                (2*i1+1) * m + 3,    # d2
            ]
            
            # V indices for N1 block: B[j, k] -> V[j*p + k, col]
            n1_indices = [
                0 * p + (2*k1),      # e
                0 * p + (2*k1+1),    # f
                1 * p + (2*k1),      # g
                1 * p + (2*k1+1),    # h
            ]
            # V indices for N2 block
            n2_indices = [
                2 * p + (2*k1),      # e2
                2 * p + (2*k1+1),    # f2
                3 * p + (2*k1),      # g2
                3 * p + (2*k1+1),    # h2
            ]
            
            # Process Strassen for M1*N1 (7 terms)
            for r in range(7):
                # Set U entries
                for pos, coeff in strassen_u[r].items():
                    U[m1_indices[pos], col] += coeff
                # Set V entries
                for pos, coeff in strassen_v[r].items():
                    V[n1_indices[pos], col] += coeff
                # Set W entries
                for (i0, k0), coeff in w_coeffs[r].items():
                    i_global = 2 * i1 + i0
                    k_global = 2 * k1 + k0
                    w_idx = k_global * n + i_global
                    W[w_idx, col] += coeff
                col += 1
            
            # Process Strassen for M2*N2 (7 terms)
            for r in range(7):
                # Set U entries
                for pos, coeff in strassen_u[r].items():
                    U[m2_indices[pos], col] += coeff
                # Set V entries
                for pos, coeff in strassen_v[r].items():
                    V[n2_indices[pos], col] += coeff
                # Set W entries
                for (i0, k0), coeff in w_coeffs[r].items():
                    i_global = 2 * i1 + i0
                    k_global = 2 * k1 + k0
                    w_idx = k_global * n + i_global
                    W[w_idx, col] += coeff
                col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try Strassen-based decomposition (rank 112)
    try:
        U, V, W = _build_strassen_decomposition(n, m, p)
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 128)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
