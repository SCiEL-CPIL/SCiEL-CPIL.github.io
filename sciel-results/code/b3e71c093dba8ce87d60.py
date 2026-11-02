import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen's Algorithm factors
    # Format: (u_coords, u_vals, v_coords, v_vals, w_coords, w_vals)
    # u_coords: (i, j), v_coords: (j, k), w_coords: (k, i)
    factors = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1]),
        # M2 = (A10 + A11) * B00
        ([(1,0), (1,1)], [1, 1], 
         [(0,0)], [1], 
         [(0,1), (1,1)], [1, -1]),
        # M3 = A00 * (B01 - B11)
        ([(0,0)], [1], 
         [(0,1), (1,1)], [1, -1], 
         [(1,0), (1,1)], [1, 1]),
        # M4 = A11 * (B10 - B00)
        ([(1,1)], [1], 
         [(1,0), (0,0)], [1, -1], 
         [(0,0), (0,1)], [1, 1]),
        # M5 = (A00 + A01) * B11
        ([(0,0), (0,1)], [1, 1], 
         [(1,1)], [1], 
         [(0,0), (1,0)], [-1, 1]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0), (0,0)], [1, -1], 
         [(0,0), (0,1)], [1, 1], 
         [(1,1)], [1]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1), (1,1)], [1, -1], 
         [(1,0), (1,1)], [1, 1], 
         [(0,0)], [1]),
    ]
    
    for r, (u_c, u_v, v_c, v_v, w_c, w_v) in enumerate(factors):
        for (i, j), val in zip(u_c, u_v):
            U[i*m + j, r] = val
        for (j, k), val in zip(v_c, v_v):
            V[j*p + k, r] = val
        for (k, i), val in zip(w_c, w_v):
            W[k*n + i, r] = val
            
    # Optional: Quick sanity check (very fast)
    if time.time() < deadline:
        # Check one random case to ensure no bug in indexing
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[5, 6], [7, 8]], dtype=np.float64)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p), dtype=np.float64)
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            # u_vec is (n*m,), reshape to (n, m)
            u_mat = u_vec.reshape(n, m)
            # v_vec is (m*p,), reshape to (m, p)
            v_mat = v_vec.reshape(m, p)
            # w_vec is (p*n,), reshape to (p, n)
            w_mat = w_vec.reshape(p, n)
            
            # The tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r
            # Contracting with A and B: sum_r (A^T u_r) (v_r^T B) w_r ?
            # Actually, the standard form is:
            # C_{ik} = sum_j A_{ij} B_{jk}
            # Decomposition: C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) * (sum_l V_{jl,r} B_{jl}) * W_{ik,r} ??
            # Let's check the convention.
            # Task: "output factor follows the official trace convention W[k*n+i,r]"
            # Standard Strassen: C = sum_r M_r * W_r where M_r are intermediate products.
            # U_{ij,r} extracts A. V_{jk,r} extracts B.
            # Product M_r = (sum U_{ij,r} A_{ij}) * (sum V_{jk,r} B_{jk})
            # C_{ik} = sum_r M_r * W_{ik,r}
            
            scalar_u = 0.0
            for i in range(n):
                for j in range(m):
                    scalar_u += U[i*m + j, r] * A[i, j]
            scalar_v = 0.0
            for j in range(m):
                for k in range(p):
                    scalar_v += V[j*p + k, r] * B[j, k]
            
            m_r = scalar_u * scalar_v
            for k in range(p):
                for i in range(n):
                    C_decomposed[i, k] += m_r * W[k*n + i, r]
                    
        if not np.allclose(C_decomposed, C_expected):
             # Fallback to schoolbook rank 8 if Strassen fails (should never happen)
             rank = 8
             U = np.zeros((n*m, rank), dtype=np.float64)
             V = np.zeros((m*p, rank), dtype=np.float64)
             W = np.zeros((p*n, rank), dtype=np.float64)
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         idx = i*m*p + j*p + k
                         U[i*m + j, idx] = 1.0
                         V[j*p + k, idx] = 1.0
                         W[k*n + i, idx] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
