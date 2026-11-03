import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 6
    
    # Schoolbook decomposition (Rank 36)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find Rank 30 (2 * Rank(2x3x3)=15)
    # We need Rank-15 decomposition of 2x3x3
    n2, m2, p2 = 2, 3, 3
    R233 = 15
    
    # Build target tensor for 2x3x3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
                
    # Fast search for Rank-15 decomposition
    # Using a simplified version of the previous search
    # If found, embed into 2x3x6
    
    found_233 = None
    # Limit search time to 10 seconds
    search_deadline = start_time + 10
    
    # Initialize with random values
    rng = np.random.RandomState(42)
    U = rng.randn(n2*m2, R233) * 0.5
    V = rng.randn(m2*p2, R233) * 0.5
    W = rng.randn(p2*n2, R233) * 0.5
    
    T_flat_u = T233.reshape(n2*m2, -1).T
    T_flat_v = T233.transpose(1, 0, 2).reshape(m2*p2, -1).T
    T_flat_w = T233.transpose(2, 0, 1).reshape(p2*n2, -1).T
    
    for it in range(500):
        if time.time() > search_deadline:
            break
            
        # Solve U
        Z_vw = np.einsum('ir,jr->rij', V, W).reshape(R233, m2*p2 * p2*n2)
        U_sol, _, _, _ = np.linalg.lstsq(Z_vw.T, T_flat_u, rcond=None)
        U = U_sol.T
        
        # Solve V
        Z_uw = np.einsum('ir,jr->rij', U, W).reshape(R233, n2*m2 * p2*n2)
        V_sol, _, _, _ = np.linalg.lstsq(Z_uw.T, T_flat_v, rcond=None)
        V = V_sol.T
        
        # Solve W
        Z_uv = np.einsum('ir,jr->rij', U, V).reshape(R233, n2*m2 * m2*p2)
        W_sol, _, _, _ = np.linalg.lstsq(Z_uv.T, T_flat_w, rcond=None)
        W = W_sol.T
        
        # Check for half-integer solution
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        if np.allclose(T_check, T233, atol=1e-9):
            found_233 = (U_r, V_r, W_r)
            break
            
    if found_233 is not None:
        U233, V233, W233 = found_233
        
        # Embed into 2x3x6
        R236 = 2 * R233
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # U: Repeat U233 twice
        U236[:, :R233] = U233
        U236[:, R233:] = U233
        
        # V: Interleave based on k
        for j in range(m2):
            for k in range(p2):
                v_idx_233 = j * p2 + k
                # Block 1 (k in 0..2)
                v_idx_b1 = j * p + k
                V236[v_idx_b1, :R233] = V233[v_idx_233, :]
                
                # Block 2 (k in 3..5)
                v_idx_b2 = j * p + k + p2
                V236[v_idx_b2, R233:] = V233[v_idx_233, :]
                
        # W: Interleave based on k
        for k in range(p2):
            for i in range(n2):
                w_idx_233 = k * n2 + i
                # Block 1 (k in 0..2)
                w_idx_b1 = k * n + i
                W236[w_idx_b1, :R233] = W233[w_idx_233, :]
                
                # Block 2 (k in 3..5)
                w_idx_b2 = (k + p2) * n + i
                W236[w_idx_b2, R233:] = W233[w_idx_233, :]
                
        # Verify
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
                    
        if np.allclose(T_final, T_target, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }
            
    # Fallback to Schoolbook
    return {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }
