"""Find rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _is_half_integer(val):
    """Check if val is an integer or half-integer."""
    return abs(val * 2 - round(val * 2)) < 1e-9

def _all_half_integers(matrix):
    """Check if all entries in matrix are half-integers."""
    for row in matrix:
        for val in row:
            if not _is_half_integer(val):
                return False
    return True

def _snap_to_half_integers(matrix):
    """Snap values to nearest half-integer."""
    return np.round(matrix * 2) / 2

def construct():
    """Return {'U', 'V', 'W'} with minimal rank decomposition."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    num_eq = n * m * p
    
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    candidate_ranks = [r for r in range(15, best_rank)]
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations with ALS refinement
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integers
            # Use small range to encourage sparsity/simplicity
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(n*m, r)).astype(float)
            V = np.random.choice(vals, size=(m*p, r)).astype(float)
            W = np.random.choice(vals, size=(p*n, r)).astype(float)
            
            # ALS Iterations
            for it in range(20):
                if time.time() > deadline - 5:
                    break
                    
                # Solve for W
                # Eq: sum_c U_ij,c V_jk,c W_ki,c = 1
                # Linear in W: sum_c (U_ij,c V_jk,c) W_ki,c = 1
                # Unknowns: W (p*n, r)
                # Equations: n*m*p
                
                num_unknowns_W = p * n * r
                A_W = np.zeros((num_eq, num_unknowns_W))
                b_W = np.ones(num_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                w_flat_idx = (k * n + i) * r + c
                                A_W[eq_idx, w_flat_idx] = U[i*m+j, c] * V[j*p+k, c]
                            eq_idx += 1
                
                try:
                    W_flat, res, rank_sys, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                    W = W_flat.reshape(p*n, r)
                except np.linalg.LinAlgError:
                    continue
                
                # Snap W to half-integers
                W = _snap_to_half_integers(W)
                
                # Solve for U
                # Eq: sum_c U_ij,c (V_jk,c W_ki,c) = 1
                # Linear in U: sum_c U_ij,c (V_jk,c W_ki,c) = 1
                # Unknowns: U (n*m, r)
                
                num_unknowns_U = n * m * r
                A_U = np.zeros((num_eq, num_unknowns_U))
                b_U = np.ones(num_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                u_flat_idx = (i * m + j) * r + c
                                A_U[eq_idx, u_flat_idx] = V[j*p+k, c] * W[k*n+i, c]
                            eq_idx += 1
                
                try:
                    U_flat, res, rank_sys, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
                    U = U_flat.reshape(n*m, r)
                except np.linalg.LinAlgError:
                    continue
                
                # Snap U to half-integers
                U = _snap_to_half_integers(U)
                
                # Solve for V
                # Eq: sum_c (U_ij,c W_ki,c) V_jk,c = 1
                # Linear in V: sum_c (U_ij,c W_ki,c) V_jk,c = 1
                # Unknowns: V (m*p, r)
                
                num_unknowns_V = m * p * r
                A_V = np.zeros((num_eq, num_unknowns_V))
                b_V = np.ones(num_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                v_flat_idx = (j * p + k) * r + c
                                A_V[eq_idx, v_flat_idx] = U[i*m+j, c] * W[k*n+i, c]
                            eq_idx += 1
                
                try:
                    V_flat, res, rank_sys, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
                    V = V_flat.reshape(m*p, r)
                except np.linalg.LinAlgError:
                    continue
                
                # Snap V to half-integers
                V = _snap_to_half_integers(V)
                
                # Check validity
                if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = r
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
            
    return best_result
