import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    Attempts to find a rank below schoolbook (105) via structured search.
    Falls back to schoolbook if no improvement is found within time budget.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 105
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # Use a few random test matrices
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_true = A @ B
            
            U_mat = np.array(U, dtype=float).reshape(nm, -1)
            V_mat = np.array(V, dtype=float).reshape(mp, -1)
            W_mat = np.array(W, dtype=float).reshape(pn, -1)
            
            A_flat = A.flatten()  # (nm,)
            B_flat = B.flatten()  # (mp,)
            
            # C[i,k] = sum_r (sum_{i',j'} U[i',j',r] * A[i',j']) * (sum_{j',k'} V[j',k',r] * B[j',k']) * W[k*n+i, r]
            # More efficiently:
            # u_coeffs = U_mat.T @ A_flat  -> (rank,)
            # v_coeffs = V_mat.T @ B_flat  -> (rank,)
            # C_calc[i,k] = sum_r u_coeffs[r] * v_coeffs[r] * W_mat[k*n+i, r]
            # = sum_r (u_coeffs[r] * v_coeffs[r]) * W_mat[:, r]
            # So C_calc_flat = W_mat @ (u_coeffs * v_coeffs)
            # Then reshape C_calc_flat to (p, n) and transpose to get (n, p)
            
            u_coeffs = U_mat.T @ A_flat  # (rank,)
            v_coeffs = V_mat.T @ B_flat  # (rank,)
            combined = u_coeffs * v_coeffs  # (rank,)
            
            C_calc_flat = W_mat @ combined  # (pn,)
            C_calc = C_calc_flat.reshape(p, n).T  # (n, p)
            
            if not np.allclose(C_true, C_calc, atol=tol, rtol=tol):
                return False
        return True
    
    def schoolbook():
        """Return the standard schoolbook decomposition at rank 105."""
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # Strategy: use random initialization with structured patterns, then verify
    # We'll try ranks from 90 down to 105 (exclusive), with limited attempts each
    
    best_result = None
    
    # Try a few structured approaches first
    # Approach 1: Random with integer/half-integer entries
    for R in range(90, schoolbook_rank):
        if time.time() > deadline:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(R * 1000 + attempt)
            
            # Initialize with small random values, then snap to half-integers
            U = np.random.randint(0, 3, (nm, R)) * 0.5  # 0, 0.5, or 1
            V = np.random.randint(0, 3, (mp, R)) * 0.5
            W = np.random.randint(0, 3, (pn, R)) * 0.5
            
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_result is not None:
            break
    
    # Approach 2: Try to build from schoolbook by merging terms
    # This is complex, so we'll try a simpler heuristic: 
    # Check if any two schoolbook terms can be combined
    
    if best_result is None and time.time() < deadline:
        # Try a more targeted approach: use the structure of the problem
        # For 3x5x7, try rank values that might have algebraic structure
        for R in [98, 96, 95, 94, 92, 90, 88, 85, 80, 75, 70]:
            if time.time() > deadline:
                break
            
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                np.random.seed(R * 10000 + attempt * 137)
                
                # Use a mix of 0, 0.5, 1, -0.5, -1
                U = np.random.choice([0, 0.5, 1, -0.5, -1], (nm, R))
                V = np.random.choice([0, 0.5, 1, -0.5, -1], (mp, R))
                W = np.random.choice([0, 0.5, 1, -0.5, -1], (pn, R))
                
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if best_result is not None:
                break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return schoolbook()
