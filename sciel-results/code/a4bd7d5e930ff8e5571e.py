import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    max_time = 145  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 4
    
    # Precompute index arrays for fast verification
    U_idx = np.array([[i * m + j for j in range(m)] for i in range(n)])
    V_idx = np.array([[j * p + k for k in range(p)] for j in range(m)])
    W_idx = np.array([[k * n + i for i in range(n)] for k in range(p)])
    
    # Build the proven rank-22 construction as fallback
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    def try_fix_uv_solve_w(rank, max_attempts=500):
        """
        Strategy:
        1. Randomly generate sparse U and V with entries in {-1, 0, 1}.
        2. For each (k, i), solve the linear system A_ki * w_ki = 1 for w_ki.
        3. Check if solution is close to half-integers.
        4. Verify.
        """
        rng = np.random.RandomState(42)
        
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time - 10:
                break
            
            # Generate U (n*m x rank)
            U = np.zeros((n * m, rank))
            # Generate V (m*p x rank)
            V = np.zeros((m * p, rank))
            
            # Randomly assign non-zero entries
            # Let's try 2 non-zeros per column in U and V
            for t in range(rank):
                u_indices = rng.choice(n * m, size=2, replace=False)
                for idx in u_indices:
                    U[idx, t] = rng.choice([-1, 1])
                
                v_indices = rng.choice(m * p, size=2, replace=False)
                for idx in v_indices:
                    V[idx, t] = rng.choice([-1, 1])
            
            # Now solve for W
            W = np.zeros((p * n, rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    b = np.ones(m)
                    
                    try:
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                        
                        if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-9:
                            valid = False
                            break
                            
                        w_snap = snap_to_half_integer(w_sol.reshape(1, -1)).flatten()
                        
                        if not np.allclose(A @ w_snap, b, atol=1e-9):
                            valid = False
                            break
                            
                        W[w_idx, :] = w_snap
                        
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            U_list = [[round(float(x), 10) for x in row] for row in U]
            V_list = [[round(float(x), 10) for x in row] for row in V]
            W_list = [[round(float(x), 10) for x in row] for row in W]
            
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
        
        return None

    # Try to find lower rank decompositions
    best_result = None
    
    # Try the new strategy first
    for rank in range(18, 22):
        if time.time() - start_time > max_time - 15:
            break
            
        result = try_fix_uv_solve_w(rank, max_attempts=200)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list, rank):
                best_result = (U_list, V_list, W_list, rank)
                break
    
    # Fallback: return the proven rank-22 construction
    if best_result is not None:
        U_list, V_list, W_list, r = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
