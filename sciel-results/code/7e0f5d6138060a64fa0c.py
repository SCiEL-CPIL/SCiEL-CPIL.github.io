<<<<<<< SEARCH
import time
import sys
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition: 5 pairs of columns (Strassen rank 7) + 1 single column (Schoolbook rank 4)
    rank = 5 * 7 + 1 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # 5 pairs of columns using Strassen's algorithm
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices for B (row 0: k0, k1; row 1: 11+k0, 11+k1)
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices for C (trace convention W[k*n+i, r])
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # bg
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # ce
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # dg
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
=======
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def _verify_decomposition(U, V, W):
    """Verify the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T = np.zeros((n, m, p))
    
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        # Expand u_col to n x m
        u_mat = u_col.reshape(n, m)
        # Expand v_col to m x p
        v_mat = v_col.reshape(m, p)
        # Expand w_col to p x n
        w_mat = w_col.reshape(p, n)
        
        # For each (i, l, j): T[i,l,j] += u_mat[i,l] * v_mat[l,j] * w_mat[j,i]
        for i in range(n):
            for l in range(m):
                for j in range(p):
                    T[i, l, j] += u_mat[i, l] * v_mat[l, j] * w_mat[j, i]
    
    # Build the target tensor
    MM = np.ones((n, m, p))
    
    diff = np.abs(T - MM)
    max_diff = np.max(diff)
    return max_diff < 1e-10

def _build_rank39():
    """Build the rank-39 baseline decomposition."""
    n, m, p = DIMENSIONS
    rank = 5 * 7 + 1 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # 5 pairs of columns using Strassen's algorithm
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices for B (row 0: k0, k1; row 1: 11+k0, 11+k1)
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices for C (trace convention W[k*n+i, r])
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # bg
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # ce
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # dg
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(U, V, W, target_rank, time_limit=60):
    """
    Try to reduce the rank using iterative refinement.
    Returns None if no improvement found, else the improved decomposition.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    # We'll try to find a lower-rank decomposition by:
    # 1. Starting with the current decomposition
    # 2. Randomly perturbing and projecting to lower rank
    # 3. Using ALS to refine
    
    best_result = None
    
    # Try a few random initializations
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Create a random lower-rank initialization
        U_low = np.random.randn(n*m, target_rank)
        V_low = np.random.randn(m*p, target_rank)
        W_low = np.random.randn(p*n, target_rank)
        
        # Scale to match the magnitude of the original
        U_low *= np.linalg.norm(U_arr, axis=1).mean()
        V_low *= np.linalg.norm(V_arr, axis=1).mean()
        W_low *= np.linalg.norm(W_arr, axis=1).mean()
        
        # Run ALS refinement
        for iteration in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for i in range(n*m):
                # U[i, :] = sum_j V[j,:] * W[k,:] for all j,k where T[i,j,k] != 0
                # This is a linear system: U[i,:] * (V^T ⊗ W^T) = target
                # Simplified: for each r, U[i,r] should satisfy the constraints
                
                # Compute the gradient for U[i, :]
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed i, we want sum_r U[i,r] * (V[:,r] ⊗ W[:,r]) = target_i
                
                # This is a least squares problem
                # Build the design matrix
                rows = []
                targets = []
                for j in range(m*p):
                    for k in range(p*n):
                        # Only if this contributes to a valid (i,j,k) triple
                        # i is fixed, j and k vary
                        # The constraint is: sum_r U[i,r] * V[j,r] * W[k,r] = 1 if valid, 0 otherwise
                        # But we need to check if (i,j,k) is a valid index combination
                        # i in [0, n*m), j in [0, m*p), k in [0, p*n)
                        # Valid if: i% m == j // p and k % n == i // m
                        i_row = i // m
                        i_col = i % m
                        j_row = j // p
                        j_col = j % p
                        k_col = k % n
                        k_row = k // n
                        
                        if i_col == j_row and k_col == i_row:
                            # This is a valid triple, target is 1
                            coeff = [V[j, r] * W[k, r] for r in range(target_rank)]
                            rows.append(coeff)
                            targets.append(1.0)
                        else:
                            # Invalid triple, target is 0
                            # But we only care about valid triples for the tensor
                            pass
                
                if not rows:
                    continue
                
                rows_arr = np.array(rows)
                targets_arr = np.array(targets)
                
                # Solve least squares
                try:
                    U_new, _, _, _ = np.linalg.lstsq(rows_arr, targets_arr, rcond=None)
                    U_low[i, :] = U_new
                except:
                    pass
            
            # Update V similarly
            for j in range(m*p):
                rows = []
                targets = []
                for i in range(n*m):
                    for k in range(p*n):
                        i_row = i // m
                        i_col = i % m
                        j_row = j // p
                        j_col = j % p
                        k_col = k % n
                        k_row = k // n
                        
                        if i_col == j_row and k_col == i_row:
                            coeff = [U[i, r] * W[k, r] for r in range(target_rank)]
                            rows.append(coeff)
                            targets.append(1.0)
                
                if not rows:
                    continue
                
                rows_arr = np.array(rows)
                targets_arr = np.array(targets)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(rows_arr, targets_arr, rcond=None)
                    V_low[j, :] = V_new
                except:
                    pass
            
            # Update W similarly
            for k in range(p*n):
                rows = []
                targets = []
                for i in range(n*m):
                    for j in range(m*p):
                        i_row = i // m
                        i_col = i % m
                        j_row = j // p
                        j_col = j % p
                        k_col = k % n
                        k_row = k // n
                        
                        if i_col == j_row and k_col == i_row:
                            coeff = [U[i, r] * V[j, r] for r in range(target_rank)]
                            rows.append(coeff)
                            targets.append(1.0)
                
                if not rows:
                    continue
                
                rows_arr = np.array(rows)
                targets_arr = np.array(targets)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(rows_arr, targets_arr, rcond=None)
                    W_low[k, :] = W_new
                except:
                    pass
        
        # Check if this decomposition is valid
        if _verify_decomposition(U_low.tolist(), V_low.tolist(), W_low.tolist()):
            best_result = {"U": U_low.tolist(), "V": V_low.tolist(), "W": W_low.tolist()}
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the baseline rank-39 decomposition
    baseline = _build_rank39()
    
    # Verify the baseline
    if not _verify_decomposition(baseline['U'], baseline['V'], baseline['W']):
        # If baseline is invalid, return it anyway (shouldn't happen)
        return baseline
    
    # Try to find a lower-rank decomposition
    # The theoretical lower bound for 2x2 x 2x11 is at least 2*2*11 / 2 = 22
    # But Strassen gives 7 for 2x2, so 5*7 + 4 = 39 is our current best
    # Let's try ranks from 38 down to 30
    
    best_result = baseline
    start_time = time.time()
    
    for target_rank in range(38, 29, -1):
        if time.time() - start_time > 100:  # Leave 20s margin
            break
        
        result = _try_rank_reduction(baseline['U'], baseline['V'], baseline['W'], target_rank, time_limit=15)
        
        if result is not None:
            best_result = result
            break  # Found a valid lower-rank decomposition
    
    return best_result
>>>>>>> REPLACE
