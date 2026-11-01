"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 baseline with rank 27 search."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        col = 0
        for b in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    
                    # U side: A[i, 2*a_blk + j]
                    # U flat index: i * 4 + (2*a_blk + j)
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff
                    
                    # V side: B[2*a_blk + j, 2*b + k]
                    # V flat index: (2*a_blk + j) * 4 + (2*b + k)
                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff
                    
                    # W side: C[i, 2*b + k]
                    # W flat index: (2*b + k) * 2 + i
                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate_fast(U, V, W, rank):
        """Fast validation using numpy vectorization."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Check diagonal entries: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i*m+j]
                    v_val = V_np[j*p+k]
                    w_val = W_np[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal entries should be 0
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b_idx in range(m*p):
                j2 = b_idx // p
                k1 = b_idx % p
                if i1 == j1 and j1 == j2:  # skip diagonal
                    continue
                for c_idx in range(p*n):
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    u_val = U_np[a]
                    v_val = V_np[b_idx]
                    w_val = W_np[c_idx]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s) > 1e-9:
                        return False
        return True
    
    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b_idx in range(m*p):
                for c_idx in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b_idx // p
                    k1 = b_idx % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b_idx][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Build baseline rank-28
    best_result = build_strassen_28()
    best_rank = 28
    
    target_rank = 27
    found = False
    
    U28 = np.array(best_result["U"], dtype=np.float64)
    V28 = np.array(best_result["V"], dtype=np.float64)
    W28 = np.array(best_result["W"], dtype=np.float64)
    
    # Build the target tensor T as a 3D array: T[i,j,k] = 1 for all i,j,k
    # (standard matrix multiplication tensor)
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Strategy: Use the 28 Strassen columns, try to express one as a combination
    # of the others in a way that preserves the tensor.
    
    # For each column c0 to remove, check if there exists a vector alpha such that
    # the tensor contribution of column c0 can be absorbed into other columns.
    # This requires: for all (i,j,k): U_c0[i*m+j]*V_c0[j*p+k]*W_c0[k*n+i] 
    #   = sum_{c != c0} alpha_c * U_c[i*m+j]*V_c[j*p+k]*W_c[k*n+i]
    # This is generally not possible with a single alpha vector since the LHS and RHS
    # are different rank-1 tensors.
    
    # Better approach: Try to find a rank-27 decomposition by solving the system
    # directly. Use random initialization and iterative refinement.
    
    # Method: Start from Strassen-28, drop one column, then solve for corrections
    # to the remaining 27 columns using Newton's method on the tensor equations.
    
    # The tensor equations are: f_{ijk} = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1 = 0
    # for all i,j,k (32 equations), plus the off-diagonal constraints.
    
    # Actually, let me just try a more aggressive random search with better
    # initialization and more attempts.
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(777 + attempt * 104729)
        np.random.seed(777 + attempt * 104729)
        
        # Initialize from Strassen-28 with a random column removed
        remove_col = random.randint(0, 27)
        keep_cols = [c for c in range(28) if c != remove_col]
        
        U = U28[:, keep_cols].copy()
        V = V28[:, keep_cols].copy()
        W = W28[:, keep_cols].copy()
        
        # Add perturbations
        U += np.random.randn(n*m, target_rank) * 0.3
        V += np.random.randn(m*p, target_rank) * 0.3
        W += np.random.randn(p*n, target_rank) * 0.3
        
        # Try to refine using alternating minimization
        # For each iteration, update one factor at a time
        for step in range(10):
            if time.time() > deadline:
                break
            
            # Update U: for each row a (corresponding to i*m+j),
            # we need: sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx] for all b,c_idx
            # This is a linear system in U[a,:] for each a.
            # For fixed a, the equation is: U[a,:] * (V[b,:] * W[c_idx,:]) = T[a,b,c_idx]
            # for all b, c_idx. This is overdetermined (16*8 = 128 equations for 27 unknowns).
            # Use least squares.
            
            for a in range(n*m):
                i = a // m
                j = a % m
                # Build the matrix M where M[b*n + c_idx, c] = V[b,c] * W[c_idx,c]
                # and the vector rhs[b*n + c_idx] = T[a,b,c_idx]
                M = np.zeros((m*p*p*n, target_rank))
                rhs = np.zeros(m*p*p*n)
                for b in range(m*p):
                    for c_idx in range(p*n):
                        M[b*p*n + c_idx, :] = V[b, :] * W[c_idx, :]
                        # T[a,b,c_idx] = 1 if indices are consistent, else 0
                        i1 = a // m
                        j1 = a % m
                        j2 = b // p
                        k1 = b % p
                        i2 = c_idx // n
                        k2 = c_idx % n
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[b*p*n + c_idx] = 1.0
                        else:
                            rhs[b*p*n + c_idx] = 0.0
                
                # Solve least squares: U[a,:] = M^+ rhs
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass
            
            # Similarly update V and W
            for b in range(m*p):
                j = b // p
                k = b % p
                M = np.zeros((n*m*p*n, target_rank))
                rhs = np.zeros(n*m*p*n)
                for a in range(n*m):
                    for c_idx in range(p*n):
                        M[a*p*n + c_idx, :] = U[a, :] * W[c_idx, :]
                        i1 = a // m
                        j1 = a % m
                        j2 = b // p
                        k1 = b % p
                        i2 = c_idx // n
                        k2 = c_idx % n
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[a*p*n + c_idx] = 1.0
                        else:
                            rhs[a*p*n + c_idx] = 0.0
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass
            
            for c_idx in range(p*n):
                i = c_idx // n
                k = c_idx % n
                M = np.zeros((n*m*m*p, target_rank))
                rhs = np.zeros(n*m*m*p)
                for a in range(n*m):
                    for b in range(m*p):
                        M[a*m*p + b, :] = U[a, :] * V[b, :]
                        i1 = a // m
                        j1 = a % m
                        j2 = b // p
                        k1 = b % p
                        i2 = c_idx // n
                        k2 = c_idx % n
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[a*m*p + b] = 1.0
                        else:
                            rhs[a*m*p + b] = 0.0
                try:
                    W[c_idx, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass
        
        # Check if we found a valid solution
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if validate(U_list, V_list, W_list, target_rank):
            best_rank = target_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            found = True
            break
    
    # If not found, try random half-integer search
    if not found and time.time() < deadline - 5:
        for attempt in range(3000):
            if time.time() > deadline:
                break
            
            random.seed(55555 + attempt * 31)
            
            def rand_half_int():
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                return random.choice(vals)
            
            U = [[rand_half_int() for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(target_rank)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(target_rank)] for _ in range(p*n)]
            
            if validate(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result

# EVOLVE-BLOCK-END
