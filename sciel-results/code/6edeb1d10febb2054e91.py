import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute the target tensor indices
    # T[alpha, beta, gamma] = 1 if matching, 0 otherwise
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    
    # We only need to check the non-zero ones for the "1" constraint
    # and the zero ones for the "0" constraint.
    
    # Indices of non-zero entries
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                nz_indices.append((alpha, beta, gamma))
                
    # Total number of entries
    total_entries = nm * mp * pn
    nz_count = len(nz_indices)
    
    # Verify function
    def verify(U, V, W, r):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Compute T_calc[alpha, beta, gamma] = sum_c U[alpha, c] V[beta, c] W[gamma, c]
        # We can do this efficiently using einsum or broadcasting
        # T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
        # This creates a (nm, mp, pn) tensor.
        # For nm=6, mp=30, pn=20, size is 3600. This is small.
        
        T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Check non-zero entries
        for alpha, beta, gamma in nz_indices:
            if abs(T_calc[alpha, beta, gamma] - 1.0) > 1e-9:
                return False
        
        # Check zero entries
        # We can create a mask or just iterate over all and check if not in nz
        # Iterating 3600 times is fast.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Check if (a,b,c) is in nz_indices
                    # To speed up, we can precompute a set
                    # But 3600 is small.
                    # We can check if T_calc is close to 0
                    if abs(T_calc[a, b, c]) > 1e-9:
                        # Check if it's supposed to be non-zero
                        # If it's not in nz_indices, it should be 0
                        # We can just check if it's in the set of nz indices
                        # For efficiency, we can assume if it's not 0, it's bad unless it's in nz
                        pass
        
        # More efficient check:
        # Create a target tensor T_target
        T_target = np.zeros((nm, mp, pn))
        for alpha, beta, gamma in nz_indices:
            T_target[alpha, beta, gamma] = 1.0
            
        err = np.max(np.abs(T_calc - T_target))
        return err < 1e-9

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank

    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Search for lower rank
    # Try ranks from 10 to rank_s
    # Use random initialization for U and V, then solve for W
    
    for r in range(10, rank_s + 1):
        if time.time() > deadline - 10:
            break
            
        max_restarts = 20 if r < 20 else 5
        
        for trial in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random U, V
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            
            # Solve for W
            W = np.zeros((pn, r))
            success = True
            
            # For each gamma (k, i), we have a system of m equations
            # A_j W_g = 1, where A_j = U[i, j] * V[j, k] (element-wise)
            # A_j is a vector of size r.
            # We need to find W_g such that A_j . W_g = 1 for all j.
            
            for k in range(p):
                for i in range(n):
                    gamma = k * n + i
                    # Build system
                    # m equations, r unknowns
                    A_sys = np.zeros((m, r))
                    for j in range(m):
                        alpha = i * m + j
                        beta = j * p + k
                        A_sys[j, :] = U[alpha, :] * V[beta, :]
                    
                    # Solve A_sys W_g = ones(m)
                    try:
                        # Least squares
                        W_g, residuals, rank_A, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                        W[gamma, :] = W_g
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                continue
                
            # Verify full tensor
            if verify(U, V, W, r):
                best_result = (U, V, W, r)
                break
        
        if best_result[3] == r:
            break

    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
