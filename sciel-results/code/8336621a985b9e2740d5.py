"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import numpy as np
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin

    # Schoolbook decomposition of rank n*m*p
    sb_rank = n * m * p
    U = [[0] * sb_rank for _ in range(n * m)]
    V = [[0] * sb_rank for _ in range(m * p)]
    W = [[0] * sb_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    def verify(U_mat, V_mat, W_mat):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        n_, m_, p_ = DIMENSIONS
        r = len(U_mat[0])
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    s = 0
                    for c in range(r):
                        s += U_mat[i * m_ + j][c] * V_mat[j * p_ + k][c] * W_mat[k * n_ + i][c]
                    if abs(s - 1) > 1e-12:
                        return False
        return True

    def verify_exact_fraction(U_mat, V_mat, W_mat):
        """Verify with exact fraction arithmetic."""
        n_, m_, p_ = DIMENSIONS
        r = len(U_mat[0])
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U_mat[i * m_ + j][c]) * Fraction(V_mat[j * p_ + k][c]) * Fraction(W_mat[k * n_ + i][c])
                    if s != 1:
                        return False
        return True

    # Verify schoolbook solution
    assert verify(U, V, W), "Schoolbook solution failed verification"

    # Best known solution
    best_U, best_V, best_W = U, V, W
    best_rank = sb_rank

    # Strategy: Try to find a rank-71 solution by numerical optimization
    # The tensor T[i,j,k] = 1 for all i in [0,2], j in [0,2], k in [0,7]
    # We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all 72 (i,j,k) triples
    
    # Approach: Start from schoolbook, remove one column, and try to adjust
    # the remaining 71 columns to still satisfy all 72 equations.
    # This is underdetermined (4047 unknowns, 72 equations), so we look for
    # solutions where entries are close to half-integers.
    
    target_rank = 71
    U_target = np.array(U, dtype=float)[:, :target_rank]  # 9 x 71
    V_target = np.array(V, dtype=float)[:, :target_rank]  # 24 x 71
    W_target = np.array(W, dtype=float)[:, :target_rank]  # 24 x 71
    
    # Try removing each of the 72 columns and see if we can fix the rest
    # Use a simple iterative approach: for each column to remove, try to
    # redistribute its contribution across the other 71 columns.
    
    # Alternative: Use the fact that we have many more unknowns than equations.
    # Fix most entries to be 0 or 1, and solve for a few free variables.
    
    # Let's try a structured approach:
    # Keep 71 of the 72 schoolbook columns, and for the removed column's
    # contribution (which was 1 at one specific (i,j,k) and 0 elsewhere),
    # try to distribute it using small adjustments to other columns.
    
    # For column c_removed corresponding to (i_r, j_r, k_r):
    # We need to add contributions that sum to 1 at (i_r, j_r, k_r) and 0 elsewhere.
    # We can adjust existing columns by adding small amounts.
    
    # Simpler: try to find a rank-71 solution by optimization
    # Initialize with schoolbook minus one random column, then optimize
    
    rng = np.random.default_rng(42)
    found = False
    
    for col_to_remove in range(sb_rank):
        if time.time() > deadline - 20:
            break
        
        # Start with schoolbook minus this column
        U_init = np.array(U, dtype=float)[:, [c for c in range(sb_rank) if c != col_to_remove]]
        V_init = np.array(V, dtype=float)[:, [c for c in range(sb_rank) if c != col_to_remove]]
        W_init = np.array(W, dtype=float)[:, [c for c in range(sb_rank) if c != col_to_remove]]
        
        # The missing contribution: at (i_r, j_r, k_r) we need +1, elsewhere 0
        # where (i_r, j_r, k_r) is the index of col_to_remove
        i_r = col_to_remove // (m * p)
        j_r = (col_to_remove // p) % m
        k_r = col_to_remove % p
        
        # Try to fix by adding a perturbation to one existing column
        # Column c in the new indexing corresponds to original column:
        # We need: for the removed (i_r, j_r, k_r), add 1 via adjustments
        
        # Simple approach: modify one column to absorb the missing term
        # Column c' that has U[i_r, c'] != 0 and V[j_r, c'] != 0 and W[k_r, c'] != 0
        # can be scaled up slightly, but that affects other (i,j,k) terms.
        
        # Better: find a column c' where U[i_r,c']=1, V[j_r,c']=1, W[k_r,c']=1
        # (i.e., it's the same schoolbook term) - but that's the one we removed.
        
        # Try: use two columns whose product can create the needed term
        # This gets complex. Let's try numerical optimization instead.
        
        # Use scipy.optimize to minimize the residual
        from scipy.optimize import minimize
        
        def residual(params):
            """Unpack params and compute residual."""
            U_flat = params[:9*target_rank]
            V_flat = params[9*target_rank:9*target_rank+24*target_rank]
            W_flat = params[9*target_rank+24*target_rank:]
            
            U_mat = U_flat.reshape(9, target_rank)
            V_mat = V_flat.reshape(24, target_rank)
            W_mat = W_flat.reshape(24, target_rank)
            
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                        res += (s - 1.0) ** 2
            return res
        
        # This is too slow for full optimization. Let's try a smarter approach.
        
        # Key insight: We only need to fix ONE (i,j,k) entry from 0 to 1,
        # while keeping all others at 1. So we need a perturbation delta such that:
        # sum_c delta_U[i,j,c]*V[j,k,c]*W[k,i,c] + ... = 1 at (i_r,j_r,k_r)
        # and 0 elsewhere.
        
        # Linearize: add small perturbations to U, V, W
        # delta_T[i,j,k] = sum_c (dU[i,j,c]*V[j,k,c]*W[k,i,c] + 
        #                          U[i,j,c]*dV[j,k,c]*W[k,i,c] +
        #                          U[i,j,c]*V[j,k,c]*dW[k,i,c])
        
        # We need delta_T[i_r,j_r,k_r] = 1 and delta_T[i,j,k] = 0 for others.
        # This is a linear system in the perturbations.
        
        # Build the linear system: 72 equations, 9*71+24*71+24*71 = 4047 unknowns
        # Use least squares to find a solution, then check if it's small enough
        # to round to half-integers.
        
        A_rows = []
        b_vec = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [0.0] * (9*target_rank + 24*target_rank + 24*target_rank)
                    # dU terms: index = (i*m+j)*target_rank + c
                    for c in range(target_rank):
                        row[(i*m+j)*target_rank + c] = V_init[j*p+k, c] * W_init[k*n+i, c]
                    # dV terms: index = 9*target_rank + (j*p+k)*target_rank + c
                    for c in range(target_rank):
                        row[9*target_rank + (j*p+k)*target_rank + c] = U_init[i*m+j, c] * W_init[k*n+i, c]
                    # dW terms: index = (9+24)*target_rank + (k*n+i)*target_rank + c
                    for c in range(target_rank):
                        row[(9+24)*target_rank + (k*n+i)*target_rank + c] = U_init[i*m+j, c] * V_init[j*p+k, c]
                    A_rows.append(row)
                    if i == i_r and j == j_r and k == k_r:
                        b_vec.append(1.0)
                    else:
                        b_vec.append(0.0)
        
        A = np.array(A_rows)
        b = np.array(b_vec)
        
        # Solve least squares
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the perturbation is small (so rounding to half-integers works)
        U_pert = x[:9*target_rank].reshape(9, target_rank)
        V_pert = x[9*target_rank:9*target_rank+24*target_rank].reshape(24, target_rank)
        W_pert = x[9*target_rank+24*target_rank:].reshape(24, target_rank)
        
        U_new = U_init + U_pert
        V_new = V_init + V_pert
        W_new = W_init + W_pert
        
        # Check residual
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c]
                    max_res = max(max_res, abs(s - 1.0))
        
        if max_res < 1e-10:
            # Now try to round to half-integers
            U_rounded = np.round(U_new * 2) / 2
            V_rounded = np.round(V_new * 2) / 2
            W_rounded = np.round(W_new * 2) / 2
            
            # Check if rounded version still works
            max_res_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                        max_res_rounded = max(max_res_rounded, abs(s - 1.0))
            
            if max_res_rounded < 1e-12:
                # Convert to lists of floats (half-integers)
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                # Verify with exact fractions
                if verify_exact_fraction(U_list, V_list, W_list):
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = target_rank
                    found = True
                    break
        
        # If linear approach didn't work, try a different column to remove
        continue
    
    if not found and time.time() < deadline - 10:
        # Try rank 71 with a different initialization
        # Use the structure: 3x3 matmul has rank 7, try to exploit that
        # For now, just try more columns to remove with the linear approach
        
        # Alternative: try to find a solution where some entries are 0, 
        # reducing the effective rank
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
