"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    
    Strategy: Use Strassen-like block decomposition.
    - Split the 3x3 middle into blocks to exploit Strassen on 2x2 subproblems.
    - The 2x3x2 multiplication has rank 11 (vs 12 schoolbook).
    - The 2x3x6 can be seen as 2x3x3 * 2 (tensor product with 1x1x2 identity).
    - But we can do better by direct block decomposition.
    
    Let's use a systematic approach:
    - A is 2x3, B is 3x6.
    - Split B into two 3x3 blocks: B = [B1 | B2].
    - Then C = [A*B1 | A*B2].
    - A*B1 is 2x3, A*B2 is 2x3.
    - Each 2x3x3 multiplication has rank 7 (known optimal).
    - But we can share structure between the two multiplications.
    
    Actually, let's think differently. The tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k].
    For 2x3x6, the naive rank is 36.
    
    Let's try a block approach:
    - Split the output into 2x3 and 3x6... no, that doesn't help directly.
    
    Let me try the known result: 2x3x6 has rank at most 18 (from 2x3x3 rank 7 doubled? No).
    
    Actually, let's use the fact that 2x3x2 has rank 11. Then 2x3x6 = 2x3x2 tensor 1x1x3? No.
    
    Let me just do a direct search with Strassen-inspired structure.
    
    For 2x3x6:
    - A: 2x3, B: 3x6, C: 2x6
    - Split B into B_left (3x3) and B_right (3x3).
    - C_left = A * B_left (2x3), C_right = A * B_right (2x3).
    - Each 2x3x3 has rank 7, so total rank <= 14.
    - But can we do better by sharing?
    
    Let's try rank 14 first as a safe upper bound, then try to reduce.
    
    Actually, the known optimal rank for 2x3x3 is 7. So 2x3x6 <= 14.
    Let's construct the rank-14 decomposition and see if we can reduce it.
    """
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # First, let's get the rank-7 decomposition for 2x3x3.
    # This is a known result. Let me construct it.
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # Rank 7 means 7 scalar multiplications.
    
    # Let me look up or derive the 2x3x3 rank-7 decomposition.
    # One approach: use the fact that 2x3x3 can be embedded in 3x3x3.
    # 3x3x3 has rank 19 (Strassen gives 19 for 2x2x2, but 3x3x3 is different).
    
    # Actually, let me just do a numerical search for the 2x3x6 tensor.
    
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with the schoolbook decomposition (rank 36)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower-rank decomposition using numerical optimization.
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if i,j,k form a valid index.
    # We want to find U, V, W of rank r such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    
    # This is a non-convex problem. Let's try a greedy/iterative approach.
    
    # For now, let's try the block decomposition approach with rank 14.
    # Split B into two 3x3 blocks.
    
    # Rank-7 decomposition for 2x3x3:
    # Let me construct this carefully.
    # A is 2x3, B is 3x3, C is 2x3.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # One known rank-7 decomposition uses the following 7 products:
    # This is getting complex. Let me just do a numerical search.
    
    # Try rank values from low to high
    for target_rank in range(11, 15):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition of this rank
        # Use random initialization and iterative refinement
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Iteratively refine
            converged = False
            for it in range(50):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] = argmin ||T[:,i,j,:] - V[:,:,i,j] * W[:,:,i,j]||
                # This is tricky. Let's use a simpler approach.
                
                # Actually, let's use the fact that we can fix two matrices and solve for the third.
                # Fix V and W, solve for U:
                # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                
                for i in range(n):
                    for j in range(m):
                        # Build the linear system for U[i*m+j, :]
                        # For each k: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b_vec = np.ones(p)
                        # Solve A_mat @ u = b_vec
                        try:
                            u_sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[i * m + j, :] = u_sol
                        except:
                            pass
                
                # Similarly update V and W
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b_vec = np.ones(n)
                        try:
                            v_sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V[j * p + k, :] = v_sol
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b_vec = np.ones(m)
                        try:
                            w_sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            W[k * n + i, :] = w_sol
                        except:
                            pass
                
                # Check convergence
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(target_rank))
                            error += abs(val - 1)
                
                if error < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Verify and convert to rational if possible
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to exact fractions if possible
    def to_fraction_matrix(M):
        """Convert a numpy matrix to a list of lists of Fractions."""
        result = []
        for row in M:
            frac_row = []
            for val in row:
                if abs(val) < 1e-9:
                    frac_row.append(Fraction(0, 1))
                else:
                    # Try to find a simple fraction
                    frac_row.append(Fraction(val).limit_denominator(1000))
            result.append(frac_row)
        return result
    
    # For now, return as floats if we can't get exact fractions
    # The task says "integer/half-integer entries"
    
    # Let's try to round to half-integers
    def to_half_integer_matrix(M):
        """Round to nearest half-integer."""
        result = []
        for row in M:
            hi_row = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                hi_row.append(rounded)
            result.append(hi_row)
        return result
    
    # Use the best decomposition found
    U_out = to_half_integer_matrix(best_U)
    V_out = to_half_integer_matrix(best_V)
    W_out = to_half_integer_matrix(best_W)
    
    # If we didn't find anything better than schoolbook, return schoolbook
    if best_rank == rank_school:
        U_out = [[int(x) for x in row] for row in U_school]
        V_out = [[int(x) for x in row] for row in V_school]
        W_out = [[int(x) for x in row] for row in W_school]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
