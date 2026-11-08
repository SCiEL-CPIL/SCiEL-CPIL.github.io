import numpy as np
import time
import random
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # safety margin
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, k*p+l, r*n+s] = 1 iff j==k and i==r
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i * m + j, j * p + l, l * n + i] = 1
    
    def verify(U, V, W, tol=1e-9):
        """Check that sum_r U[:,r] outer V[:,r] outer W[:,r] == T exactly."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        recon = np.zeros(T.shape, dtype=np.float64)
        for c in range(r):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(recon, T, atol=tol, rtol=0):
            return False
        # Check integer/half-integer constraint
        for M in (U, V, W):
            for v in M.flat:
                if abs(v) > 1e-12:
                    if abs(v - round(v)) > 1e-9 and abs(2*v - round(2*v)) > 1e-9:
                        return False
        return True
    
    def make_base_solution():
        """Return the known rank-22 solution."""
        rank = 22
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx, col] += val
            for idx, val in v_indices:
                V[idx, col] += val
            for idx, val in w_indices:
                W[idx, col] += val
        
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
        
        # Outer: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W
    
    # Start with the base rank-22 solution
    U, V, W = make_base_solution()
    best_rank = 22
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies among the rank-22 terms
    # Strategy: for each pair of columns, check if their "outer product tensors" are linearly dependent
    # If so, we can merge them into one column
    
    def try_reduce_rank(U0, V0, W0, target_rank):
        """Try to find a decomposition of rank target_rank by searching."""
        n_rows_U, r0 = U0.shape
        n_rows_V, _ = V0.shape
        n_rows_W, _ = W0.shape
        
        # Method 1: Try to find if any two columns can be merged
        # For columns a and b, check if there exists alpha such that
        # U[:,a] + alpha*U[:,b], V[:,a] + alpha*V[:,b], W[:,a] + alpha*W[:,b]
        # combined with the remaining columns still gives T
        
        # Method 2: Direct search over small integer/half-integer values
        # This is expensive, so we limit the search space
        
        # Try a more structured approach: 
        # The tensor has 24 non-zero entries. Each rank-1 term covers some subset.
        # We need at least 24/3 = 8 terms (if each covers 3 entries), but realistically more.
        
        # Let's try a different approach: use the fact that the problem has structure
        # A is 2x3, B is 3x4, C is 2x4
        # C[i,l] = sum_j A[i,j]*B[j,l]
        
        # The 8 outer products for j=2 (middle column) are:
        # A[0,2]*B[2,l] for l=0,1,2,3 -> 4 terms
        # A[1,2]*B[2,l] for l=0,1,2,3 -> 4 terms
        # These 8 terms involve U indices {2,5}, V indices {8,9,10,11}, W indices {0,1,2,3,4,5,6,7}
        
        # Can we reduce these 8 terms? 
        # The sub-tensor for j=2 is: sum_l A[0,2]*B[2,l]*e_0 e_l^T + A[1,2]*B[2,l]*e_1 e_l^T
        # = A[0,2]*(B[2,0]e_0e_0^T + B[2,1]e_0e_1^T + B[2,2]e_0e_2^T + B[2,3]e_0e_3^T)
        #   + A[1,2]*(B[2,0]e_1e_0^T + B[2,1]e_1e_1^T + B[2,2]e_1e_2^T + B[2,3]e_1e_3^T)
        
        # This is a sum of 8 rank-1 tensors, each with 1 non-zero entry.
        # The rank of this sub-tensor: the U part is 2-dimensional, V part is 4-dimensional, W part is 8-dimensional.
        # The tensor is (a0*b_l, a1*b_l) for l=0,1,2,3 in the (U,V,W) space.
        # This has rank min(2*4, 8) = 8 in general, but let's check.
        
        # Actually, the 8 terms are: for each l, we have two terms:
        # (e_0, e_l, e_{2l}) and (e_1, e_l, e_{2l+1})
        # These are 8 linearly independent rank-1 tensors, so the sub-tensor has rank 8.
        
        # So we can't reduce the 8 outer products below 8.
        
        # What about the Strassen blocks? Each 2x2x2 block has rank 7 (Strassen).
        # Can we do better? The 2x2x2 matrix multiplication tensor has rank 7 (proven).
        # So each block needs at least 7 terms.
        
        # Total: 7 + 7 + 8 = 22. This is the lower bound if the blocks are independent.
        
        # But wait - the two Strassen blocks share the same U indices (0,1,3,4) but use different V indices.
        # Block 1 uses V indices 0,1,4,5 and W indices 0,1,2,3
        # Block 2 uses V indices 2,3,6,7 and W indices 4,5,6,7
        
        # The U vectors for the two blocks are the same! 
        # Block 1 U vectors: u0=(1,0,0,1,0,0), u1=(0,0,0,1,1,0), u2=(1,0,0,0,0,0), 
        #                   u3=(0,0,0,0,1,0), u4=(1,0,1,0,0,0), u5=(0,1,0,1,0,0), u6=(0,1,0,0,1,0)
        # Block 2 U vectors: same as Block 1
        
        # So the U space is shared. This might allow some reduction.
        
        # Let's try a direct search for rank 21 by modifying the base solution.
        
        # Approach: try to find a linear combination of two columns that, when added to a third,
        # allows us to eliminate one column.
        
        # Simpler: try all pairs of columns and check if removing one and adjusting the other works.
        
        for col_a in range(r0):
            if time.time() - start_time > deadline:
                return None
            for col_b in range(col_a + 1, r0):
                if time.time() - start_time > deadline:
                    return None
                
                # Try to merge columns a and b into a single column
                # New column: U_new = U[:,a] + c*U[:,b], V_new = V[:,a] + c*V[:,b], W_new = W[:,a] + c*W[:,b]
                # for some c, and remove column b (or a)
                
                # The contribution of columns a and b is:
                # U_a * V_a^T * W_a^T + U_b * V_b^T * W_b^T
                # We want this to equal a single rank-1 term: U_new * V_new^T * W_new^T
                # This is generally not possible unless the two terms are "parallel" in some sense.
                
                # Alternative: try to find if the sum of all terms except b can be adjusted to include b's contribution.
                
                # Let's try: keep all columns except b, and see if we can adjust one column to absorb b.
                # This is a linear system: find delta such that
                # sum_{c != b} (U_c + delta_c) * (V_c + delta'_c)^T * (W_c + delta''_c)^T = T
                # This is nonlinear, so let's try a simpler approach.
                
                # Try: can we express column b as a linear combination of the others in a way that
                # preserves the tensor decomposition?
                
                # For each column c != b, check if U[:,b] is in the span of {U[:,c] for c != b}
                # AND V[:,b] is in the span of {V[:,c] for c != b}  
                # AND W[:,b] is in the span of {W[:,c] for c != b}
                # AND the coefficients are consistent.
                
                # This is a sufficient but not necessary condition.
                
                # Build matrices for U, V, W excluding column b
                U_no_b = np.delete(U0, col_b, axis=1)
                V_no_b = np.delete(V0, col_b, axis=1)
                W_no_b = np.delete(W0, col_b, axis=1)
                
                # Check if U[:,b] is in the column space of U_no_b
                try:
                    U_coeffs, _, _, _ = np.linalg.lstsq(U_no_b, U0[:, col_b], rcond=None)
                    U_residual = np.linalg.norm(U0[:, col_b] - U_no_b @ U_coeffs)
                    if U_residual > 1e-9:
                        continue
                    
                    V_coeffs, _, _, _ = np.linalg.lstsq(V_no_b, V0[:, col_b], rcond=None)
                    V_residual = np.linalg.norm(V0[:, col_b] - V_no_b @ V_coeffs)
                    if V_residual > 1e-9:
                        continue
                    
                    W_coeffs, _, _, _ = np.linalg.lstsq(W_no_b, W0[:, col_b], rcond=None)
                    W_residual = np.linalg.norm(W0[:, col_b] - W_no_b @ W_coeffs)
                    if W_residual > 1e-9:
                        continue
                    
                    # Check if the coefficients are consistent (should be the same for all three)
                    if not np.allclose(U_coeffs, V_coeffs, atol=1e-9) or not np.allclose(U_coeffs, W_coeffs, atol=1e-9):
                        continue
                    
                    # If we get here, column b is redundant!
                    # Create new matrices without column b
                    U_new = U_no_b.copy()
                    V_new = V_no_b.copy()
                    W_new = W_no_b.copy()
                    
                    if verify(U_new, V_new, W_new):
                        return (U_new, V_new, W_new)
                        
                except np.linalg.LinAlgError:
                    continue
        
        return None
    
    # Try to reduce from rank 22
    reduced = try_reduce_rank(U, V, W, 21)
    if reduced is not None:
        best_rank = 21
        best_U, best_V, best_W = reduced
    
    # If we found rank 21, try to go lower
    if best_rank == 21:
        reduced2 = try_reduce_rank(best_U, best_V, best_W, 20)
        if reduced2 is not None:
            best_rank = 20
            best_U, best_V, best_W = reduced2
    
    # If we found rank 20, try to go lower
    if best_rank == 20:
        reduced3 = try_reduce_rank(best_U, best_V, best_W, 19)
        if reduced3 is not None:
            best_rank = 19
            best_U, best_V, best_W = reduced3
    
    # If we found rank 19, try to go lower
    if best_rank == 19:
        reduced4 = try_reduce_rank(best_U, best_V, best_W, 18)
        if reduced4 is not None:
            best_rank = 18
            best_U, best_V, best_W = reduced4
    
    # If we found rank 18, try to go lower
    if best_rank == 18:
        reduced5 = try_reduce_rank(best_U, best_V, best_W, 17)
        if reduced5 is not None:
            best_rank = 17
            best_U, best_V, best_W = reduced5
    
    # If we found rank 17, try to go lower
    if best_rank == 17:
        reduced6 = try_reduce_rank(best_U, best_V, best_W, 16)
        if reduced6 is not None:
            best_rank = 16
            best_U, best_V, best_W = reduced6
    
    # If we found rank 16, try to go lower
    if best_rank == 16:
        reduced7 = try_reduce_rank(best_U, best_V, best_W, 15)
        if reduced7 is not None:
            best_rank = 15
            best_U, best_V, best_W = reduced7
    
    # If we found rank 15, try to go lower
    if best_rank == 15:
        reduced8 = try_reduce_rank(best_U, best_V, best_W, 14)
        if reduced8 is not None:
            best_rank = 14
            best_U, best_V, best_W = reduced8
    
    # If we found rank 14, try to go lower
    if best_rank == 14:
        reduced9 = try_reduce_rank(best_U, best_V, best_W, 13)
        if reduced9 is not None:
            best_rank = 13
            best_U, best_V, best_W = reduced9
    
    # If we found rank 13, try to go lower
    if best_rank == 13:
        reduced10 = try_reduce_rank(best_U, best_V, best_W, 12)
        if reduced10 is not None:
            best_rank = 12
            best_U, best_V, best_W = reduced10
    
    # If we found rank 12, try to go lower
    if best_rank == 12:
        reduced11 = try_reduce_rank(best_U, best_V, best_W, 11)
        if reduced11 is not None:
            best_rank = 11
            best_U, best_V, best_W = reduced11
    
    # If we found rank 11, try to go lower
    if best_rank == 11:
        reduced12 = try_reduce_rank(best_U, best_V, best_W, 10)
        if reduced12 is not None:
            best_rank = 10
            best_U, best_V, best_W = reduced12
    
    # If we found rank 10, try to go lower
    if best_rank == 10:
        reduced13 = try_reduce_rank(best_U, best_V, best_W, 9)
        if reduced13 is not None:
            best_rank = 9
            best_U, best_V, best_W = reduced13
    
    # If we found rank 9, try to go lower
    if best_rank == 9:
        reduced14 = try_reduce_rank(best_U, best_V, best_W, 8)
        if reduced14 is not None:
            best_rank = 8
            best_U, best_V, best_W = reduced14
    
    # If we found rank 8, try to go lower
    if best_rank == 8:
        reduced15 = try_reduce_rank(best_U, best_V, best_W, 7)
        if reduced15 is not None:
            best_rank = 7
            best_U, best_V, best_W = reduced15
    
    # If we found rank 7, try to go lower
    if best_rank == 7:
        reduced16 = try_reduce_rank(best_U, best_V, best_W, 6)
        if reduced16 is not None:
            best_rank = 6
            best_U, best_V, best_W = reduced16
    
    # If we found rank 6, try to go lower
    if best_rank == 6:
        reduced17 = try_reduce_rank(best_U, best_V, best_W, 5)
        if reduced17 is not None:
            best_rank = 5
            best_U, best_V, best_W = reduced17
    
    # If we found rank 5, try to go lower
    if best_rank == 5:
        reduced18 = try_reduce_rank(best_U, best_V, best_W, 4)
        if reduced18 is not None:
            best_rank = 4
            best_U, best_V, best_W = reduced18
    
    # If we found rank 4, try to go lower
    if best_rank == 4:
        reduced19 = try_reduce_rank(best_U, best_V, best_W, 3)
        if reduced19 is not None:
            best_rank = 3
            best_U, best_V, best_W = reduced19
    
    # If we found rank 3, try to go lower
    if best_rank == 3:
        reduced20 = try_reduce_rank(best_U, best_V, best_W, 2)
        if reduced20 is not None:
            best_rank = 2
            best_U, best_V, best_W = reduced20
    
    # If we found rank 2, try to go lower
    if best_rank == 2:
        reduced21 = try_reduce_rank(best_U, best_V, best_W, 1)
        if reduced21 is not None:
            best_rank = 1
            best_U, best_V, best_W = reduced21
    
    # If we found rank 1, we're done
    
    # Convert to lists
    rank = best_rank
    U_final, V_final, W_final = best_U, best_V, best_W
    
    U_list = [[int(U_final[i, j]) if abs(U_final[i, j] - int(U_final[i, j])) < 1e-9 else float(U_final[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V_final[i, j]) if abs(V_final[i, j] - int(V_final[i, j])) < 1e-9 else float(V_final[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W_final[i, j]) if abs(W_final[i, j] - int(W_final[i, j])) < 1e-9 else float(W_final[i, j]) for j in range(rank)] for i in range(p * n)]
    
    # Final verification
    if not verify(U_list, V_list, W_list):
        # Fall back to base solution
        U, V, W = make_base_solution()
        rank = 22
        U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
