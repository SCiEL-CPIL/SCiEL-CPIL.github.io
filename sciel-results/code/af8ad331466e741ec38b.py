"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
# n=3, m=5, p=6
N, M, P = 3, 5, 6
SCHOOLBOOK_RANK = N * M * P  # 90

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s hard limit
    
    n, m, p = N, M, P
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    
    # The multiplication tensor T[i,j,k] = delta_{ik} for the (i,k)-th entry
    # We need U, V, W such that for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if the (i,k) entry is produced by j
    # Actually, the matrix multiplication tensor is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{i,j,k} = 1 (it's just the structure of multiplication)
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)
    # W has shape (p*n, rank), indexed by (k,i)
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (the tensor is just all ones in the
    # appropriate sense - actually no, the multiplication tensor has T[i,j,k]=1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient of A[i,j]*B[j,k] in C[i,k] is 1).
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is equivalent to: the rank-1 tensors U_c ⊗ V_c ⊗ W_c sum to the all-ones tensor
    # where U_c is a vector in R^{nm}, V_c in R^{mp}, W_c in R^{pn}.
    
    # Let's think about this differently. We need to decompose the matrix multiplication tensor.
    # The standard approach: use known low-rank decompositions.
    
    # For 3x5 * 5x6, let's try to find a decomposition using the structure.
    # One approach: block the matrices and use known decompositions for smaller blocks.
    
    # Actually, let's try a direct search approach with integer/rational entries.
    # We'll try ranks from a lower bound up to the schoolbook rank.
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Also, the tensor rank is at least the max of the slice ranks.
    # For matrix multiplication, the rank is at least n*m*p / max(n,m,p) roughly.
    
    # Let's try to use a structured approach.
    # Key insight: if we can find matrices U, V, W with the property that
    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
    
    # Strategy: Try small ranks with random integer vectors, then verify.
    # But random search in high dimension is unlikely to work.
    
    # Better strategy: Use the fact that we can scale. If we find a decomposition with
    # rational entries, we can scale to integers.
    
    # Let's try a more systematic approach:
    # For rank r, we need r vectors u_c in R^{nm}, v_c in R^{mp}, w_c in R^{pn}
    # such that sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    
    # This is a system of nm*p = 90 equations in 3*r unknowns (well, r*(nm+mp+pn) = r*63 unknowns).
    # For r=30: 1890 unknowns, 90 equations. Very underdetermined.
    
    # Let's try a constructive approach using the structure of matrix multiplication.
    
    # Approach: Use the "border rank" or "tensor rank" known results.
    # For multiplying an n×m matrix by an m×p matrix, the rank is at most n*m*p (schoolbook).
    # There are known improvements for small cases.
    
    # Let me try a specific construction:
    # If we let r = n*m*p - something, we might save some rank.
    
    # Actually, let's try a different approach: use the fact that we can combine terms.
    # For example, if two terms have the same (i,j) or (j,k) or (k,i) index, we might combine them.
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms.
    
    # Schoolbook: for each (i,j,k), we have one term with U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # Can we merge two terms? If we have terms for (i1,j1,k1) and (i2,j2,k2), can we replace
    # them with a single term? Only if i1=i2, j1=j2, k1=k2 (trivial) or if we can find
    # u, v, w such that u[i1,j1]*v[j1,k1]*w[k1,i1] + u[i2,j2]*v[j2,k2]*w[k2,i2] = 1
    # and for all other (i,j,k), the contribution is 0.
    
    # This is hard. Let me try a different approach.
    
    # Known result: The rank of matrix multiplication for small matrices.
    # For 2x2 * 2x2, rank is 7 (Strassen).
    # For 3x3 * 3x3, rank is at most 23 (Lader).
    
    # For 3x5 * 5x6, let's estimate. The "standard" lower bound is max(nm, mp, pn) = 30.
    # A better lower bound might be available.
    
    # Let me try to construct a decomposition with rank < 90.
    
    # Idea: Use the Strassen-like idea. For multiplying two 2x2 matrices, we use 7 products
    # instead of 8. The key is to find linear combinations.
    
    # For our case, let's try to use a block structure.
    # Split the 3x5 matrix into blocks and the 5x6 matrix into blocks.
    
    # Actually, let's try a simpler approach: use random vectors and solve the linear system.
    # For a given rank r, pick random u_c and v_c, then solve for w_c.
    # The condition is: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k.
    # For fixed u_c and v_c, this is a linear system in w_c.
    # We have nm*p = 90 equations and r*pn = 18r unknowns.
    # For r >= 5, we have 90 <= 90, so it might be solvable.
    
    # Let's try this approach:
    # 1. Pick random rational u_c and v_c.
    # 2. Set up the linear system for w_c.
    # 3. Solve and check if the solution is rational (or can be made rational).
    
    # But this is a nonlinear problem in general. Let me think again.
    
    # Alternative: Use the fact that we can choose the structure of U, V, W.
    # For example, if we make U and V sparse, the problem becomes easier.
    
    # Let me try a specific construction:
    # For each i, j, k, we need the sum to be 1.
    # If we use r = nm + mp + pn - 2 or something, we might be able to do it.
    
    # Actually, let's try the simplest possible improvement: use the fact that
    # we can share computation.
    
    # Let me try a direct construction with rank = 60 or so.
    
    # Here's an idea: for each pair (i,k), the sum over j of A[i,j]*B[j,k] gives C[i,k].
    # We can think of this as: for each (i,k), we need to compute a dot product of length m=5.
    # The rank of a single dot product of length m is m (using the standard basis).
    # But we have n*p = 18 such dot products, and they share the A and B entries.
    
    # Let me try a different approach: use the "tensor rank" of the matrix multiplication tensor.
    # For 3x5*5x6, the rank is at most 90 (schoolbook).
    
    # Let me try to find a decomposition with rank 89, 88, etc. by modifying the schoolbook.
    
    # In the schoolbook, each term corresponds to a unique (i,j,k).
    # Can we replace two terms with one? If we have terms for (i,j,k1) and (i,j,k2),
    # can we replace them with a single term?
    # We need: u[i,j]*v[j,k1]*w[k1,i] + u[i,j]*v[j,k2]*w[k2,i] = 1 for (i,j,k1) and (i,j,k2),
    # and 0 for all other (i',j',k').
    # If u[i,j] = 1, then we need v[j,k1]*w[k1,i] + v[j,k2]*w[k2,i] = 1 for k=k1,
    # and v[j,k1]*w[k1,i] + v[j,k2]*w[k2,i] = 1 for k=k2? No, that's not right.
    
    # Let me be more careful. The condition is:
    # For (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # For (i',j',k') with (i',j',k') != (i,j,k): the sum must be 0.
    
    # If we have two schoolbook terms, say for (i1,j1,k1) and (i2,j2,k2), and we want to
    # replace them with a single term, we need:
    # U[i1*m+j1,c]*V[j1*p+k1,c]*W[k1*n+i1,c] = 1
    # U[i2*m+j2,c]*V[j2*p+k2,c]*W[k2*n+i2,c] = 1
    # And for all other (i,j,k), the product must be 0.
    
    # This is only possible if the supports don't overlap, which is hard.
    
    # Let me try a completely different approach: use known algorithms for matrix multiplication.
    
    # For 3x5 * 5x6, we can use the following:
    # 1. Pad to 4x6 * 6x8 or something? No, that increases the rank.
    # 2. Use Strassen for sub-blocks.
    
    # Let's try to use Strassen for 2x2 blocks.
    # Split the 3x5 matrix into:
    #   A = [[A11, A12], [A21, A22], [A31, A32]]
    # where A11, A21, A31 are 1x2 and A12, A22, A32 are 1x3.
    # This doesn't work well with Strassen.
    
    # Let me try: split into 2x2 and 1x3 blocks.
    # A = [[A11, A12], [A21, A22], [A31, A32]]
    # B = [[B11, B12], [B21, B22], [B31, B32]]
    # where B11, B21, B31 are 2x3 and B12, B22, B32 are 2x3? No, B is 5x6.
    
    # Let me be more careful. A is 3x5, B is 5x6.
    # Split A into [A_top (2x5); A_bottom (1x5)]
    # Split B into [B_left (5x3) | B_right (5x3)]
    
    # Then C = A*B = [A_top*B_left, A_top*B_right; A_bottom*B_left, A_bottom*B_right]
    # Each block is 2x3 or 1x3.
    
    # A_top*B_left: 2x5 * 5x3. Rank at most 2*5*3 = 30.
    # But we can use Strassen-like methods.
    
    # This is getting complicated. Let me try a numerical approach.
    
    # Use numpy to solve for a decomposition with a given rank.
    # For rank r, we have r*(nm+mp+pn) = 63r variables and 90 equations.
    # For r >= 2, it's underdetermined.
    
    # Let's try to find a solution with small integer entries.
    
    # Approach: Start with schoolbook (rank 90) and try to reduce the rank.
    # Or, start from scratch and try to find a low-rank decomposition.
    
    # Let me try a random approach with numpy:
    # For a given rank r, generate random U and V, then solve for W.
    # The system is: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    # This is linear in W if U and V are fixed.
    
    # Let's set up the linear system:
    # For each (i,j,k), we have: sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
    # This is a system of 90 equations in r*18 = 18r unknowns (the W entries).
    
    # For r=5: 90 equations, 90 unknowns. Might be solvable.
    # For r=6: 90 equations, 108 unknowns. Underdetermined.
    
    # Let's try r=5 first.
    
    import numpy as np
    
    best_rank = SCHOOLBOOK_RANK
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(nm, mp, pn) = 30? No, that's not right.
    # The rank is at least the max of the "fiber" ranks.
    # For matrix multiplication, the rank is at least n*m*p / max(n,m,p)^2 roughly.
    # For 3x5*5x6: 90/25 = 3.6, so at least 4.
    
    # Let's try ranks from 5 to 90.
    
    for r in range(5, SCHOOLBOOK_RANK + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + r * 1000 + trial)
            
            # Generate random U and V
            U_rand = np.random.randn(nm, r)
            V_rand = np.random.randn(mp, r)
            
            # Set up the linear system for W
            # For each (i,j,k): sum_c U_rand[i*m+j,c]*V_rand[j*p+k,c]*W[k*n+i,c] = 1
            # This is: A_w * w_flat = b
            # where A_w is 90 x (18*r)
            
            A_w = np.zeros((nm * p, pn * r))
            b = np.ones(nm * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        for c in range(r):
                            col_idx = (k * n + i) * r + c
                            A_w[eq_idx, col_idx] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
            
            # Solve A_w * w_flat = b
            try:
                w_flat, residuals, rank_w, sv = np.linalg.lstsq(A_w, b, rcond=None)
                # Check if the solution is good
                if residuals.size > 0 and residuals[0] < 1e-10:
                    W_rand = w_flat.reshape(pn, r)
                    # Verify the solution
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = W_rand.tolist()
                    
                    # Check all conditions
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(r):
                                    s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                                if abs(s - 1.0) > 1e-8:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                    
                    if valid:
                        best_rank = r
                        best_U = U_rand
                        best_V = V_rand
                        best_W = W_rand
                        break  # Found a solution for this rank, move to next rank
            except Exception:
                pass
        
        if best_rank < SCHOOLBOOK_RANK:
            # Found a better solution, can stop
            break
    
    # If we found a solution, convert to lists
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: schoolbook
    U = [[0] * SCHOOLBOOK_RANK for _ in range(nm)]
    V = [[0] * SCHOOLBOOK_RANK for _ in range(mp)]
    W = [[0] * SCHOOLBOOK_RANK for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
