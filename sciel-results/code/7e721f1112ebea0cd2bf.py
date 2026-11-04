import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        """Schoolbook decomposition with rank n*m*p"""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Strategy: Use exact rational arithmetic to find minimal rank
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find minimal rank R such that T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    # Key insight: For matrix multiplication, we can use the structure to find
    # linear dependencies. The rank of the matricizations gives lower bounds.
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Lower bounds from matricization ranks
    T_mat1 = T.reshape(n * m, -1)
    T_mat2 = T.reshape(m * p, -1)
    T_mat3 = T.reshape(p * n, -1)
    
    rank_lb1 = np.linalg.matrix_rank(T_mat1)
    rank_lb2 = np.linalg.matrix_rank(T_mat2)
    rank_lb3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lb1, rank_lb2, rank_lb3)
    
    if time.time() > deadline - 30:
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to find a decomposition with rank = rank_lower using exact arithmetic
    # Approach: Set up a linear system where we try to express T as a sum of rank-1 tensors
    # with the minimum number of terms
    
    # For this specific problem (2x4x7), let's use a structured approach:
    # The multiplication can be viewed as 2 independent rows of A times B
    # Row i of C = Row i of A * B
    
    # Let's try a block-based approach with exact rational coefficients
    # A is 2x4, B is 4x7, C is 2x7
    
    # We can decompose this as:
    # - 2x2 blocks using Strassen (rank 7 each)
    # - Handle the remaining columns
    
    # A: 2 rows, 4 cols -> split cols into (0,1) and (2,3)
    # B: 4 rows, 7 cols -> split rows into (0,1) and (2,3), cols into (0,1),(2,3),(4,5),(6)
    
    # For the 2x2 x 2x2 blocks (6 of them: 2 row-pairs x 3 col-pairs), use Strassen
    # For the remaining column 6, use direct multiplication
    
    U = np.zeros((n * m, 100))  # Start with generous rank
    V = np.zeros((m * p, 100))
    W = np.zeros((p * n, 100))
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2x2 block"""
        nonlocal col
        # M1: (a+d)(e+h)
        U[i0 * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1
        
        # M2: (c+d)e
        U[(i0 + 1) * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] -= 1
        col += 1
        
        # M3: a(f-h)
        U[i0 * m + j0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1
        
        # M4: d(g-e)
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + k0, col] += 1
        V[j0 * p + k0, col] -= 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        col += 1
        
        # M5: (a+b)h
        U[i0 * m + j0, col] += 1
        U[i0 * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        col += 1
        
        # M6: (c-a)(e+f)
        U[(i0 + 1) * m + j0, col] += 1
        U[i0 * m + j0, col] -= 1
        V[j0 * p + k0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1
        
        # M7: (b-d)(g+h)
        U[i0 * m + (j0 + 1), col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] -= 1
        V[(j0 + 1) * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        col += 1
    
    # Add Strassen blocks for the 2x2 x 2x2 subproblems
    # j0 in {0, 2} for A col pairs / B row pairs
    # k0 in {0, 2, 4} for B col pairs
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
    
    # Handle column 6 of B (and C) with direct rank-1 terms
    for j in range(m):
        for i in range(n):
            U[i * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + i, col] = 1
            col += 1
    
    # Trim to actual rank used
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    actual_rank = col
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce the rank using exact linear algebra
    # Check for linear dependencies among the rank-1 terms
    # For each column r, check if u_r ⊗ v_r ⊗ w_r is in the span of the others
    
    # Build the matricization and find the null space
    # T = sum_r u_r v_r^T w_r^T
    # Matricize along first mode: T_mat1[a, bc] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # To check if column r is redundant, we need to see if there exist
    # coefficients alpha_s (s != r) such that:
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_{s != r} alpha_s * U[:,s] ⊗ V[:,s] ⊗ W[:,s]
    
    # This is a bilinear constraint, so it's not straightforward.
    # Instead, let's try a greedy approach: try removing each column and
    # see if we can solve for the remaining columns to still match T.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = actual_rank
    
    # Try removing columns one at a time (greedy)
    for r in range(actual_rank):
        if time.time() > deadline - 10:
            break
        
        # Skip if this would make rank < lower bound
        if actual_rank - 1 < rank_lower:
            break
        
        # Try removing column r
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        
        # Check if the remaining columns can still represent T
        # This requires solving a nonlinear system, which is hard.
        # Instead, use a simpler heuristic: check if the matricization
        # of the remaining terms has the same rank as T's matricization.
        
        recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
        residual = T - recon
        
        # If residual is zero, we can remove this column
        if np.allclose(residual, 0, atol=1e-9):
            U, V, W = U_test, V_test, W_test
            actual_rank -= 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = actual_rank
            r -= 1  # Re-check the same index since we removed a column
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
