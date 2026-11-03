"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before 150s kill
    
    # The tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # with U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    # Lower bound on rank: at least max(n*p, m*p, n*m) / something
    # Actually for matmul, rank >= n*p is NOT a valid lower bound in general
    # But rank must be at least max(n, m, p) * min(n, m, p) / something
    # Let's just search from reasonable lower bounds
    
    # Known lower bounds: rank >= max(n*p, m*p, n*m) is wrong
    # rank >= n*m*p / max(n,m,p)^2 is a weak bound
    # For 2x3x7: n*m*p = 42, so rank >= 42/49 is weak
    # Better: the rank of the matmul tensor is at least max(n*p, m*p, n*m) / max(n,m,p)
    # Actually the standard lower bound is rank >= max(n*p, m*p, n*m) / max(n,m,p)
    # For 2x3x7: max(14,21,6) = 21, /7 = 3. That's too weak.
    
    # Let me think differently. The tensor has structure:
    # For fixed j, the slice T[:,j,:] is an n x p all-ones matrix
    # So rank >= rank of any slice... but that's not quite right for tensor rank.
    
    # A better approach: use the fact that for each j, we need
    # sum_c U[i*m+j,c] * W[k*n+i,c] = 1/V[j*p+k,c] ... no that's not right either.
    
    # Let me try a direct algebraic construction.
    # For small dimensions, we can try rank = r and solve the system.
    
    # Strategy: try ranks from a reasonable lower bound up to 42
    # Use the structure: for each column c, we have vectors u_c (len n*m), v_c (len m*p), w_c (len p*n)
    # Constraint: for all (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # Key insight: we can fix one factor and solve for others.
    # Or use a structured ansatz.
    
    # Let's try: for rank r, assume entries are in {-1, -0.5, 0, 0.5, 1}
    # and use a greedy/iterative approach.
    
    # Better: use the known structure of matmul decompositions.
    # For 2x3x7, let's try to find a decomposition with rank < 42.
    
    # Approach: systematic search with algebraic constraints
    # For each (i,j,k), the constraint is linear in each factor.
    
    # Let me try a different tactic: use the fact that we can combine columns.
    # Start with schoolbook and try to merge columns.
    
    # Actually, let me try a direct construction for small ranks.
    # For rank r, we need to find r triples (u_r, v_r, w_r) such that
    # the bilinear form reproduces the matmul.
    
    # Let's use a randomized approach with algebraic verification.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify exact decomposition using Fraction arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if total != 1:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, n, m, p):
        """Quick float check before exact verification."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    # Try to find decomposition with rank r
    # Use a structured approach: for small n, m, p, we can be more clever.
    
    # For 2x3x7, let's try ranks from 20 to 42
    # Use a greedy construction: build up the decomposition one column at a time.
    
    def greedy_construct(n, m, p, max_rank, time_budget):
        """Greedy construction of decomposition."""
        start_time = time.time()
        
        # The constraint: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
        # Equivalently: for each (i,j,k), the sum over c of the product = 1
        
        # Greedy: at each step, add a rank-1 component that best reduces the residual.
        # Residual R[i,j,k] = 1 - sum_{c added} u_c[i,j]*v_c[j,k]*w_c[k,i]
        
        # We want to find u, v, w such that u[i,j]*v[j,k]*w[k,i] approximates R[i,j,k]
        # This is a rank-1 approximation problem in 3D.
        
        # For exact decomposition, we need R to become exactly 0.
        
        # Strategy: try to find integer/half-integer solutions.
        # Assume all entries are multiples of 0.5.
        
        # Let's try a different approach: use the structure of the problem.
        # For fixed j, the constraint is:
        # sum_c u_c[i,j] * w_c[k,i] = 1 / v_c[j,k] ... no, that's not separable.
        
        # Actually: sum_c (u_c[i*m+j] * v_c[j*p+k]) * w_c[k*n+i] = 1
        # For fixed (i,k): sum_c (u_c[i*m+j] * v_c[j*p+k]) * w_c[k*n+i] = 1 for all j
        # This means: for each (i,k), the vector [u_c[i*m+j]*v_c[j*p+k] for j] dotted with w_c[k*n+i] = 1
        
        # Let me try yet another approach: use known results.
        # For 2x3x7, the minimal rank is likely around 20-30.
        
        # Let's use a systematic search with the following structure:
        # Assume U has a block structure based on i and j.
        
        # Actually, let me just try to find ANY decomposition with rank < 42
        # using a randomized local search.
        
        best_rank = n * m * p
        best_U = None
        best_V = None
        best_W = None
        
        # Schoolbook as initial
        U0 = [[0] * (n*m*p) for _ in range(n*m)]
        V0 = [[0] * (n*m*p) for _ in range(m*p)]
        W0 = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j][col] = 1
                    V0[j*p+k][col] = 1
                    W0[k*n+i][col] = 1
                    col += 1
        
        # Try to reduce rank by merging columns
        # For two columns c1, c2, can we replace them with one column?
        # We need u_new[i,j] * v_new[j,k] * w_new[k,i] = 
        #   u1[i,j]*v1[j,k]*w1[k,i] + u2[i,j]*v2[j,k]*w2[k,i]
        
        # This is generally not possible, but sometimes it is.
        
        # Let me try a different strategy: use the algebraic structure.
        # For n=2, m=3, p=7, let's try to find a decomposition with rank r.
        
        # Use a parameterized family of solutions.
        # Assume:
        # U[i*m+j, c] = a_c[i,j]
        # V[j*p+k, c] = b_c[j,k]  
        # W[k*n+i, c] = d_c[k,i]
        
        # Constraint: sum_c a_c[i,j] * b_c[j,k] * d_c[k,i] = 1
        
        # For small r, we can try to solve this system.
        
        # Let's try r = 21, 22, ..., 42 and see if we can find a solution.
        
        # Actually, let me try a smarter approach.
        # The tensor T[i,j,k] = 1 has a special structure.
        # We can write T = sum_c outer(u_c, v_c, w_c) where
        # u_c[i,j] = alpha_c[i] * beta_c[j]
        # v_c[j,k] = gamma_c[j] * delta_c[k]
        # w_c[k,i] = epsilon_c[k] * zeta_c[i]
        
        # Then sum_c alpha_c[i]*beta_c[j]*gamma_c[j]*delta_c[k]*epsilon_c[k]*zeta_c[i] = 1
        # = sum_c (alpha_c[i]*zeta_c[i]) * (beta_c[j]*gamma_c[j]) * (delta_c[k]*epsilon_c[k]) = 1
        
        # So we need: sum_c A_c[i] * B_c[j] * C_c[k] = 1 for all i,j,k
        # where A_c[i] = alpha_c[i]*zeta_c[i], B_c[j] = beta_c[j]*gamma_c[j], C_c[k] = delta_c[k]*epsilon_c[k]
        
        # This means the tensor is a sum of rank-1 tensors in the "separable" form.
        # But T[i,j,k] = 1 is itself separable: 1 = 1*1*1.
        # So we can take r=1 with A[0]=A[1]=1, B[0]=B[1]=B[2]=1, C[0]=...=C[6]=1.
        
        # But wait, this doesn't work because U, V, W have specific index structures.
        # U[i*m+j, c] must be a single number, not a product of separate i and j factors
        # unless we allow that structure.
        
        # Actually, U[i*m+j, c] CAN be written as alpha_c[i]*beta_c[j] if we allow
        # the entries to be products. But the constraint is that U is a matrix of shape (n*m, r).
        
        # So U[i*m+j, c] = alpha_c[i] * beta_c[j] is valid if we define it that way.
        
        # Similarly V[j*p+k, c] = gamma_c[j] * delta_c[k]
        # W[k*n+i, c] = epsilon_c[k] * zeta_c[i]
        
        # Then the constraint becomes:
        # sum_c alpha_c[i]*beta_c[j]*gamma_c[j]*delta_c[k]*epsilon_c[k]*zeta_c[i] = 1
        # = sum_c (alpha_c[i]*zeta_c[i]) * (beta_c[j]*gamma_c[j]) * (delta_c[k]*epsilon_c[k]) = 1
        
        # Let A_c[i] = alpha_c[i]*zeta_c[i], B_c[j] = beta_c[j]*gamma_c[j], C_c[k] = delta_c[k]*epsilon_c[k]
        # Then sum_c A_c[i]*B_c[j]*C_c[k] = 1
        
        # This is a tensor of rank r (in the usual tensor rank sense) that equals the all-ones tensor.
        # The all-ones tensor of size 2x3x7 has tensor rank... well, it's rank 1 since it's separable.
        # But we need A_c, B_c, C_c to be such that we can factor them back into
        # alpha_c, zeta_c, beta_c, gamma_c, delta_c, epsilon_c.
        
        # For A_c[i] = alpha_c[i]*zeta_c[i], we need A_c[0] = alpha_c[0]*zeta_c[0] and
        # A_c[1] = alpha_c[1]*zeta_c[1]. This is always possible (e.g., alpha_c = A_c, zeta_c = 1).
        
        # So the constraint reduces to: find r vectors A_c (len 2), B_c (len 3), C_c (len 7)
        # such that sum_c A_c[i]*B_c[j]*C_c[k] = 1 for all i,j,k.
        
        # The all-ones tensor 2x3x7 can be written as sum_c A_c ⊗ B_c ⊗ C_c.
        # Since the all-ones tensor is rank 1 (it's 1⊗1⊗1), we can take r=1 with
        # A_0 = [1,1], B_0 = [1,1,1], C_0 = [1,1,1,1,1,1,1].
        
        # Then U[i*m+j, 0] = alpha_0[i]*beta_0[j] = A_0[i]*1 = A_0[i] (if zeta_0 = 1)
        # Wait, I need to be more careful.
        
        # Let me redo this. We have:
        # U[i*m+j, c] = alpha_c[i] * beta_c[j]
        # V[j*p+k, c] = gamma_c[j] * delta_c[k]
        # W[k*n+i, c] = epsilon_c[k] * zeta_c[i]
        
        # Product: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # = alpha_c[i]*beta_c[j]*gamma_c[j]*delta_c[k]*epsilon_c[k]*zeta_c[i]
        # = (alpha_c[i]*zeta_c[i]) * (beta_c[j]*gamma_c[j]) * (delta_c[k]*epsilon_c[k])
        
        # Let A_c[i] = alpha_c[i]*zeta_c[i], B_c[j] = beta_c[j]*gamma_c[j], C_c[k] = delta_c[k]*epsilon_c[k]
        # Then we need sum_c A_c[i]*B_c[j]*C_c[k] = 1
        
        # For r=1: A_0[i]*B_0[j]*C_0[k] = 1 for all i,j,k
        # This requires A_0[i] = a_i, B_0[j] = b_j, C_0[k] = c_k with a_i*b_j*c_k = 1
        # So a_i = 1/a_i, b_j = 1/b_j, c_k = 1/c_k... 
        # Actually: a_0*b_0*c_0 = 1, a_1*b_0*c_0 = 1, so a_0 = a_1.
        # Similarly b_0 = b_1 = b_2, c_0 = c_1 = ... = c_6.
        # So A_0 = [a, a], B_0 = [b, b, b], C_0 = [c, c, c, c, c, c, c] with a*b*c = 1.
        
        # Then:
        # U[i*m+j, 0] = alpha_0[i]*beta_0[j]
        # V[j*p+k, 0] = gamma_0[j]*delta_0[k]
        # W[k*n+i, 0] = epsilon_0[k]*zeta_0[i]
        
        # With A_0[i] = alpha_0[i]*zeta_0[i] = a
        # B_0[j] = beta_0[j]*gamma_0[j] = b
        # C_0[k] = delta_0[k]*epsilon_0[k] = c
        
        # We can choose:
        # alpha_0[i] = 1, zeta_0[i] = a for all i
        # beta_0[j] = 1, gamma_0[j] = b for all j
        # delta_0[k] = 1, epsilon_0[k] = c for all k
        
        # Then U[i*m+j, 0] = 1*1 = 1 for all i,j
        # V[j*p+k, 0] = 1*1 = 1 for all j,k
        # W[k*n+i, 0] = 1*1 = 1 for all k,i
        
        # Check: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. 
        # But we need the sum over c to equal 1, and with r=1, this works!
        
        # Wait, but this gives rank 1? That can't be right for matrix multiplication.
        # Let me re-examine.
        
        # The issue: U[i*m+j, c] is a single entry, not a product. I was assuming
        # it can be factored as alpha_c[i]*beta_c[j], but that's an additional constraint
        # on the form of U.
        
        # In general, U[i*m+j, c] is an arbitrary number, not necessarily factorable.
        # So my separable ansatz is too restrictive.
        
        # Let me go back to basics. The correct formulation is:
        # We need matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
        # for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a system of n*m*p = 42 equations in r*(n*m + m*p + p*n) = r*(6+21+14) = 41r unknowns.
        # For r=1: 42 equations, 41 unknowns. Generally no solution.
        # For r=2: 42 equations, 82 unknowns. Should have solutions.
        
        # So rank 2 might work! Let me try to find a rank-2 decomposition.
        
        # For r=2:
        # U has shape (6, 2), V has shape (21, 2), W has shape (14, 2)
        # Total unknowns: 12 + 42 + 28 = 82
        # Equations: 42
        
        # Let me set up the system and try to solve it.
        
        # Actually, let me try a constructive approach.
        # Assume:
        # U[i*m+j, 0] = u0_ij, U[i*m+j, 1] = u1_ij
        # V[j*p+k, 0] = v0_jk, V[j*p+k, 1] = v1_jk
        # W[k*n+i, 0] = w0_ki, W[k*n+i, 1] = w1_ki
        
        # Constraint: u0_ij*v0_jk*w0_ki + u1_ij*v1_jk*w1_ki = 1
        
        # This is a sum of two rank-1 tensors.
        
        # Let me try a simple ansatz:
        # u0_ij = a_i * b_j, u1_ij = c_i * d_j
        # v0_jk = e_j * f_k, v1_jk = g_j * h_k
        # w0_ki = p_k * q_i, w1_ki = r_k * s_i
        
        # Then: (a_i*b_j*e_j*f_k*p_k*q_i) + (c_i*d_j*g_j*h_k*r_k*s_i) = 1
        # = (a_i*q_i)*(b_j*e_j)*(f_k*p_k) + (c_i*s_i)*(d_j*g_j)*(h_k*r_k) = 1
        
        # Let A0_i = a_i*q_i, B0_j = b_j*e_j, C0_k = f_k*p_k
        # Let A1_i = c_i*s_i, B1_j = d_j*g_j, C1_k = h_k*r_k
        
        # Then A0_i*B0_j*C0_k + A1_i*B1_j*C1_k = 1
        
        # For this to hold for all i,j,k, we can try:
        # A0_i = 1, B0_j = 1, C0_k = 1/2 for all i,j,k
        # A1_i = 1, B1_j = 1, C1_k = 1/2 for all i,j,k
        # Then 1*1*1/2 + 1*1*1/2 = 1. 
        
        # So:
        # a_i*q_i = 1, b_j*e_j = 1, f_k*p_k = 1/2
        # c_i*s_i = 1, d_j*g_j = 1, h_k*r_k = 1/2
        
        # Choose:
        # a_i = 1, q_i = 1 for all i
        # b_j = 1, e_j = 1 for all j
        # f_k = 1/2, p_k = 1 for all k
        # c_i = 1, s_i = 1 for all i
        # d_j = 1, g_j = 1 for all j
        # h_k = 1/2, r_k = 1 for all k
        
        # Then:
        # U[i*m+j, 0] = a_i * b_j = 1
        # U[i*m+j, 1] = c_i * d_j = 1
        # V[j*p+k, 0] = e_j * f_k = 1/2
        # V[j*p+k, 1] = g_j * h_k = 1/2
        # W[k*n+i, 0] = p_k * q_i = 1
        # W[k*n+i, 1] = r_k * s_i = 1
        
        # Check: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for c=0: 1*(1/2)*1 = 1/2
        # For c=1: 1*(1/2)*1 = 1/2
        # Sum = 1/2 + 1/2 = 1. 
        
        # So rank 2 works! Let me verify this is correct.
        
        # U shape: (n*m, 2) = (6, 2)
        # V shape: (m*p, 2) = (21, 2)
        # W shape: (p*n, 2) = (14, 2)
        
        # U[i*m+j, 0] = 1 for all i,j
        # U[i*m+j, 1] = 1 for all i,j
        # V[j*p+k, 0] = 1/2 for all j,k
        # V[j*p+k, 1] = 1/2 for all j,k
        # W[k*n+i, 0] = 1 for all k,i
        # W[k*n+i, 1] = 1 for all k,i
        
        # Let me verify with a specific example:
        # i=0, j=0, k=0:
        # U[0*3+0, 0]*V[0*7+0, 0]*W[0*2+0, 0] + U[0*3+0, 1]*V[0*7+0, 1]*W[0*2+0, 1]
        # = 1*(1/2)*1 + 1*(1/2)*1 = 1/2 + 1/2 = 1. 
        
        # i=0, j=0, k=1:
        # U[0,0]*V[1,0]*W[2,0] + U[0,1]*V[1,1]*W[2,1]
        # = 1*(1/2)*1 + 1*(1/2)*1 = 1. 
        
        # This seems to work! Rank 2.
        
        # But wait, I need to double-check the index conventions.
        # The problem says: output factor follows W[k*n+i,r]
        # So W has shape (p*n, rank) with W[k*n+i, c]
        
        # And U has shape (n*m, rank) with U[i*m+j, c]
        # V has shape (m*p, rank) with V[j*p+k, c]
        
        # Yes, that matches what I have.
        
        # Let me construct the actual matrices.
        
        r = 2
        U = [[1, 1] for _ in range(n*m)]  # 6 rows, each [1, 1]
        V = [[0.5, 0.5] for _ in range(m*p)]  # 21 rows, each [0.5, 0.5]
        W = [[1, 1] for _ in range(p*n)]  # 14 rows, each [1, 1]
        
        # Verify
        if verify_decomposition_fast(U, V, W, n, m, p):
            if verify_decomposition(U, V, W, n, m, p):
                # Convert to half-integers (they already are)
                U_out = [[x if x == int(x) else x for x in row] for row in U]
                V_out = [[x if x == int(x) else x for x in row] for row in V]
                W_out = [[x if x == int(x) else x for x in row] for row in W]
                return {"U": U_out, "V": V_out, "W": W_out}
        
        # If rank 2 didn't work (shouldn't happen), fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": W, "W": W}
# EVOLVE-BLOCK-END

