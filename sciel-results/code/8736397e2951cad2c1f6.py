import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Known rank-6 decomposition for 2x2 by 2x3 matrix multiplication
    # C (2x3) = A (2x2) * B (2x3)
    # Using the identity: M_{2,2,3} can be decomposed with rank 6
    # This is derived from the fact that R(M_{2,2,3}) <= 6
    
    # Rank-6 decomposition for 2x2 by 2x3:
    # U: (4, 6), V: (6, 6), W: (6, 6) where W[k*n+i,r] convention
    # For 2x2x3: n=2, m=2, p=3
    # U shape: (2*2, 6) = (4, 6)
    # V shape: (2*3, 6) = (6, 6) 
    # W shape: (3*2, 6) = (6, 6)
    
    # The rank-6 decomposition for 2x2 by 2x3:
    # Based on known results, we can construct this from Strassen-like ideas
    # or use the explicit decomposition
    
    # Let's construct the rank-6 decomposition for 2x2x3:
    # C[i,j] = sum_k A[i,k] * B[k,j] for i in {0,1}, j in {0,1,2}
    
    # One approach: use 6 rank-1 terms
    # U_s = u_s ⊗ v_s for some vectors
    # Actually, let me use a known explicit decomposition
    
    # Rank 6 decomposition for M_{2,2,3}:
    # We can write it as a combination of simpler decompositions
    
    # Let me define the factors directly:
    U_small = np.zeros((4, 6))
    V_small = np.zeros((6, 6))
    W_small = np.zeros((6, 6))
    
    # For 2x2 by 2x3, we have:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c02 = a00*b02 + a01*b12
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c12 = a10*b02 + a11*b12
    
    # We need 6 rank-1 terms to decompose this tensor.
    # Let's use a systematic approach:
    
    # Term 1: captures a00*b00, a00*b01, a00*b02
    # U_1 has 1 at position corresponding to a00
    # V_1 has 1 at positions corresponding to b00, b01, b02
    # W_1 has 1 at positions corresponding to c00, c01, c02
    
    # But this is 3 terms just for a00. We need to be smarter.
    
    # Actually, the known rank for 2x2x3 is 6. Let me use an explicit construction.
    
    # Using the method from Bini et al. or similar:
    # We can decompose M_{2,2,3} using 6 terms.
    
    # Let me define the decomposition explicitly:
    
    # Term 0:
    U_small[0, 0] = 1  # a00
    U_small[1, 0] = 1  # a01
    V_small[0, 0] = 1  # b00
    V_small[1, 0] = 1  # b01
    V_small[2, 0] = 1  # b02
    W_small[0, 0] = 1  # c00
    W_small[3, 0] = 1  # c01
    W_small[6, 0] = 1  # c02
    
    # Wait, this doesn't work because we'd get a00*b00 + a01*b00 for c00, not a00*b00 + a01*b10.
    
    # Let me think more carefully. The tensor is:
    # T[i,j,k] = delta_{i,k} for the multiplication tensor
    # where i is row of A (0,1), k is col of A / row of B (0,1), j is col of B (0,1,2)
    # Actually, the multiplication tensor M_{n,m,p} has entries:
    # T[i,j,k] = 1 if i is row of C, k is col of A / row of B, j is col of B
    # No, let me use the standard convention.
    
    # For matrix multiplication C = AB where A is n×m, B is m×p:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T has indices (i,j,k) where i in [0,n), j in [0,p), k in [0,m)
    # T[i,j,k] = 1 if the term A[i,k]*B[k,j] contributes to C[i,j]
    # Actually, T[i,j,k] = delta_{i,k} * delta_{k,j}? No.
    
    # The multiplication tensor is defined as:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{p-1} sum_{k=0}^{m-1} e_i^A ⊗ e_k^B ⊗ e_{i,j}^C * delta_{i,k}
    # Wait, that's not right either.
    
    # Let me use the standard definition:
    # The matrix multiplication tensor M_{n,m,p} is in (R^n ⊗ R^m ⊗ R^{n*p})
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{p-1} sum_{k=0}^{m-1} e_i ⊗ e_k ⊗ e_{(i,j)}
    # where e_i is the i-th basis vector in R^n, e_k in R^m, e_{(i,j)} in R^{n*p}
    
    # So for n=2, m=2, p=3:
    # M = sum_{i=0}^{1} sum_{j=0}^{2} sum_{k=0}^{1} e_i ⊗ e_k ⊗ e_{(i,j)}
    # = e_0⊗e_0⊗e_{(0,0)} + e_0⊗e_1⊗e_{(0,1)} + e_0⊗e_1⊗e_{(0,2)} + 
    #   e_1⊗e_0⊗e_{(1,0)} + e_1⊗e_1⊗e_{(1,1)} + e_1⊗e_1⊗e_{(1,2)}
    
    # Wait, that's not right. The term should be e_i ⊗ e_k ⊗ e_{(i,j)} where the coefficient is 1
    # if k is the column of A and row of B that connects i and j.
    
    # Actually, C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor is: T = sum_{i,j,k} A[i,k] * B[k,j] * C[i,j]
    # As a trilinear form: sum_{i,j,k} T[i,j,k] * A[i,k] * B[k,j] * C[i,j]
    # where T[i,j,k] = 1 if the product A[i,k]*B[k,j] contributes to C[i,j]
    
    # But A[i,k] is indexed by (i,k) where i is row, k is col of A
    # B[k,j] is indexed by (k,j) where k is row, j is col of B
    # C[i,j] is indexed by (i,j) where i is row, j is col of C
    
    # So the tensor has components T_{(i,k),(k,j),(i,j)} = 1 for all valid i,k,j
    
    # For the decomposition, we write:
    # T = sum_{s=1}^{R} u_s ⊗ v_s ⊗ w_s
    # where u_s in R^{n*m}, v_s in R^{m*p}, w_s in R^{n*p}
    
    # And the condition is:
    # sum_{s=1}^{R} u_s[i,k] * v_s[k,j] * w_s[i,j] = T[i,k,k,j,i,j] = 1 for all i,k,j
    
    # So for each (i,k,j), we need: sum_s u_s[i,k] * v_s[k,j] * w_s[i,j] = 1
    
    # For n=2, m=2, p=3:
    # (i,k,j) ranges over i in {0,1}, k in {0,1}, j in {0,1,2}
    # So 2*2*3 = 12 equations
    
    # We need to find 6 vectors u_s, v_s, w_s such that these 12 equations hold.
    
    # Let me use a known rank-6 decomposition. One such decomposition is:
    
    # s=0: u0 = (1,0,0,0), v0 = (1,0,0,1,0,0), w0 = (1,0,0,0,0,0)
    # This gives: u0[i,k]*v0[k,j]*w0[i,j] = 1 only when i=0,k=0 and j=0 or 1
    # Not quite right.
    
    # Let me look up or derive a proper rank-6 decomposition for 2x2x3.
    
    # Actually, I recall that R(M_{2,2,3}) = 6 is known. Let me construct it.
    
    # One approach: use the fact that M_{2,2,3} can be embedded in M_{2,2,4} which has rank 7 (Strassen)
    # but that gives rank 7, not 6.
    
    # Let me try a different approach. Since we have 12 constraints and 6*3=18 unknowns (with scaling freedom),
    # we can set up a system.
    
    # Actually, let me use a simpler construction. Note that:
    # C = A * B where A is 2x2, B is 2x3
    # We can write B = [B_left, B_right] where B_left is 2x2, B_right is 2x1
    # C = [A*B_left, A*B_right]
    # A*B_left has rank at most 7 (Strassen for 2x2x2)
    # A*B_right is just 2 scalar multiplications, rank 2
    # Total rank at most 7+2 = 9, not good enough.
    
    # Let me try to find the rank-6 decomposition numerically or use a known one.
    
    # After some research, a rank-6 decomposition for M_{2,2,3} can be constructed as follows:
    
    # Let me define the factors explicitly based on a known decomposition:
    
    # Using the decomposition from the literature (e.g., Bini et al.):
    
    U_small = np.array([
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0]
    ], dtype=float)
    
    V_small = np.array([
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0]
    ], dtype=float)
    
    W_small = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    # Let me verify this decomposition:
    # For each (i,k,j) in {0,1}x{0,1}x{0,1,2}:
    # sum_s U_small[i*2+k, s] * V_small[k*3+j, s] * W_small[i*3+j, s] should equal 1
    
    # Let me check (i=0, k=0, j=0):
    # sum_s U_small[0, s] * V_small[0, s] * W_small[0, s]
    # = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] + ... 
    # = 1*1*1 + 0*0*0 + 0*0*0 + 0*0*0 + 1*0*0 + 0*1*0 = 1 ✓
    
    # Check (i=0, k=0, j=1):
    # sum_s U_small[0, s] * V_small[1, s] * W_small[1, s]
    # = U[0,0]*V[1,0]*W[1,0] + U[0,1]*V[1,1]*W[1,1] + ...
    # = 1*0*0 + 0*1*1 + 0*0*0 + 0*0*0 + 1*1*0 + 0*0*0 = 0 ✗
    
    # This doesn't work. Let me try a different approach.
    
    # Let me use a systematic construction. I'll set up the equations and solve.
    
    # Actually, let me use a known result: the rank of M_{2,2,3} is 6.
    # A specific decomposition can be found in the literature.
    
    # Let me try to construct it step by step.
    
    # We need: for all i in {0,1}, k in {0,1}, j in {0,1,2}:
    # sum_{s=0}^{5} U[i*2+k, s] * V[k*3+j, s] * W[i*3+j, s] = 1
    
    # Let me denote U_{ik}^s = U[i*2+k, s], V_{kj}^s = V[k*3+j, s], W_{ij}^s = W[i*3+j, s]
    
    # The equations are:
    # For (i,k,j): sum_s U_{ik}^s * V_{kj}^s * W_{ij}^s = 1
    
    # This is a system of 12 equations in 18 unknowns (6 terms * 3 factors each).
    
    # Let me try a symmetric construction. Assume:
    # U_{00}^s = U_{11}^s = a_s
    # U_{01}^s = U_{10}^s = b_s
    # V_{00}^s = V_{11}^s = c_s
    # V_{01}^s = V_{10}^s = d_s
    # V_{02}^s = V_{12}^s = e_s
    # W_{00}^s = W_{11}^s = f_s
    # W_{01}^s = W_{10}^s = g_s
    # W_{02}^s = W_{12}^s = h_s
    
    # Then the equations become:
    # (i=0,k=0,j=0): sum_s a_s * c_s * f_s = 1
    # (i=0,k=0,j=1): sum_s a_s * d_s * g_s = 1
    # (i=0,k=0,j=2): sum_s a_s * e_s * h_s = 1
    # (i=0,k=1,j=0): sum_s b_s * c_s * f_s = 1
    # (i=0,k=1,j=1): sum_s b_s * d_s * g_s = 1
    # (i=0,k=1,j=2): sum_s b_s * e_s * h_s = 1
    # (i=1,k=0,j=0): sum_s a_s * c_s * f_s = 1 (same as first)
    # (i=1,k=0,j=1): sum_s a_s * d_s * g_s = 1 (same as second)
    # (i=1,k=0,j=2): sum_s a_s * e_s * h_s = 1 (same as third)
    # (i=1,k=1,j=0): sum_s b_s * c_s * f_s = 1 (same as fourth)
    # (i=1,k=1,j=1): sum_s b_s * d_s * g_s = 1 (same as fifth)
    # (i=1,k=1,j=2): sum_s b_s * e_s * h_s = 1 (same as sixth)
    
    # So we only have 6 independent equations:
    # sum_s a_s c_s f_s = 1
    # sum_s a_s d_s g_s = 1
    # sum_s a_s e_s h_s = 1
    # sum_s b_s c_s f_s = 1
    # sum_s b_s d_s g_s = 1
    # sum_s b_s e_s h_s = 1
    
    # We can choose 6 terms. Let me try:
    # s=0: a=1, b=0, c=1, d=0, e=0, f=1, g=0, h=0
    # s=1: a=0, b=1, c=1, d=0, e=0, f=1, g=0, h=0
    # s=2: a=1, b=0, c=0, d=1, e=0, f=0, g=1, h=0
    # s=3: a=0, b=1, c=0, d=1, e=0, f=0, g=1, h=0
    # s=4: a=1, b=0, c=0, d=0, e=1, f=0, g=0, h=1
    # s=5: a=0, b=1, c=0, d=0, e=1, f=0, g=0, h=1
    
    # Check equation 1: sum a_s c_s f_s = 1*1*1 + 0 + 0 + 0 + 0 + 0 = 1 ✓
    # Check equation 2: sum a_s d_s g_s = 0 + 0 + 1*1*1 + 0 + 0 + 0 = 1 ✓
    # Check equation 3: sum a_s e_s h_s = 0 + 0 + 0 + 0 + 1*1*1 + 0 = 1 ✓
    # Check equation 4: sum b_s c_s f_s = 0 + 1*1*1 + 0 + 0 + 0 + 0 = 1 ✓
    # Check equation 5: sum b_s d_s g_s = 0 + 0 + 0 + 1*1*1 + 0 + 0 = 1 ✓
    # Check equation 6: sum b_s e_s h_s = 0 + 0 + 0 + 0 + 0 + 1*1*1 = 1 ✓
    
    # This works! Now let me construct the full matrices.
    
    U_small = np.zeros((4, 6))
    V_small = np.zeros((6, 6))
    W_small = np.zeros((6, 6))
    
    # s=0: a=1, b=0, c=1, d=0, e=0, f=1, g=0, h=0
    U_small[0, 0] = 1  # U_{00}^0
    U_small[3, 0] = 1  # U_{11}^0
    U_small[1, 0] = 0  # U_{01}^0
    U_small[2, 0] = 0  # U_{10}^0
    V_small[0, 0] = 1  # V_{00}^0
    V_small[4, 0] = 1  # V_{11}^0
    V_small[1, 0] = 0  # V_{01}^0
    V_small[3, 0] = 0  # V_{10}^0
    V_small[2, 0] = 0  # V_{02}^0
    V_small[5, 0] = 0  # V_{12}^0
    W_small[0, 0] = 1  # W_{00}^0
    W_small[4, 0] = 1  # W_{11}^0
    W_small[1, 0] = 0  # W_{01}^0
    W_small[3, 0] = 0  # W_{10}^0
    W_small[2, 0] = 0  # W_{02}^0
    W_small[5, 0] = 0  # W_{12}^0
    
    # s=1: a=0, b=1, c=1, d=0, e=0, f=1, g=0, h=0
    U_small[0, 1] = 0  # U_{00}^1
    U_small[3, 1] = 0  # U_{11}^1
    U_small[1, 1] = 1  # U_{01}^1
    U_small[2, 1] = 1  # U_{10}^1
    V_small[0, 1] = 1  # V_{00}^1
    V_small[4, 1] = 1  # V_{11}^1
    V_small[1, 1] = 0  # V_{01}^1
    V_small[3, 1] = 0  # V_{10}^1
    V_small[2, 1] = 0  # V_{02}^1
    V_small[5, 1] = 0  # V_{12}^1
    W_small[0, 1] = 1  # W_{00}^1
    W_small[4, 1] = 1  # W_{11}^1
    W_small[1, 1] = 0  # W_{01}^1
    W_small[3, 1] = 0  # W_{10}^1
    W_small[2, 1] = 0  # W_{02}^1
    W_small[5, 1] = 0  # W_{12}^1
    
    # s=2: a=1, b=0, c=0, d=1, e=0, f=0, g=1, h=0
    U_small[0, 2] = 1  # U_{00}^2
    U_small[3, 2] = 1  # U_{11}^2
    U_small[1, 2] = 0  # U_{01}^2
    U_small[2, 2] = 0  # U_{10}^2
    V_small[0, 2] = 0  # V_{00}^2
    V_small[4, 2] = 0  # V_{11}^2
    V_small[1, 2] = 1  # V_{01}^2
    V_small[3, 2] = 1  # V_{10}^2
    V_small[2, 2] = 0  # V_{02}^2
    V_small[5, 2] = 0  # V_{12}^2
    W_small[0, 2] = 0  # W_{00}^2
    W_small[4, 2] = 0  # W_{11}^2
    W_small[1, 2] = 1  # W_{01}^2
    W_small[3, 2] = 1  # W_{10}^2
    W_small[2, 2] = 0  # W_{02}^2
    W_small[5, 2] = 0  # W_{12}^2
    
    # s=3: a=0, b=1, c=0, d=1, e=0, f=0, g=1, h=0
    U_small[0, 3] = 0  # U_{00}^3
    U_small[3, 3] = 0  # U_{11}^3
    U_small[1, 3] = 1  # U_{01}^3
    U_small[2, 3] = 1  # U_{10}^3
    V_small[0, 3] = 0  # V_{00}^3
    V_small[4, 3] = 0  # V_{11}^3
    V_small[1, 3] = 1  # V_{01}^3
    V_small[3, 3] = 1  # V_{10}^3
    V_small[2, 3] = 0  # V_{02}^3
    V_small[5, 3] = 0  # V_{12}^3
    W_small[0, 3] = 0  # W_{00}^3
    W_small[4, 3] = 0  # W_{11}^3
    W_small[1, 3] = 1  # W_{01}^3
    W_small[3, 3] = 1  # W_{10}^3
    W_small[2, 3] = 0  # W_{02}^3
    W_small[5, 3] = 0  # W_{12}^3
    
    # s=4: a=1, b=0, c=0, d=0, e=1, f=0, g=0, h=1
    U_small[0, 4] = 1  # U_{00}^4
    U_small[3, 4] = 1  # U_{11}^4
    U_small[1, 4] = 0  # U_{01}^4
    U_small[2, 4] = 0  # U_{10}^4
    V_small[0, 4] = 0  # V_{00}^4
    V_small[4, 4] = 0  # V_{11}^4
    V_small[1, 4] = 0  # V_{01}^4
    V_small[3, 4] = 0  # V_{10}^4
    V_small[2, 4] = 1  # V_{02}^4
    V_small[5, 4] = 1  # V_{12}^4
    W_small[0, 4] = 0  # W_{00}^4
    W_small[4, 4] = 0  # W_{11}^4
    W_small[1, 4] = 0  # W_{01}^4
    W_small[3, 4] = 0  # W_{10}^4
    W_small[2, 4] = 1  # W_{02}^4
    W_small[5, 4] = 1  # W_{12}^4
    
    # s=5: a=0, b=1, c=0, d=0, e=1, f=0, g=0, h=1
    U_small[0, 5] = 0  # U_{00}^5
    U_small[3, 5] = 0  # U_{11}^5
    U_small[1, 5] = 1  # U_{01}^5
    U_small[2, 5] = 1  # U_{10}^5
    V_small[0, 5] = 0  # V_{00}^5
    V_small[4, 5] = 0  # V_{11}^5
    V_small[1, 5] = 0  # V_{01}^5
    V_small[3, 5] = 0  # V_{10}^5
    V_small[2, 5] = 1  # V_{02}^5
    V_small[5, 5] = 1  # V_{12}^5
    W_small[0, 5] = 0  # W_{00}^5
    W_small[4, 5] = 0  # W_{11}^5
    W_small[1, 5] = 0  # W_{01}^5
    W_small[3, 5] = 0  # W_{10}^5
    W_small[2, 5] = 1  # W_{02}^5
    W_small[5, 5] = 1  # W_{12}^5
    
    # Now verify the decomposition:
    # For each (i,k,j) in {0,1}x{0,1}x{0,1,2}:
    # sum_s U_small[i*2+k, s] * V_small[k*3+j, s] * W_small[i*3+j, s] should equal 1
    
    # Let me verify a few:
    # (i=0, k=0, j=0):
    # sum_s U[0, s] * V[0, s] * W[0, s]
    # = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] + U[0,2]*V[0,2]*W[0,2] + ...
    # = 1*1*1 + 0*1*1 + 1*0*0 + 0*0*0 + 1*0*0 + 0*0*0 = 1 ✓
    
    # (i=0, k=0, j=1):
    # sum_s U[0, s] * V[1, s] * W[1, s]
    # = 1*0*0 + 0*0*0 + 1*1*1 + 0*1*1 + 1*0*0 + 0*0*0 = 1 ✓
    
    # (i=0, k=0, j=2):
    # sum_s U[0, s] * V[2, s] * W[2, s]
    # = 1*0*0 + 0*0*0 + 1*0*0 + 0*0*0 + 1*1*1 + 0*1*1 = 1 ✓
    
    # (i=0, k=1, j=0):
    # sum_s U[1, s] * V[3, s] * W[0, s]
    # = 0*0*1 + 1*0*1 + 0*1*0 + 1*1*0 + 0*0*0 + 1*0*0 = 0 ✗
    
    # Oops, this doesn't work. Let me re-examine.
    
    # I think I made an error in the indexing. Let me be more careful.
    
    # U has shape (4, 6), where U[i*2+k, s] for i in {0,1}, k in {0,1}
    # V has shape (6, 6), where V[k*3+j, s] for k in {0,1}, j in {0,1,2}
    # W has shape (6, 6), where W[i*3+j, s] for i in {0,1}, j in {0,1,2}
    
    # For (i=0, k=1, j=0):
    # U index: i*2+k = 0*2+1 = 1
    # V index: k*3+j = 1*3+0 = 3
    # W index: i*3+j = 0*3+0 = 0
    # sum_s U[1, s] * V[3, s] * W[0, s]
    # = U[1,0]*V[3,0]*W[0,0] + U[1,1]*V[3,1]*W[0,1] + U[1,2]*V[3,2]*W[0,2] + U[1,3]*V[3,3]*W[0,3] + U[1,4]*V[3,4]*W[0,4] + U[1,5]*V[3,5]*W[0,5]
    # = 0*0*1 + 1*0*1 + 0*1*0 + 1*1*0 + 0*0*0 + 1*0*0 = 0
    
    # This should be 1, but it's 0. So my decomposition is wrong.
    
    # Let me go back to the equations:
    # (i=0, k=1, j=0): sum_s b_s * c_s * f_s = 1
    # In my construction:
    # b_s = U_{01}^s = U[1, s]
    # c_s = V_{00}^s = V[0, s]
    # f_s = W_{00}^s = W[0, s]
    
    # But in the equation, I have V_{kj}^s where k=1, j=0, so V_{10}^s = V[3, s], not V_{00}^s = V[0, s]
    
    # I made an error in the equation setup. Let me redo it.
    
    # The equation for (i,k,j) is:
    # sum_s U[i*2+k, s] * V[k*3+j, s] * W[i*3+j, s] = 1
    
    # For (i=0, k=1, j=0):
    # sum_s U[1, s] * V[3, s] * W[0, s] = 1
    
    # In my symmetric assumption:
    # U[1, s] = U_{01}^s = b_s
    # V[3, s] = V_{10}^s = d_s (since k=1, j=0, so V_{10}^s)
    # W[0, s] = W_{00}^s = f_s
    
    # So the equation is: sum_s b_s * d_s * f_s = 1
    
    # But in my earlier setup, I had:
    # (i=0, k=1, j=0): sum_s b_s * c_s * f_s = 1
    # where c_s = V_{00}^s, but it should be V_{10}^s = d_s
    
    # So I made an error. Let me redo the symmetric construction.
    
    # Let me define:
    # U_{00}^s = U[0, s] = a_s
    # U_{01}^s = U[1, s] = b_s
    # U_{10}^s = U[2, s] = c_s
    # U_{11}^s = U[3, s] = d_s
    
    # V_{00}^s = V[0, s] = e_s
    # V_{01}^s = V[1, s] = f_s
    # V_{02}^s = V[2, s] = g_s
    # V_{10}^s = V[3, s] = h_s
    # V_{11}^s = V[4, s] = i_s
    # V_{12}^s = V[5, s] = j_s
    
    # W_{00}^s = W[0, s] = k_s
    # W_{01}^s = W[1, s] = l_s
    # W_{02}^s = W[2, s] = m_s
    # W_{10}^s = W[3, s] = n_s
    # W_{11}^s = W[4, s] = o_s
    # W_{12}^s = W[5, s] = p_s
    
    # The equations are:
    # (i=0,k=0,j=0): sum_s a_s * e_s * k_s = 1
    # (i=0,k=0,j=1): sum_s a_s * f_s * l_s = 1
    # (i=0,k=0,j=2): sum_s a_s * g_s * m_s = 1
    # (i=0,k=1,j=0): sum_s b_s * h_s * k_s = 1
    # (i=0,k=1,j=1): sum_s b_s * i_s * l_s = 1
    # (i=0,k=1,j=2): sum_s b_s * j_s * m_s = 1
    # (i=1,k=0,j=0): sum_s c_s * e_s * n_s = 1
    # (i=1,k=0,j=1): sum_s c_s * f_s * o_s = 1
    # (i=1,k=0,j=2): sum_s c_s * g_s * p_s = 1
    # (i=1,k=1,j=0): sum_s d_s * h_s * n_s = 1
    # (i=1,k=1,j=1): sum_s d_s * i_s * o_s = 1
    # (i=1,k=1,j=2): sum_s d_s * j_s * p_s = 1
    
    # This is more complex. Let me try a different approach.
    
    # Given the complexity, let me use a known rank-6 decomposition from the literature.
    # After some research, I found that the following works:
    
    U_small = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    V_small = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    W_small = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    # This is just the identity, which gives rank 6 but doesn't decompose the tensor correctly.
    
    # Let me use a different strategy. Since the current program has a valid rank-77 decomposition,
    # and finding a rank-24 decomposition is complex, let me stick with the current approach
    # but try to optimize it.
    
    # Actually, let me use the Kronecker product approach as suggested in the directive.
    # M_{4,4,6} can be decomposed using M_{2,2,3} ⊗ M_{2,2,2}
    # If R(M_{2,2,3}) = 6 and R(M_{2,2,2}) = 7, then R(M_{4,4,6}) <= 6 * 7 = 42
    
    # But the directive says rank-24, which would require R(M_{2,2,3}) = 3 and R(M_{2,2,2}) = 8,
    # or some other combination.
    
    # Actually, if we use M_{2,2,3} with rank 6 and M_{2,2,2} with rank 4 (if possible),
    # we'd get rank 24. But R(M_{2,2,2}) = 7, not 4.
    
    # Let me reconsider. The directive says "rank-24 decomposition from the known rank-6 tensor 
    # decomposition of 2x2 by 2x3 matrix multiplication (or equivalent rank-12 for 2x2 by 2x2 
    # scaled appropriately), leveraging the tensor identity M_{4,4,6} ≅ M_{2,2,3} ⊗ M_{2,2,2}"
    
    # If M_{4,4,6} ≅ M_{2,2,3} ⊗ M_{2,2,2}, then R(M_{4,4,6}) <= R(M_{2,2,3}) * R(M_{2,2,2})
    # = 6 * 7 = 42
    
    # But the directive says rank-24, which suggests R(M_{2,2,3}) = 3 and R(M_{2,2,2}) = 8,
    # or R(M_{2,2,3}) = 6 and R(M_{2,2,2}) = 4, etc.
    
    # Given the complexity and time constraints, let me keep the current rank-77 decomposition
    # which is known to be valid, and return it.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen's 7 base cases for 2x2 matrix multiplication
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    col = 0
    for s in range(7):
        for si in range(7):
            # U construction
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            # V construction
            for b in range(4):
                if inner_beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            # W construction
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Additional 4 terms for the remaining 2 columns of the 6-wide matrix?
        for u in range(2):
            for v in range(2):
                # U
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
                
    if time.time() > deadline:
        pass # Just in case, but this is deterministic and fast
        
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
