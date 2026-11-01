import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin

    n, m, p = DIMENSIONS

    # Fallback: Schoolbook (Rank 32)
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

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Target Rank 28 Construction
    # The 2x4x4 matrix multiplication tensor can be decomposed into 4 independent
    # 2x2x2 subproblems if we partition the middle dimension (4) into two halves (2+2).
    # A is 2x4, B is 4x4, C is 2x4.
    # Partition B's rows into B_top (4x2) and B_bottom (4x2)? No.
    # Let's look at the indices:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # i in 0..1, j in 0..3, k in 0..3.
    #
    # We can split the sum over j into j=0,1 and j=2,3.
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2] ?? No.
    # C[i, k] = sum_{j=0}^{1} A[i, j] B[j, k] + sum_{j=2}^{3} A[i, j] B[j, k]
    #
    # This doesn't separate cleanly into independent 2x2x2 blocks for Strassen directly
    # because the A and B parts are interleaved in the sum for a single C column.
    #
    # However, there is a known rank 28 decomposition for 2x4x4.
    # It is constructed by viewing the multiplication as a block matrix multiplication
    # where one dimension is split.
    #
    # Actually, a simpler and more robust approach to get Rank 28 is to use the
    # fact that the rank of M_{2,4,4} is at most 4 * rank(M_{2,2,2}) = 4 * 7 = 28.
    # This works if we can decompose the problem into 4 independent 2x2x2 multiplications.
    #
    # Let's check if 2x4x4 decomposes into 4 independent 2x2x2s.
    # A is 2x4. B is 4x4. C is 2x4.
    # If we split A into A1 (2x2) and A2 (2x2).
    # If we split B into B11, B12, B21, B22 (each 2x2).
    # C = A1 * [B11 B12] + A2 * [B21 B22]
    #    = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    #
    # This results in 4 distinct 2x2 matrix multiplications:
    # 1. A1 * B11 -> contributes to C[:, 0:2]
    # 2. A1 * B12 -> contributes to C[:, 2:4]
    # 3. A2 * B21 -> contributes to C[:, 0:2]
    # 4. A2 * B22 -> contributes to C[:, 2:4]
    #
    # Wait, C[:, 0:2] is the sum of two 2x2 products.
    # C[:, 2:4] is the sum of two 2x2 products.
    #
    # The tensor T is linear. The rank of a sum of tensors is not simply the sum of ranks
    # unless they are on independent supports.
    # Here, the outputs for C[:, 0:2] are shared between subproblem 1 and 3.
    # The outputs for C[:, 2:4] are shared between subproblem 2 and 4.
    #
    # However, the *inputs* are disjoint?
    # Subproblem 1 uses A1, B11.
    # Subproblem 2 uses A1, B12.
    # Subproblem 3 uses A2, B21.
    # Subproblem 4 uses A2, B22.
    #
    # The U factors (for A) would be:
    # For SP1: U_A1
    # For SP2: U_A1
    # For SP3: U_A2
    # For SP4: U_A2
    #
    # The V factors (for B) would be:
    # For SP1: V_B11
    # For SP2: V_B12
    # For SP3: V_B21
    # For SP4: V_B22
    #
    # The W factors (for C) would be:
    # For SP1: W_C1 (maps to C[:, 0:2])
    # For SP2: W_C2 (maps to C[:, 2:4])
    # For SP3: W_C3 (maps to C[:, 0:2])
    # For SP4: W_C4 (maps to C[:, 2:4])
    #
    # Since the W factors for SP1 and SP3 act on the same output space C[:, 0:2],
    # and SP2 and SP4 act on C[:, 2:4], we can just sum their contributions.
    #
    # Total Rank = Rank(SP1) + Rank(SP2) + Rank(SP3) + Rank(SP4)
    # If each is Rank 7 (Strassen), Total Rank = 28.
    #
    # This construction is valid because the bilinear form is linear in the outputs.
    # C_total = C1 + C3 for the first half, and C2 + C4 for the second half.
    # The decomposition T = T1 + T2 + T3 + T4 holds if the supports are compatible.
    # Specifically, T1[i, j, k] is non-zero only for j in 0..1 (B11 rows) and k in 0..1 (C1 cols).
    # T3[i, j, k] is non-zero only for j in 2..3 (B21 rows) and k in 0..1 (C3 cols).
    # Since j ranges are disjoint (0-1 vs 2-3), the terms do not interfere in the summation
    # over j for a fixed i, k.
    # Wait, the tensor definition is T[i, j, k] = 1 if A[i,j]B[j,k] contributes to C[i,k].
    #
    # Let's verify the disjointness.
    # For a fixed output index k (in C), which is col_c * 2 + row_c.
    # If col_c is 0 or 1, it belongs to the first half of C columns.
    # The contribution to C[i, col_c] comes from sum_j A[i, j] B[j, col_c].
    # This sum is over j=0..3.
    # We can split this sum into j=0..1 and j=2..3.
    # Term 1: sum_{j=0}^1 A[i, j] B[j, col_c]. This corresponds to A1 * B11 (if col_c < 2) or A1 * B12 (if col_c >= 2).
    # Actually, B is 4x4.
    # B[j, col_c].
    # If col_c is 0 or 1, we are looking at columns 0,1 of B.
    # The rows j are 0,1 (top half of B) and 2,3 (bottom half of B).
    #
    # So, for C[:, 0:2] (cols 0,1):
    # C[i, 0:2] = A[i, 0:2] * B[0:2, 0:2] + A[i, 2:4] * B[2:4, 0:2]
    # This is the sum of two 2x2 matrix multiplications.
    # Let M1 = A1 * B11 and M3 = A2 * B21.
    # C[:, 0:2] = M1 + M3.
    #
    # Similarly for C[:, 2:4] (cols 2,3):
    # C[i, 2:4] = A[i, 0:2] * B[0:2, 2:4] + A[i, 2:4] * B[2:4, 2:4]
    # Let M2 = A1 * B12 and M4 = A2 * B22.
    # C[:, 2:4] = M2 + M4.
    #
    # The tensor T can be written as the sum of 4 tensors T1, T2, T3, T4 corresponding to these 4 products.
    # T1 corresponds to A1 * B11 -> C1 (which is C[:, 0:2]).
    # T3 corresponds to A2 * B21 -> C3 (which is C[:, 0:2]).
    # T2 corresponds to A1 * B12 -> C2 (which is C[:, 2:4]).
    # T4 corresponds to A2 * B22 -> C4 (which is C[:, 2:4]).
    #
    # Since T = T1 + T2 + T3 + T4, and each T_s is a 2x2x2 multiplication tensor (embedded in the larger space),
    # rank(T) <= sum rank(T_s) = 7 + 7 + 7 + 7 = 28.
    #
    # We can construct the factors U, V, W for the rank 28 decomposition by concatenating the factors
    # of the 4 Strassen decompositions.
    
    rank_28 = 28
    U_28 = np.zeros((8, rank_28), dtype=float)
    V_28 = np.zeros((16, rank_28), dtype=float)
    W_28 = np.zeros((8, rank_28), dtype=float)
    
    # Strassen 2x2x2 factors
    # We need U_s, V_s, W_s such that sum_s U_s[i,s] V_s[j,s] W_s[k,s] = T_2x2x2[i,j,k]
    # T_2x2x2[i,j,k] = 1 if i,j,k compatible.
    
    # Standard Strassen factors for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # i = a(e+h) - (c+d)(g-h)
    # j = b(f+h)
    # k = a(d-f) + c(e+d)
    # l = (a+b)(h-g) - d(g+f)
    #
    # Let's use the standard integer Strassen decomposition.
    # Inputs: a,b,c,d (A), e,f,g,h (B).
    # Outputs: i,j,k,l (C).
    # Note: The task uses flattened indices.
    # A flat: a=0, b=1, c=2, d=3.
    # B flat: e=0, f=1, g=2, h=3.
    # C flat: i=0, j=1, k=2, l=3. (Wait, C is 2x2, flat index k = col*2 + row.
    # C[0,0]=i -> 0*2+0=0.
    # C[0,1]=j -> 1*2+0=2.
    # C[1,0]=k -> 0*2+1=1.
    # C[1,1]=l -> 1*2+1=3.
    # So C flat indices: i->0, k->1, j->2, l->3.
    
    # Let's define the 7 rank-1 terms for 2x2x2.
    # Term 1: (a) * (e+h) -> contributes to i, k
    # Term 2: (c+d) * (g-h) -> contributes to i, k (with sign)
    # Term 3: (b) * (f+h) -> contributes to j
    # Term 4: (a) * (d-f) -> contributes to k
    # Term 5: (c) * (e+d) -> contributes to k
    # Term 6: (a+b) * (h-g) -> contributes to l
    # Term 7: (d) * (g+f) -> contributes to l (with sign)
    
    # Let's map these to U, V, W vectors.
    # U (4x7) for A inputs (a,b,c,d)
    # V (4x7) for B inputs (e,f,g,h)
    # W (4x7) for C outputs (i,k,j,l) -- Note the order 0,1,2,3 corresponds to i,k,j,l.
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Term 0: a * (e+h)
    U_s[0, 0] = 1  # a
    V_s[0, 0] = 1  # e
    V_s[3, 0] = 1  # h
    W_s[0, 0] = 1  # i
    W_s[1, 0] = 1  # k
    
    # Term 1: (c+d) * (g-h)
    U_s[2, 1] = 1  # c
    U_s[3, 1] = 1  # d
    V_s[2, 1] = 1  # g
    V_s[3, 1] = -1 # -h
    W_s[0, 1] = -1 # -i
    W_s[1, 1] = -1 # -k
    
    # Term 2: b * (f+h)
    U_s[1, 2] = 1  # b
    V_s[1, 2] = 1  # f
    V_s[3, 2] = 1  # h
    W_s[2, 2] = 1  # j
    
    # Term 3: a * (d-f)
    U_s[0, 3] = 1  # a
    V_s[1, 3] = -1 # -f
    V_s[3, 3] = 1  # d? Wait, d is in A.
    # Correction: The term is a * (d - f). d is from A, f is from B.
    # This term involves A and B inputs mixed in the factor?
    # No, Strassen terms are products of linear forms in A and linear forms in B.
    # a is linear in A. (d-f) is NOT linear in B only. d is in A.
    # My Strassen formula recall was slightly off for the factorization form.
    #
    # Correct Strassen factorization:
    # m1 = a(e+h)
    # m2 = (c+d)(g-h)
    # m3 = (a+b)h
    # m4 = d(g-a)
    # m5 = (e+f)a
    # m6 = (g-h)(b+d)
    # m7 = (e-f)(a+c)
    #
    # Let's use a verified set of factors for 2x2x2.
    # I will generate them programmatically to ensure correctness.
    
    # Build T_2x2x2
    T_2x2x2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A[i,j] * B[j,k] -> C[i,k]
                # Flat A: i*2+j
                # Flat B: j*2+k
                # Flat C: k*2+i
                T_2x2x2[i*2+j, j*2+k, k*2+i] = 1
                
    # We want U (4x7), V (4x7), W (4x7) such that sum_s U[:,s] V[:,s] W[:,s] = T
    # This is a rank decomposition problem.
    # Since we know rank is 7, we can pick any 7 linearly independent "slices" or use a known algorithm.
    # Easier: Use the known integer Strassen factors.
    
    # Here is a known correct set for 2x2x2 (a,b,c,d) * (e,f,g,h) -> (i,j,k,l)
    # where C[0,0]=i, C[0,1]=j, C[1,0]=k, C[1,1]=l.
    # Flat C: i=0, j=2, k=1, l=3.
    
    # Let's hardcode a verified Strassen decomposition.
    # Reference: "A 2x2x2 matrix multiplication algorithm with 7 multiplications"
    
    # Inputs A: a(0), b(1), c(2), d(3)
    # Inputs B: e(0), f(1), g(2), h(3)
    # Outputs C: i(0), k(1), j(2), l(3)
    
    # Term 0: a * (e+h)
    # Term 1: (c+d) * (g-h)
    # Term 2: b * f
    # Term 3: a * d
    # Term 4: c * e
    # Term 5: (a+b) * (h-g)
    # Term 6: (c+d) * (e+f)
    
    # Let's check if this set works.
    # i = a(e+h) - (c+d)(g-h)
    # k = a(d-f) + c(e+d) ... wait.
    
    # I will use a more robust method: Solve for W given U and V.
    # Choose U and V to be the first 7 columns of a random matrix? No, must be structured.
    # Actually, we can just use the schoolbook rank 8 for the 2x2x2 subproblems?
    # Rank 8 * 4 = 32. That's the fallback.
    # We need Rank 7 for the subproblems to get 28.
    
    # Let's try to find a rank 7 decomposition for 2x2x2 using a simple search or known factors.
    # Known factors (from literature):
    # U (4x7):
    # [1, 0, 1, 1, 0, 1, 1]
    # [0, 0, 0, 1, 1, 1, 0]
    # [0, 1, 0, 0, 1, 0, 1]
    # [1, 1, 0, 1, 0, 0, 1]
    #
    # V (4x7):
    # [1, 0, 0, 0, 1, 0, 1]
    # [0, 1, 0, 1, 0, 1, 0]
    # [0, 0, 1, 0, 1, 0, 1]
    # [1, 1, 0, 1, 0, 1, 0]
    #
    # W (4x7):
    # [1, -1, 0, 0, 0, 0, 0]
    # [1, -1, 0, 0, 0, 0, 0]  # Wait, C indices are i,k,j,l.
    
    # Let's just construct the 2x2x2 decomposition by solving the linear system
    # with a specific U, V pattern that is known to work.
    # Or, simpler: Use the fact that we can verify the final 28-rank decomposition directly.
    # If we construct U_28, V_28, W_28 by concatenating 4 blocks of (U_s, V_s, W_s),
    # and if (U_s, V_s, W_s) is a valid 2x2x2 decomposition, then the whole thing is valid.
    
    # Let's find a valid 2x2x2 decomposition.
    # I will use a small random search for integer solutions for 2x2x2 rank 7.
    # Or better, use the explicit factors from Strassen's paper.
    
    # Explicit Strassen Factors (a,b,c,d) * (e,f,g,h)
    # m1 = a(e+h)
    # m2 = (c+d)(g-h)
    # m3 = (a+b)h
    # m4 = d(g-a)
    # m5 = (e+f)a
    # m6 = (g-h)(b+d)
    # m7 = (e-f)(a+c)
    
    # Let's map these to U, V, W.
    # m1: U=[1,0,0,0], V=[1,0,0,1], W=[1,1,0,0] (i,k)
    # m2: U=[0,0,1,1], V=[0,0,1,-1], W=[-1,-1,0,0] (-i,-k)
    # m3: U=[1,1,0,0], V=[0,0,0,1], W=[0,0,1,0] (j)
    # m4: U=[0,0,0,1], V=[0,0,1,0], W=[0,1,0,0] (k) ... wait, d(g-a) = dg - da.
    #      This term has 'a' in B? No, a is in A.
    #      The factorization requires linear forms in A and linear forms in B separately.
    #      m4 = d(g-a) is not separable as (linear in A) * (linear in B) because a is in A and g is in B?
    #      No, a is in A, g is in B. So d*g is separable. d*a is not (both in A).
    #      So m4 as written is NOT a valid rank-1 term in the tensor decomposition sense.
    #      Strassen's algorithm uses 7 multiplications, but the linear combinations are formed
    #      from the inputs. The "multiplications" are of the linear forms.
    #      So the rank-1 terms are indeed (linear in A) * (linear in B).
    #
    # Let's re-verify the separability of Strassen terms.
    # m1 = a(e+h). A-form: a. B-form: e+h. OK.
    # m2 = (c+d)(g-h). A-form: c+d. B-form: g-h. OK.
    # m3 = (a+b)h. A-form: a+b. B-form: h. OK.
    # m4 = d(g-a). A-form: d. B-form: g-a. a is in A! So B-form contains an A variable. NOT OK.
    #
    # Ah, the standard Strassen algorithm defines:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)h
    # p3 = a(e+f)
    # p4 = d(g+f)
    # p5 = (a+b)h
    # p6 = (c+d)(g-h)
    # p7 = (a-b)(e+h)
    #
    # Let's check p1: (a+d) is in A. (e+h) is in B. OK.
    # p2: (c+d) in A. h in B. OK.
    # p3: a in A. (e+f) in B. OK.
    # p4: d in A. (g+f) in B. OK.
    # p5: (a+b) in A. h in B. OK.
    # p6: (c+d) in A. (g-h) in B. OK.
    # p7: (a-b) in A. (e+h) in B. OK.
    #
    # These are the 7 terms.
    # Now we need to express C in terms of p1..p7.
    # C[0,0] = p1 - p4 + p5 - p7
    # C[0,1] = p3 + p4
    # C[1,0] = p2 + p6
    # C[1,1] = p1 - p2 + p3 - p5
    #
    # Let's verify C[0,0]:
    # p1 - p4 + p5 - p7
    # = (a+d)(e+h) - d(g+f) + (a+b)h - (a-b)(e+h)
    # = ae + ah + de + dh - dg - df + ah + bh - (ae + ah - be - bh)
    # = ae + ah + de + dh - dg - df + ah + bh - ae - ah + be + bh
    # = (ah - ah + ah - ah) + (ae - ae) + de + dh - dg - df + bh + be + bh
    # = de + dh - dg - df + be + 2bh ??
    # This doesn't look like ae + bg.
    #
    # Let's use a trusted source for the factors.
    # I will use the factors from the Wikipedia article on Strassen's algorithm.
    # m1 = a(e+h)
    # m2 = (c+d)(g-h)
    # m3 = (a+b)h
    # m4 = d(g-a)
    # m5 = (e+f)a
    # m6 = (g-h)(b+d)
    # m7 = (e-f)(a+c)
    #
    # Wait, m4 = d(g-a) is not separable.
    # Wikipedia says:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)h
    # m3 = a(e + f)
    # m4 = d(g + f)
    # m5 = (a + b)h
    # m6 = (c + d)(g - h)
    # m7 = (a - b)(e + h)
    #
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m4
    # C21 = m2 + m6
    # C22 = m1 - m2 + m3 - m5
    #
    # Let's check C11 = m1 - m4 + m5 - m7
    # m1 = ae + ah + de + dh
    # m4 = dg + df
    # m5 = ah + bh
    # m7 = ae + ah - be - bh
    # Sum:
    # (ae + ah + de + dh) - (dg + df) + (ah + bh) - (ae + ah - be - bh)
    # = ae + ah + de + dh - dg - df + ah + bh - ae - ah + be + bh
    # = (ae - ae) + (ah + ah - ah) + de + dh - dg - df + (bh + bh) + be
    # = ah + de + dh - dg - df + 2bh + be
    # This is not ae + bg.
    #
    # There is a mistake in my manual verification or the factors.
    # Let's try a different set.
    #
    # Actually, I can just use the schoolbook rank 8 for the 2x2x2 blocks,
    # which gives rank 32 total.
    # But I want rank 28.
    #
    # Let's use a numerical solver to find the 2x2x2 rank 7 factors.
    # I will fix U and V to be random matrices with entries in {-1, 0, 1}
    # and solve for W. If the solution is integer/half-integer, it's valid.
    #
    # Given time constraints, I will implement the rank 28 construction
    # using the 4-block Strassen approach, but I will verify the 2x2x2 factors
    # numerically first.
    
    # Generate 2x2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Try a few random integer patterns for U and V
    found = False
    for _ in range(100):
        # Random U, V with small integers
        U_try = np.random.randint(-2, 3, (4, 7))
        V_try = np.random.randint(-2, 3, (4, 7))
        
        # Build system for W
        # T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
        # For each (i,j), we have a linear system for W[:,s]
        # But W is shared across i,j.
        # We can vectorize:
        # M (64x28) * W_vec (28) = T_vec (64)
        
        M = np.zeros((64, 28))
        T_vec = np.zeros(64)
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    eq_idx = i * 16 + j * 4 + k
                    T_vec[eq_idx] = T_2x2x2[i, j, k]
                    for s in range(7):
                        M[eq_idx, k * 7 + s] = U_try[i, s] * V_try[j, s]
        
        W_vec, residuals, rank_M, sv = np.linalg.lstsq(M, T_vec, rcond=None)
        W_try = W_vec.reshape(4, 7)
        
        # Check if exact
        error = np.max(np.abs(M @ W_vec - T_vec))
        if error < 1e-9:
            # Check if W is close to integer/half-integer
            W_rounded = np.round(W_try * 2) / 2
            if np.max(np.abs(W_try - W_rounded)) < 1e-6:
                U_s = U_try
                V_s = V_try
                W_s = W_rounded
                found = True
                break
                
    if not found:
        # Fallback to schoolbook for 2x2x2 (Rank 8)
        # This would make total rank 32.
        # But we want 28.
        # Let's use a known valid set if random search fails.
        # For now, if not found, we return schoolbook rank 32.
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Construct full 2x4x4 decomposition
    # 4 subproblems:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # 2. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # 3. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    subproblems = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 0, 0),
        (2, 2, 2, 2)
    ]
    
    for sp_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(subproblems):
        for s in range(7):
            r = sp_idx * 7 + s
            
            # U: A factors
            # Local A is 2x2. Global A is 2x4.
            # Local A (row, col) -> Global A (row, a_col_off + col)
            for local_row in range(2):
                for local_col in range(2):
                    global_col = a_col_off + local_col
                    i_flat = local_row * 4 + global_col
                    local_i = local_row * 2 + local_col
                    U_28[i_flat, r] = U_s[local_i, s]
                    
            # V: B factors
            # Local B is 2x2. Global B is 4x4.
            # Local B (row, col) -> Global B (b_row_off + row, b_col_off + col)
            for local_row in range(2):
                for local_col in range(2):
                    global_row = b_row_off + local_row
                    global_col = b_col_off + local_col
                    j_flat = global_row * 4 + global_col
                    local_j = local_row * 2 + local_col
                    V_28[j_flat, r] = V_s[local_j, s]
                    
            # W: C factors
            # Local C is 2x2. Global C is 2x4.
            # Local C (row, col) -> Global C (row, c_col_off + col)
            # Note: C flat index is col * 2 + row.
            for local_row in range(2):
                for local_col in range(2):
                    global_col = c_col_off + local_col
                    k_flat = global_col * 2 + local_row
                    local_k = local_col * 2 + local_row
                    W_28[k_flat, r] = W_s[local_k, s]

    # Verify the decomposition
    T_full = np.zeros((8, 16, 8), dtype=float)
    for row_a in range(2):
        for col_a in range(4):
            i = row_a * 4 + col_a
            for row_b in range(4):
                for col_b in range(4):
                    j = row_b * 4 + col_b
                    for row_c in range(2):
                        for col_c in range(4):
                            k = col_c * 2 + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T_full[i, j, k] = 1
    
    max_error = 0.0
    for i in range(8):
        for j in range(16):
            for k in range(8):
                val = np.sum(U_28[i, :] * V_28[j, :] * W_28[k, :])
                err = abs(val - T_full[i, j, k])
                if err > max_error:
                    max_error = err
    
    if max_error < 1e-9:
        best_U, best_V, best_W = U_28, V_28, W_28
        best_rank = 28
    else:
        # Fall back to schoolbook
        pass

    if time.time() - start_time > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
