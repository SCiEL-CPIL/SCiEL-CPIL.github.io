import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(U_vec, V_vec, W_vec):
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)

    # Schoolbook decomposition: rank = n*m*p = 96
    # This is the baseline. We'll try to improve with Strassen.
    
    # Strassen 2x2x2 rank-7 decomposition
    # U entries: [u00, u01, u10, u11]
    # V entries: [v00, v01, v10, v11]
    # W entries: [w00, w01, w10, w11]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 1, 0]),
        ([0, 0, 0, 1], [1, 1, 0, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([1, 0, -1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1]),
    ]

    # Strategy:
    # Decompose the 3x4x8 problem.
    # We can view the 4 columns of A as two pairs: {0,1} and {2,3}.
    # We can view the 8 rows of B as four pairs: {0,1}, {2,3}, {4,5}, {6,7}.
    # For each pair of rows of A (i0, i1) and each pair of columns of A (j0, j1)
    # and each pair of rows of B (k0, k1), we have a 2x2x2 subproblem.
    #
    # However, the rows of A are 3, not a power of 2.
    # Let's group the rows of A as:
    # Group 1: Rows {0, 1}
    # Group 2: Row {2}
    #
    # For Group 1 (2 rows), we have 2x4x8.
    # We can further split the 4 columns of A into {0,1} and {2,3}.
    # And the 8 rows of B into 4 pairs.
    # So for Group 1, we have 2 (j-pairs) * 4 (k-pairs) = 8 subproblems of size 2x2x2.
    # Each 2x2x2 uses rank 7 (Strassen). Total rank for Group 1 = 8 * 7 = 56.
    #
    # For Group 2 (1 row), we have 1x4x8.
    # We can split the 4 columns of A into {0,1} and {2,3}.
    # And the 8 rows of B into 4 pairs.
    # So for Group 2, we have 2 (j-pairs) * 4 (k-pairs) = 8 subproblems of size 1x2x2.
    # A 1x2x2 problem: C_k = sum_j A_j * B_{j,k}.
    # This is equivalent to two independent 1x1x2 problems? No.
    # Let's check the rank of 1x2x2.
    # T[(j), (j,k), (k)] = 1.
    # Inputs: j=0,1. Outputs: k=0,1.
    # C_0 = A_0 B_{0,0} + A_1 B_{1,0}
    # C_1 = A_0 B_{0,1} + A_1 B_{1,1}
    # This is the same as matrix multiplication of 1x2 by 2x2.
    # Schoolbook rank is 1*2*2 = 4.
    # Can we do better?
    # Let's try Strassen-like.
    # M1 = A_0 * B_{0,0}
    # M2 = A_1 * B_{1,1}
    # M3 = (A_0+A_1) * (B_{0,1} + B_{1,0}) ... doesn't look right.
    # Actually, 1x2x2 is isomorphic to 2x1x2 (transpose).
    # Let's just use schoolbook for the 1x2x2 parts. Rank 4 each.
    # Total rank for Group 2 = 8 * 4 = 32.
    #
    # Total rank = 56 + 32 = 88.
    # This matches the previous program's logic.
    # The previous program scored 1.0000, which implies it might have failed verification
    # or the seed score is actually 88?
    # "score is the rank". "1.0 equals the seed program".
    # If seed is schoolbook (96), and we produce 88, score should be 96/88 = 1.09.
    # But the current program is labeled "score 1.0000".
    # This suggests the current program is failing verification and falling back to schoolbook (96),
    # resulting in score 96/96 = 1.0.
    # OR the seed is actually 88? No, "seed program" usually refers to the initial naive implementation.
    # Let's look at the "Current Program" code. It has a fallback to schoolbook if error > 1e-6.
    # If the Strassen part has a bug, it falls back to 96.
    # Let's verify the Strassen indexing in the previous code carefully.
    
    # U_vec indices: i*m + j. m=4.
    # Group 1: i in {0,1}.
    # j-pair 0: j in {0,1}.
    #   i=0, j=0 -> idx 0
    #   i=0, j=1 -> idx 1
    #   i=1, j=0 -> idx 4
    #   i=1, j=1 -> idx 5
    #   Previous code:
    #   U_vec[0] = u_c[0]
    #   U_vec[1] = u_c[1]
    #   U_vec[4] = u_c[2]
    #   U_vec[5] = u_c[3]
    #   This maps [u00, u01, u10, u11] to [A00, A01, A10, A11]. Correct.
    
    # V_vec indices: j*p + k. p=8.
    # k-pair 0: k in {0,1}.
    #   j=0, k=0 -> idx 0
    #   j=0, k=1 -> idx 1
    #   j=1, k=0 -> idx 8
    #   j=1, k=1 -> idx 9
    #   Previous code:
    #   V_vec[k0] = v_c[0]   # j=0, k=k0
    #   V_vec[k1] = v_c[1]   # j=0, k=k1
    #   V_vec[8 + k0] = v_c[2] # j=1, k=k0
    #   V_vec[8 + k1] = v_c[3] # j=1, k=k1
    #   This maps [v00, v01, v10, v11] to [B00, B01, B10, B11]. Correct.
    
    # W_vec indices: k*n + i. n=3.
    #   k=0, i=0 -> idx 0
    #   k=1, i=0 -> idx 3
    #   k=0, i=1 -> idx 1
    #   k=1, i=1 -> idx 4
    #   Previous code:
    #   W_vec[3*k0] = w_c[0]   # k=k0, i=0
    #   W_vec[3*k1] = w_c[1]   # k=k1, i=0
    #   W_vec[3*k0 + 1] = w_c[2] # k=k0, i=1
    #   W_vec[3*k1 + 1] = w_c[3] # k=k1, i=1
    #   This maps [w00, w01, w10, w11] to [C00, C01, C10, C11]. Correct.
    
    # So the Strassen part seems correct. Why did it score 1.0?
    # Maybe the verification failed due to floating point?
    # Strassen coefficients are integers. Products are integers.
    # Sum of integers is integer.
    # Target is 1.0.
    # np.dot on floats should be exact for small integers.
    # Unless... the verification loop is slow and times out?
    # Verification is O(n*m*p * rank) = 96 * 96 * 88 ~ 800k ops. Fast.
    
    # Wait, the "Current Program" in the prompt IS the one that scored 1.0.
    # And it has the Strassen code.
    # If it scored 1.0, it means it produced rank 96 (fallback) or the seed rank is 88?
    # If the seed rank is 88, then 1.0 means we matched the seed.
    # But the task says "score is the rank". Lower is better.
    # "1.0 equals the seed program".
    # If seed is schoolbook (96), and we output 88, score = 96/88 > 1.
    # If we output 96, score = 1.
    # The fact that the current program (which has Strassen) scores 1.0 strongly suggests
    # it is falling back to schoolbook.
    
    # Let's re-examine the fallback condition.
    # `if error > 1e-6:`
    # Is there a bug in the Strassen implementation?
    
    # Let's check the Strassen terms again.
    # Standard Strassen for C = A*B (2x2):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let's map the previous `strassen_terms` to these.
    # Term 0: U=[1,0,0,1] (A00+A11), V=[1,0,0,1] (B00+B11), W=[1,0,0,1] (C00+C11?)
    #   This looks like M1 contributes to C00 and C11?
    #   In the tensor decomposition, each term is a rank-1 tensor.
    #   If U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1], it adds 1 to C00, C01, C10, C11?
    #   No, W indices: w00=1, w01=0, w10=0, w11=1.
    #   So it adds to C00 and C11.
    #   M1 = (A00+A11)(B00+B11).
    #   In C00: +M1. In C11: +M1.
    #   So W should be [1, 0, 0, 1]. This matches.
    
    # Term 1: U=[0,0,1,1] (A10+A11), V=[1,0,0,0] (B00), W=[0,0,1,-1] (C10 - C11)
    #   M2 = (A10+A11)B00.
    #   In C10: +M2. In C11: -M2.
    #   So W should be [0, 0, 1, -1]. This matches.
    
    # Term 2: U=[1,0,0,0] (A00), V=[0,0,1,-1] (B01-B11), W=[0,1,1,0] (C01+C10)
    #   M3 = A00(B01-B11).
    #   In C01: +M3.
    #   In C10: +M3? Wait.
    #   C10 = M2 + M4. M3 is not in C10.
    #   C01 = M3 + M5.
    #   So M3 should only contribute to C01.
    #   But W here is [0,1,1,0], which means it contributes to C01 AND C10.
    #   This is WRONG.
    #   Let's re-check Strassen formulas.
    #   C01 = M3 + M5.
    #   C10 = M2 + M4.
    #   So M3 should have W = [0, 1, 0, 0].
    #   The previous code has W = [0, 1, 1, 0].
    #   This is a BUG.
    
    # Let's check Term 3: U=[0,0,0,1] (A11), V=[1,1,0,-1] (B00+B01-B11), W=[1,0,1,0] (C00+C10)
    #   M4 = A11(B10-B00).
    #   Wait, V is [1,1,0,-1]. This is B00+B01-B11. That's not B10-B00.
    #   M4 = A11 * (B10 - B00). V should be [ -1, 0, 1, 0 ].
    #   The previous code has V = [1, 1, 0, -1].
    #   This is also WRONG.
    
    # It seems the `strassen_terms` in the previous code were not standard Strassen
    # or were incorrectly transcribed.
    
    # I will implement the correct Strassen decomposition.
    
    # Correct Strassen terms (U, V, W):
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
    # M3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # M4: U=[0,0,0,1], V=[-1,0,1,0], W=[0,0,1,0]  # Note: C10 gets M4
    # M5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    # M6: U=[1,-1,0,0], V=[1,1,0,0], W=[0,0,0,1]
    # M7: U=[0,1,-1,0], V=[0,0,1,1], W=[1,0,0,1]
    
    # Let's verify C00: M1 - M5 + M7?
    # C00 = M1 + M4 - M5 + M7
    # My M4 W is [0,0,1,0] (C10). It doesn't contribute to C00.
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-derive the W vectors based on this.
    # M1: W=[1,0,0,1]
    # M2: W=[0,0,1,-1]
    # M3: W=[0,1,0,0]
    # M4: W=[1,0,1,0]  # Contributes to C00 and C10?
    #    Wait. C00 = ... + M4 ...
    #    C10 = ... + M4 ...
    #    So M4 W should be [1,0,1,0].
    # M5: W=[0,1,0,0]  # Contributes to C01?
    #    C01 = M3 + M5.
    #    C00 = ... - M5 ...
    #    So M5 W should be [-1, 1, 0, 0].
    # M6: W=[0,0,0,1]
    # M7: W=[1,0,0,1]
    
    # Let's check M4 again.
    # M4 = A11 * (B10 - B00).
    # V = [-1, 0, 1, 0].
    # C00 gets +M4. C10 gets +M4.
    # So W4 = [1, 0, 1, 0].
    
    # Let's check M5 again.
    # M5 = (A00 + A01) * B11.
    # U = [1, 1, 0, 0].
    # V = [0, 0, 0, 1].
    # C00 gets -M5. C01 gets +M5.
    # So W5 = [-1, 1, 0, 0].
    
    # Let's check M7 again.
    # M7 = (A01 - A11) * (B10 + B11).
    # U = [0, 1, -1, 0].
    # V = [0, 0, 1, 1].
    # C00 gets +M7. C11 gets +M7.
    # So W7 = [1, 0, 0, 1].
    
    # Let's check M3 again.
    # M3 = A00 * (B01 - B11).
    # U = [1, 0, 0, 0].
    # V = [0, 1, 0, -1].
    # C01 gets +M3. C11 gets +M3.
    # So W3 = [0, 1, 0, 1].
    
    # Let's check M6 again.
    # M6 = (A10 - A00) * (B00 + B01).
    # U = [1, 0, -1, 0] -> [ -1, 0, 1, 0 ]? No, A10-A00 is [ -1, 0, 1, 0 ] if order is A00,A01,A10,A11.
    # My U order is [u00, u01, u10, u11].
    # A10 - A00: u00=-1, u10=1. U=[-1, 0, 1, 0].
    # V = B00 + B01: [1, 1, 0, 0].
    # C11 gets +M6.
    # So W6 = [0, 0, 0, 1].
    
    # Let's check M2 again.
    # M2 = (A10 + A11) * B00.
    # U = [0, 0, 1, 1].
    # V = [1, 0, 0, 0].
    # C10 gets +M2. C11 gets -M2.
    # So W2 = [0, 0, 1, -1].
    
    # Let's check M1 again.
    # M1 = (A00 + A11) * (B00 + B11).
    # U = [1, 0, 0, 1].
    # V = [1, 0, 0, 1].
    # C00 gets +M1. C11 gets +M1.
    # So W1 = [1, 0, 0, 1].
    
    # Summary of Correct Strassen Terms:
    # 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
    # 3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
    # 4: U=[0,0,0,1], V=[-1,0,1,0], W=[1,0,1,0]
    # 5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
    # 6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # 7: U=[0,1,-1,0], V=[0,0,1,1], W=[1,0,0,1]
    
    # Let's double check C00: M1 + M4 - M5 + M7
    # M1 W0=1, M4 W0=1, M5 W0=-1, M7 W0=1. Sum = 1+1-1+1 = 2?
    # Wait. C00 = A00B00 + A01B10.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    #
    # Sum:
    # A00B00: +1 (M1)
    # A00B11: +1 (M1) -1 (M5) = 0
    # A11B00: +1 (M1) -1 (M4) = 0
    # A11B11: +1 (M1) -1 (M7) = 0
    # A11B10: +1 (M4) -1 (M7) = 0
    # A01B10: +1 (M7)
    # A01B11: +1 (M5) +1 (M7) = 2?
    #
    # Wait. C00 should be A00B00 + A01B10.
    # My sum has A01B10 (1) and A01B11 (2).
    # Something is wrong.
    #
    # Let's re-read Strassen.
    # C00 = M1 + M4 - M5 + M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Let's re-sum for C00:
    # M1: A00B00 + A00B11 + A11B00 + A11B11
    # M4: A11B10 - A11B00
    # -M5: -A00B11 - A01B11
    # M7: A01B10 + A01B11 - A11B10 - A11B11
    #
    # Terms:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    #
    # Result: A00B00 + A01B10. Correct!
    
    # So the W vectors are correct.
    # 1: [1,0,0,1]
    # 2: [0,0,1,-1]
    # 3: [0,1,0,1]
    # 4: [1,0,1,0]
    # 5: [-1,1,0,0]
    # 6: [0,0,0,1]
    # 7: [1,0,0,1]
    
    # Now, let's implement this correctly.
    
    strassen_terms_correct = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    # Part 1: Rows {0,1}, j in {0,1}, k in pairs -> 2x2x2 Strassen (4 k-pairs)
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for (u_c, v_c, w_c) in strassen_terms_correct:
            U_vec = [0.0] * (n*m)
            U_vec[0] = float(u_c[0])   # i=0,j=0
            U_vec[1] = float(u_c[1])   # i=0,j=1
            U_vec[4] = float(u_c[2])   # i=1,j=0
            U_vec[5] = float(u_c[3])   # i=1,j=1

            V_vec = [0.0] * (m*p)
            V_vec[k0] = float(v_c[0])       # j=0,k=k0
            V_vec[k1] = float(v_c[1])       # j=0,k=k1
            V_vec[8 + k0] = float(v_c[2])   # j=1,k=k0
            V_vec[8 + k1] = float(v_c[3])   # j=1,k=k1

            W_vec = [0.0] * (p*n)
            W_vec[3*k0] = float(w_c[0])       # k=k0,i=0
            W_vec[3*k1] = float(w_c[1])       # k=k1,i=0
            W_vec[3*k0 + 1] = float(w_c[2])   # k=k0,i=1
            W_vec[3*k1 + 1] = float(w_c[3])   # k=k1,i=1

            add_rank1(U_vec, V_vec, W_vec)

    # Part 2: Rows {0,1}, j in {2,3}, k in pairs -> 2x2x2 Strassen (4 k-pairs)
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for (u_c, v_c, w_c) in strassen_terms_correct:
            U_vec = [0.0] * (n*m)
            U_vec[2] = float(u_c[0])   # i=0,j=2
            U_vec[3] = float(u_c[1])   # i=0,j=3
            U_vec[6] = float(u_c[2])   # i=1,j=2
            U_vec[7] = float(u_c[3])   # i=1,j=3

            V_vec = [0.0] * (m*p)
            V_vec[16 + k0] = float(v_c[0])   # j=2,k=k0
            V_vec[16 + k1] = float(v_c[1])   # j=2,k=k1
            V_vec[24 + k0] = float(v_c[2])   # j=3,k=k0
            V_vec[24 + k1] = float(v_c[3])   # j=3,k=k1

            W_vec = [0.0] * (p*n)
            W_vec[3*k0] = float(w_c[0])
            W_vec[3*k1] = float(w_c[1])
            W_vec[3*k0 + 1] = float(w_c[2])
            W_vec[3*k1 + 1] = float(w_c[3])

            add_rank1(U_vec, V_vec, W_vec)

    # Part 3: Row {2}, all j in {0,1,2,3}, k in {0..7} -> schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U_vec = [0.0] * (n*m)
            U_vec[2*4 + j] = 1.0

            V_vec = [0.0] * (m*p)
            V_vec[j*8 + k] = 1.0

            W_vec = [0.0] * (p*n)
            W_vec[k*3 + 2] = 1.0

            add_rank1(U_vec, V_vec, W_vec)

    rank = len(U_cols)
    
    # Verify
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    error = 0.0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = np.dot(U[a] * V[b], W[c])
                # Determine true value
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                i2 = c % n
                k2 = c // n
                true_val = 0.0
                if i == i2 and j == j2 and k == k2:
                    true_val = 1.0
                error += (val - true_val)**2

    if error > 1e-6:
        # Fallback
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_vec = [0.0] * (n*m)
                    U_vec[i*m+j] = 1.0
                    V_vec = [0.0] * (m*p)
                    V_vec[j*p+k] = 1.0
                    W_vec = [0.0] * (p*n)
                    W_vec[k*n+i] = 1.0
                    add_rank1(U_vec, V_vec, W_vec)
        rank = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
