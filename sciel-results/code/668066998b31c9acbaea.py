"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
import time
import sys

DIMENSIONS = [5, 5, 5]

# Helper to get the 2x2 Strassen decomposition factors
def get_strassen_2x2():
    """
    Returns U, V, W for 2x2 Strassen (rank 7).
    U: 4x7, V: 4x7, W: 4x7
    Rows 0-3 correspond to indices 00, 01, 10, 11.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1 # C00
    W[3][0] += 1 # C11
    
    # M2 = (A10 + A11) B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1 # C01
    
    # M3 = A00 (B01 - B11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[2][2] += 1 # C10
    
    # M4 = A11 (B10 - B00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += 1 # C00
    W[2][3] += 1 # C10
    
    # M5 = (A00 + A01) B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[3][4] += 1 # C11
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[0][5] += -1 # C00
    W[1][5] += 1  # C01
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1  # C00
    W[3][6] += 1  # C11
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    # Track start time for deadline
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    
    def elapsed():
        return time.time() - start_time
    
    def time_exceeded():
        return elapsed() > timeout - 5.0
    
    # Fallback: Schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    try:
        # 1. Initialize 5x5 matrices
        # Rank = 49 (Strassen 4x4) + 61 (Cross terms) = 110
        rank = 110
        U = [[0.0] * rank for _ in range(25)]
        V = [[0.0] * rank for _ in range(25)]
        W = [[0.0] * rank for _ in range(25)]
        
        # 2. Construct Strassen 49
        # Get 2x2 factors
        U2, V2, W2 = get_strassen_2x2()
        
        # We have 7 meta-terms and 7 leaf-terms.
        # U2 is 4x7, where row r corresponds to A[r%2, r//2] ??
        # My U2 definition:
        # Index 0: 00, 1: 01, 2: 10, 3: 11.
        # Let's map index to (row, col):
        # 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1).
        
        for meta in range(7):
            for leaf in range(7):
                t = meta * 7 + leaf
                
                # The A-part of this term is U2[?, meta] * U2[?, leaf]?
                # No.
                # The 4x4 matrix A is composed of 2x2 blocks.
                # A_{block I, J} where I, J in 0..1, each block is 2x2.
                # The meta-strassen operates on the 2x2 matrix of blocks.
                # So, the linear form on A for meta-term 'meta' is defined by U2.
                # But U2[0] is A_{block_00} + A_{block_11}?
                # Actually, in Strassen 2x2, U2[i] gives the coefficient of A_{i_row, i_col} in the linear form.
                # Here, the "elements" of the 2x2 meta-matrix ARE the 2x2 blocks.
                # So, if U2[0][meta] == 1, it means the top-left block (A_{0:2, 0:2}) is involved with coeff 1.
                # If U2[3][meta] == 1, the bottom-right block (A_{2:4, 2:4}) is involved with coeff 1.
                
                # Similarly for the leaf level, we are multiplying the resulting "meta-scalar" (which is actually a 2x2 matrix sum/product?)
                # No, the Kronecker product structure is:
                # Term = (Sum_{I,J} c^meta_{I,J} A_{I,J}) * (Sum_{K,L} d^meta_{K,L} B_{K,L})
                # Where A_{I,J} is a 2x2 block.
                # And then this product is expanded using the 2x2 Strassen on the blocks?
                # No, that's not right.
                
                # Correct view:
                # The multiplication of two 4x4 matrices is the multiplication of two 2x2 matrices of 2x2 matrices.
                # Let $\mathcal{A}_{IJ}$ be the 2x2 block.
                # $\mathcal{C}_{IK} = \sum_J \mathcal{A}_{IJ} \mathcal{B}_{JK}$.
                # We apply Strassen to the 2x2 structure.
                # Term $M_{meta}$ is a linear combination of block products $\mathcal{A}_{IJ} \mathcal{B}_{JK}$.
                # But $\mathcal{A}_{IJ} \mathcal{B}_{JK}$ is a product of two 2x2 matrices.
                # We decompose EACH such 2x2 block product using Strassen (leaf).
                
                # So, for each meta-term `meta`, we have specific block pairs (I,J) and (K,L) involved.
                # Actually, the meta-strassen U2, V2, W2 defines:
                # $U_{meta}$: linear form on the 4 blocks of A.
                # $V_{meta}$: linear form on the 4 blocks of B.
                # $W_{meta}$: linear form on the 4 blocks of C.
                
                # Wait, the standard Kronecker approach:
                # The rank-49 decomposition is the tensor product of the rank-7 decomposition.
                # $T_4 = T_2 \otimes T_2$.
                # The factors are $U_4 = U_2 \otimes U_2$?
                # Yes.
                # $U_4$ has dimensions $16 \times 49$.
                # Entry $(r_4, c_4)$ where $r_4 = r_{meta} \cdot 2 + r_{leaf}$?
                # No, the row index of the 16-dim vector corresponds to the flat index of the 4x4 matrix.
                # Let flat index $R = r \cdot 4 + c$.
                # This can be seen as $(r_{meta}, r_{leaf}, c_{meta}, c_{leaf})$?
                # $r = r_{meta} \cdot 2 + r_{leaf}$.
                # $c = c_{meta} \cdot 2 + c_{leaf}$.
                # So $R$ determines $r_{meta}, r_{leaf}, c_{meta}, c_{leaf}$.
                
                # The entry $U_4[R, T]$ where $T = T_{meta} \cdot 7 + T_{leaf}$.
                # $U_4[R, T] = U_2[r_{meta}, T_{meta}] \times U_2[r_{leaf}, T_{leaf}] \times \delta_{c_{meta}, c_{leaf\_constraint}}?$
                # No, the Kronecker product $U_2 \otimes U_2$ doesn't mix row indices like that simply.
                
                # Let's use the explicit block expansion.
                # For each meta term `m` and leaf term `l`:
                # Identify which 2x2 blocks of A and B are involved in the meta-form.
                # Let $A\_coeffs$ be the coefficients for the 4 blocks of A from U2[?, m].
                # Note: U2[0] is A00, U2[1] is A01, U2[2] is A10, U2[3] is A11.
                # So if U2[0][m] = 1, Block A00 is involved.
                
                a_blocks = [U2[0][m], U2[1][m], U2[2][m], U2[3][m]] # Coeffs for A00, A01, A10, A11
                b_blocks = [V2[0][m], V2[1][m], V2[2][m], V2[3][m]] # Coeffs for B00, B01, B10, B11
                
                # For each pair of blocks (IA, IB) where coeff is non-zero:
                for ib_a in range(4):
                    if a_blocks[ib_a] == 0: continue
                    # ib_a 0: A[0:2, 0:2], 1: A[0:2, 2:4], 2: A[2:4, 0:2], 3: A[2:4, 2:4]
                    # Let block index I, J.
                    I, J = divmod(ib_a, 2)
                    
                    for ib_b in range(4):
                        if b_blocks[ib_b] == 0: continue
                        K, L = divmod(ib_b, 2)
                        
                        # This pair of blocks A[I, J] and B[K, L] contributes to C[I, L] ?
                        # In block multiplication: C_IK = sum J A_IJ B_JK.
                        # So A[I, J] * B[J, K] contributes to C[I, K].
                        # But Strassen M_m is a linear combo of products A_IJ B_KL.
                        # Specifically, M_m = sum I,J,K,L u_A[I,J] v_B[K,L] A_IJ B_KL.
                        # Wait, the structure of Strassen is:
                        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
                        # These products result in C blocks.
                        # A00B00 -> C00. A00B11 -> C01? No.
                        # A is 2x2 blocks: A00 (top-left), A01 (top-right), A10 (bot-left), A11 (bot-right).
                        # B is 2x2 blocks.
                        # C00 = A00B00 + A01B10.
                        # C01 = A00B01 + A01B11.
                        # C10 = A10B00 + A11B10.
                        # C11 = A10B01 + A11B11.
                        
                        # In Strassen, M1 = A00B00 + A00B11 + A11B00 + A11B11.
                        # This sum contributes to C00 and C11.
                        # Specifically, W2[0][0]=1 (C00) and W2[3][0]=1 (C11) for M1.
                        
                        # So, for the leaf level:
                        # We have a product of two 2x2 matrices: P * Q.
                        # P = a_blocks[ib_a] * A[I, J] ? No, P is the 2x2 matrix block itself, scaled?
                        # No, the meta-strassen computes:
                        # $M_{meta} = (\sum_{I,J} u_{IJ} A_{IJ}) (\sum_{K,L} v_{KL} B_{KL})$.
                        # This is the product of two linear combinations of 2x2 blocks.
                        # Let $S_A = \sum u_{IJ} A_{IJ}$ and $S_B = \sum v_{KL} B_{KL}$.
                        # Then $M_{meta} = S_A S_B$.
                        # $S_A$ and $S_B$ are 2x2 matrices.
                        # We multiply them using leaf-strassen.
                        
                        # So, for the term $T = (m, l)$, we are adding:
                        # $u_{meta, ij} v_{meta, kl} \cdot U_{leaf, i'l'} V_{leaf, k'l'} \dots$
                        
                        # This is getting too convoluted to debug on the fly.
                        # I will use the simpler explicit generation:
                        
                        # The 49th term U, V, W entries are determined by:
                        # $U_{49}[r, c, t] = U_{meta}[r\_meta, t\_meta] \cdot U_{leaf}[r\_leaf, t\_leaf] \cdot \delta_{c\_meta, c\_leaf?}$
                        # No, the Kronecker product $U_4 = U_2 \otimes U_2$ works if the indices align.
                        
                        # Let's just use the property:
                        # $U_4$ has size $16 \times 49$.
                        # Row index $R$ in 0..15 corresponds to $r, c$ in 4x4.
                        # $r = R // 4$, $c = R \% 4$.
                        # Decompose $r = r_m \cdot 2 + r_l$, $c = c_m \cdot 2 + c_l$.
                        # Column index $T$ in 0..48 corresponds to $t_m, t_l$.
                        # $T = t_m \cdot 7 + t_l$.
                        
                        # Is $U_4[R, T] = U_2[r_m, c_m, t_m] \times U_2[r_l, c_l, t_l]$?
                        # U2 is defined on 2x2 matrices.
                        # The "row" of U2 in the meta-step corresponds to the block index (I, J).
                        # Let's map U2 index to (I, J).
                        # U2[0] -> (0,0). U2[1] -> (0,1). U2[2] -> (1,0). U2[3] -> (1,1).
                        # So for a given R, we have block I = r//2, J = c//2.
                        # And leaf pos r_l = r%2, c_l = c%2.
                        
                        # The coefficient of $A_{r,c}$ in the $t_m$-th meta-linear-form is $U_2[\text{index}(I,J), t_m]$.
                        # The coefficient of $A_{r_l, c_l}$ within that block?
                        # No, the meta-linear-form applies to the WHOLE block.
                        # But the leaf-strassen applies to the elements of the block.
                        
                        # The correct identity is:
                        # The term $T_{m,l}$ in the decomposition of the 4x4 tensor is:
                        # $\sum_{I,J,K,L} U_2[(I,J), m] V_2[(K,L), m] W_2[(I,L), m] \times$
                        # $\sum_{i,j,k,l \in 0,1} U_2[(i,j), l] V_2[(k,l), l] W_2[(i,l), l] \delta_{J,K} \dots$
                        
                        # I cannot derive this perfectly in 2 minutes.
                        # I will rely on the fact that 110 is the correct bound, and if the Strassen part is buggy, I should fallback.
                        
                        pass # Placeholder for complex logic, skipping to cross terms which are easy
                        # BUT if I skip Strassen, I only have 61 + 96 schoolbook for rest? No.
                        
                        # Let's try a different, safer approach:
                        # Use the known Rank 110 construction from literature "Lam, 1985" or similar.
                        # Since I cannot look it up, I will implement the Schoolbook Rank 125 
                        # BUT I will attempt the 110 and verify it with NumPy in 10s. If it fails, return 125.
                        
                        pass
                        
        # Due to complexity and risk of subtle index errors in manual Kronecker construction,
        # and the hard deadline, I will return the schoolbook rank 125 to ensure validity,
        # UNLESS I can be 100% sure.
        
        # Actually, the prompt asks to "Write a new program... with a stronger algorithm".
        # Returning schoolbook is the same as parent.
        # I will implement the 110 rank carefully.
        
        # Let's fill the 49 Strassen terms using the Kronecker product definition directly on the matrices.
        # $U_{4} = U_{2} \otimes U_{2}$ is NOT correct because U2 is for scalar multiplication.
        # However, if we view the 4x4 multiplication as 2x2 multiplication of 2x2 matrices,
        # and each 2x2 multiplication is rank 7...
        # The total rank is $7 \times 7 = 49$.
        
        # We construct U4, V4, W4 of size 16x49.
        # For t in 0..48:
        #   tm = t // 7, tl = t % 7
        #   For each row R in 0..15:
        #     r = R // 4, c = R % 4
        #     rm = r // 2, cm = c // 2
        #     rl = r % 2, cl = c % 2
        #     # Coefficient in U4
        #     U4[R, t] = U2[rm*2+cm, tm] * U2[rl*2+cl, tl]
        
        # Let's test this logic.
        # Does this correctly expand?
        # $A_{r,c} = A_{rm \cdot 2 + rl, cm \cdot 2 + cl}$.
        # The term $t=(tm, tl)$ involves product of linear forms.
        # This formula assumes the linear form separates into block-index and within-block-index.
        # This is TRUE for Kronecker product decompositions.
        
        U4 = [[0.0]*49 for _ in range(16)]
        V4 = [[0.0]*49 for _ in range(16)]
        W4 = [[0.0]*49 for _ in range(16)]
        
        for t in range(49):
            tm = t // 7
            tl = t % 7
            for R in range(16):
                r = R // 4
                c = R % 4
                rm = r // 2
                cm = c // 2
                rl = r % 2
                cl = c % 2
                idx_meta = rm * 2 + cm
                idx_leaf = rl * 2 + cl
                U4[R][t] = U2[idx_meta][tm] * U2[idx_leaf][tl]
                
            for R in range(16):
                r = R // 4
                c = R % 4
                rm = r // 2
                cm = c // 2
                rl = r % 2
                cl = c % 2
                idx_meta = rm * 2 + cm
                idx_leaf = rl * 2 + cl
                V4[R][t] = V2[idx_meta][tm] * V2[idx_leaf][tl]
                
            for R in range(16):
                r = R // 4
                c = R % 4
                rm = r // 2
                cm = c // 2
                rl = r % 2
                cl = c % 2
                idx_meta = rm * 2 + cm
                idx_leaf = rl * 2 + cl
                W4[R][t] = W2[idx_meta][tm] * W2[idx_leaf][tl]

        # Copy to U, V, W
        for i in range(16):
            for t in range(49):
                U[i][t] = U4[i][t]
                V[i][t] = V4[i][t]
                W[i][t] = W4[i][t]

        # 3. Fill Cross Terms (49 to 109)
        # a = A[0:4, 4], d = B[4, 0:4]
        # Term 49 + i*4 + j: A[i, 4] * B[4, j] -> C[i, j]
        # i in 0..3, j in 0..3
        idx = 49
        for i in range(4):
            for j in range(4):
                # U row: i, col 4 -> index i*5 + 4
                # V row: 4, col j -> index 4*5 + j
                # W row: i, col j -> index i*5 + j
                U[i * 5 + 4][idx] = 1
                V[4 * 5 + j][idx] = 1
                W[i * 5 + j][idx] = 1
                idx += 1

        # A4 * c: A[i, k] * B[k, 4] -> C[i, 4]
        # i in 0..3, k in 0..3
        for i in range(4):
            for k in range(4):
                U[i * 5 + k][idx] = 1
                V[k * 5 + 4][idx] = 1
                W[i * 5 + 4][idx] = 1
                idx += 1

        # a * s: A[i, 4] * B[4, 4] -> C[i, 4]
        for i in range(4):
            U[i * 5 + 4][idx] = 1
            V[4 * 5 + 4][idx] = 1
            W[i * 5 + 4][idx] = 1
            idx += 1

        # u * B4: A[4, k] * B[k, j] -> C[4, j]
        for k in range(4):
            for j in range(4):
                U[4 * 5 + k][idx] = 1
                V[k * 5 + j][idx] = 1
                W[4 * 5 + j][idx] = 1
                idx += 1

        # s * d: A[4, 4] * B[4, j] -> C[4, j]
        for j in range(4):
            U[4 * 5 + 4][idx] = 1
            V[4 * 5 + j][idx] = 1
            W[4 * 5 + j][idx] = 1
            idx += 1

        # u * c: A[4, k] * B[k, 4] -> C[4, 4]
        for k in range(4):
            U[4 * 5 + k][idx] = 1
            V[k * 5 + 4][idx] = 1
            W[4 * 5 + 4][idx] = 1
            idx += 1

        # s * t: A[4, 4] * B[4, 4] -> C[4, 4]
        U[4 * 5 + 4][idx] = 1
        V[4 * 5 + 4][idx] = 1
        W[4 * 5 + 4][idx] = 1
        idx += 1

        # idx should be 110
        if idx != 110:
            return schoolbook()

        # Verification with NumPy
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Tensor reconstruction check for a few random matrices
        np.random.seed(42)
        A_test = np.random.rand(5, 5)
        B_test = np.random.rand(5, 5)
        C_expected = A_test @ B_test
        
        # C[k*n+i, r] = sum_p U[i*m+p, r] V[j*p+k, r] ?
        # Contract: C_{ij} = sum_r U_{i, \cdot, r} V_{\cdot, j, r} ?
        # U shape (25, 110). Row i corresponds to A_{row, col}.
        # V shape (25, 110). Row j corresponds to B_{row, col}.
        # W shape (25, 110). Row k corresponds to C_{row, col}.
        
        C_calc = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                total = 0
                for k in range(5):
                    # Contribution to C[i, j] from A[i, k] * B[k, j]
                    # In our W, the row is k*n + i.
                    # Wait, the task says W[k*n+i, r].
                    # So W is indexed by output (k, i) where k is col of A / row of B?
                    # Standard: C_{ij} = sum_k A_{ik} B_{kj}.
                    # W index: k*n + i. So row k*n+i of W corresponds to the output component for sum over k?
                    # Yes, the "row" of W is the triple (k, i) which identifies the specific multiplication step.
                    # So for a fixed output (i, j), we sum over k?
                    # No, W is a matrix. 
                    # The formula is: $C = W^T U A V^T B$?
                    # Usually: $C_{ij} = \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} \dots$
                    
                    # Let's stick to the definition:
                    # $C_{ij} = \sum_{r} W_{row\_of\_C, r} \cdot (U \cdot A) \dots$
                    # Actually, the simplest check is:
                    # Flatten A to vector a (25), B to vector b (25).
                    # C_flat (25) = W^T (U * a) (V * b)? No.
                    # C_flat[r] = sum_c U[a_idx, c] V[b_idx, c] W[c_idx, r]? No.
                    
                    # The identity is:
                    # $C_{ij} = \sum_{r} W_{k n + i, r} U_{i m + k, r} V_{j p + k, r}$?
                    # No, W index is $k n + i$.
                    # The decomposition is:
                    # $\sum_{r} U_{i, k, r} V_{k, j, r} W_{i, j, r}$?
                    
                    # Let's check the schoolbook in the seed:
                    # U[i*m+j][col] = 1
                    # V[j*p+k][col] = 1
                    # W[k*n+i][col] = 1
                    # C[i,j] = sum_k A[i,k] B[k,j].
                    # So the term is A[i,k] B[k,j].
                    # U row is i*m+k. (I used i*m+j in seed, let's assume j is k here).
                    # V row is k*p+j.
                    # W row is j*n+i.
                    
                    # So for our 110 rank:
                    # Term r contributes $U_{A\_idx, r} A_{A\_idx} \cdot V_{B\_idx, r} B_{B\_idx} \cdot W_{C\_idx, r}$ to C?
                    # No, it contributes $U_{a, r} V_{b, r} W_{c, r} A_a B_b$ to $C_c$.
                    
                    # So $C_c = \sum_{r} W_{c, r} ( \sum_{a} U_{a, r} A_a ) ( \sum_{b} V_{b, r} B_b )$.
                    
                    C_calc_flat = np.zeros(25)
                    for r in range(110):
                        ua = sum(U[a][r] * A_test.flat[a] for a in range(25))
                        vb = sum(V[b][r] * B_test.flat[b] for b in range(25))
                        for c in range(25):
                            C_calc_flat[c] += W[c][r] * ua * vb
                            
                    C_calc = C_calc_flat.reshape(5,5)
                    if not np.allclose(C_calc, C_expected, atol=1e-10):
                        return schoolbook()
        
                    if time_exceeded():
                        return schoolbook()
        
                    return {"U": U, "V": V, "W": W}

    except Exception as e:
        # Fallback on any error
        return schoolbook()

# EVOLVE-BLOCK-END
