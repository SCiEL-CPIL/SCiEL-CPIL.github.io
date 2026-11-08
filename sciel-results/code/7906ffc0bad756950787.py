"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = 4, 4, 5
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    # Build the matrix multiplication tensor indices
    # M[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i
    # We only need to store the non-zero indices
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    # Schoolbook decomposition (Rank 80)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = 80

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        rank = U.shape[1]
        # Check all nonzero entries
        for a, b, c in nonzero_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Check zero entries (sample)
        # To be rigorous, we should check all, but 16*20*20 = 6400 is small enough
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Check if it's a nonzero triple
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue # Skip nonzero
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def round_to_half_int(x):
        return np.round(2 * x) / 2.0

    # --- Structural Strategy: Combine Rank-19 (for 4x4x4 part) + 16 Schoolbook (for 5th col) ---
    # Total Rank = 35
    
    # We need to construct a valid rank-19 decomposition for the 4x4x4 multiplication tensor.
    # The tensor is T_{ijk} where i,j in 0..3 (A), j,k in 0..3 (B), k,i in 0..3 (C).
    # This is a known problem. The "Winograd" or "Strassen-like" decompositions exist.
    # However, generating a *correct* rank-19 decomposition from scratch is complex.
    # Let's look for a simpler, guaranteed valid decomposition that is better than 80.
    
    # Actually, the prompt suggests a strategy: "Utilize the known rank-19 tensor decomposition for 4x4 matrix multiplication to handle the first four columns of the output, and append 16 independent rank-1 terms to handle the fifth column via standard scalar multiplications, resulting in a total rank of 35."
    
    # Let's verify if we can just use a known algorithm.
    # Strassen for 4x4:
    # Split A into 4 blocks of 2x2. Split B into 4 blocks of 2x2.
    # Compute 7 products of 2x2 matrices.
    # Each 2x2 product can be done with 7 scalar multiplications (Strassen).
    # Total scalar multiplications = 7 * 7 = 49.
    # This gives a rank-49 decomposition for the 4x4x4 part.
    # Plus 16 for the 5th column (4x4 * 4x1).
    # Total Rank = 49 + 16 = 65.
    
    # Can we do better?
    # The inspiration code attempts rank 65.
    # The prompt suggests rank 35. This implies a rank-19 decomposition for 4x4x4.
    # Is rank 19 valid for 4x4x4?
    # The lower bound for 4x4x4 is known to be around 49? No, Strassen is 49.
    # Wait, Strassen is O(n^log2(7)). For n=4, it's 4^log2(7) ~ 4^2.8 ~ 21.8? No.
    # Strassen complexity for n=2 is 7. For n=4, it is 7 * 7 = 49.
    # So rank 49 is the standard Strassen rank.
    # Is there a rank < 49 for 4x4x4?
    # The best known upper bound for 4x4x4 is 49 (Strassen). Lower bound is 49? No, lower bound is 49?
    # Actually, the exact rank of the 4x4x4 matrix multiplication tensor is not known to be 49.
    # But 49 is a valid upper bound.
    # If the prompt claims rank 19 is known, I should trust the prompt's "Strategy" hint, but I must ensure correctness.
    # If I cannot generate a valid rank-19 decomposition, I will fall back to the valid rank-65 (Strassen + Schoolbook) which is better than 80.
    
    # Let's implement the Rank-65 decomposition (Strassen for 4x4 + Schoolbook for col 4).
    # This is a structural improvement over the generic ALS search in the current program which struggles to find solutions < 80.
    
    def build_rank_65_decomposition():
        rank = 65
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Part 1: 4x4 * 4x4 (cols 0-3 of B, cols 0-3 of C)
        # We use Strassen's algorithm for 4x4 matrices.
        # A (4x4) split into 2x2 blocks: A00, A01, A10, A11
        # B (4x4) split into 2x2 blocks: B00, B01, B10, B11
        # 
        # Indices for A (i, j):
        # A00: i in [0,1], j in [0,1]
        # A01: i in [0,1], j in [2,3]
        # A10: i in [2,3], j in [0,1]
        # A11: i in [2,3], j in [2,3]
        #
        # Indices for B (j, k) where k in [0,3]:
        # B00: j in [0,1], k in [0,1]
        # B01: j in [0,1], k in [2,3]
        # B10: j in [2,3], k in [0,1]
        # B11: j in [2,3], k in [2,3]
        #
        # Indices for C (k, i) where k in [0,3], i in [0,3]:
        # C00: k in [0,1], i in [0,1]
        # C01: k in [0,1], i in [2,3]
        # C10: k in [2,3], i in [0,1]
        # C11: k in [2,3], i in [2,3]
        
        # Strassen steps for 2x2 blocks:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Each Mi is a 2x2 matrix product. We decompose each 2x2 product using Strassen (7 scalar mults).
        # So we have 7 * 7 = 49 terms.
        
        # Let's define the 7 scalar multiplications for a generic 2x2 product P * Q = R.
        # P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
        # s1 = (p00 + p11) * (q00 + q11)
        # s2 = (p10 + p11) * q00
        # s3 = p00 * (q01 - q11)
        # s4 = p11 * (q10 - q00)
        # s5 = (p00 + p01) * q11
        # s6 = (p10 - p00) * (q00 + q01)
        # s7 = (p01 - p11) * (q10 + q11)
        #
        # R00 = s1 + s4 - s5 + s7
        # R01 = s3 + s5
        # R10 = s2 + s4
        # R11 = s1 - s2 + s3 + s6
        
        # We need to map the "super" Strassen terms (M1..M7) to the "inner" Strassen terms.
        # For each super term, we identify the linear combinations of A and B elements involved in P and Q.
        
        # Helper to add a rank-1 term
        def add_term(col, u_list, v_list, w_list):
            for i, j, c in u_list:
                if c != 0:
                    U[i * m + j, col] += c
            for j, k, c in v_list:
                if c != 0:
                    V[j * p + k, col] += c
            for k, i, c in w_list:
                if c != 0:
                    W[k * n + i, col] += c
        
        col = 0
        
        # Define block indices for A, B, C
        # A blocks: 0:A00, 1:A01, 2:A10, 3:A11
        # B blocks: 0:B00, 1:B01, 2:B10, 3:B11
        # C blocks: 0:C00, 1:C01, 2:C10, 3:C11
        
        # Super Term 0: M1 = (A00 + A11) * (B00 + B11)
        # P = A00 + A11, Q = B00 + B11
        # Result contributes to all 4 C blocks?
        # M1 contributes to C00, C01, C10, C11 via the inner Strassen formula.
        
        # Let's iterate through the 7 super terms.
        # For each super term, we have a specific P (linear combo of A blocks) and Q (linear combo of B blocks).
        # We then decompose P*Q into 7 scalar multiplications.
        
        super_terms = [
            # (P_blocks, Q_blocks)
            # P_blocks: coeffs for A00, A01, A10, A11
            # Q_blocks: coeffs for B00, B01, B10, B11
            ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1]) # M7
        ]
        
        # Inner Strassen terms for 2x2 product
        # Each inner term is a scalar multiplication of a linear combo of P elements and a linear combo of Q elements.
        # We need to map P elements (which are 2x2 matrices) to global A indices.
        # P is a 2x2 matrix. P[r,c] is a 2x2 block? No, P is the 2x2 matrix of coefficients.
        # Wait, in the super Strassen, P and Q are 2x2 matrices whose entries are 2x2 blocks?
        # No. A is 4x4. We split into 2x2 blocks.
        # A00 is 2x2.
        # In M1 = (A00+A11)(B00+B11), the operands are 2x2 matrices.
        # Let P = A00+A11. P is a 2x2 matrix.
        # Let Q = B00+B11. Q is a 2x2 matrix.
        # We compute R = P*Q using Strassen (7 scalar mults).
        # R is a 2x2 matrix.
        # The entries of R correspond to the blocks of the result C.
        # Specifically, for M1, the result R is added to all 4 C blocks?
        # Let's check the Strassen reconstruction for 4x4.
        # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
        # C01 = M1[0,1] + M4[0,1] - M5[0,1] + M7[0,1]
        # ...
        # Actually, the standard Strassen for 4x4 using 2x2 blocks works as follows:
        # The 7 products M1..M7 are 2x2 matrices.
        # The 4 blocks of C are linear combinations of the 4 entries of each Mi.
        
        # Let's define the contribution of each Mi to the C blocks.
        # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
        # C01 = M1[0,1] + M4[0,1] - M5[0,1] + M7[0,1]
        # C10 = M1[1,0] + M4[1,0] - M5[1,0] + M7[1,0]
        # C11 = M1[1,1] + M4[1,1] - M5[1,1] + M7[1,1]
        # Wait, this is for the 2x2 case.
        # For 4x4, the blocks are 2x2.
        # C is 4x4.
        # C00 (top-left 2x2) = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
        # C01 (top-right 2x2) = M1[0,1] + M4[0,1] - M5[0,1] + M7[0,1]
        # C10 (bottom-left 2x2) = M1[1,0] + M4[1,0] - M5[1,0] + M7[1,0]
        # C11 (bottom-right 2x2) = M1[1,1] + M4[1,1] - M5[1,1] + M7[1,1]
        
        # This means each entry M_i[r,c] contributes to one specific 2x2 block of C.
        # And each entry M_i[r,c] is computed by 7 scalar multiplications.
        
        # So for each super term i (0..6), and for each inner entry (r,c) in 0..1,
        # we have 7 scalar multiplications.
        # Total 7 * 4 * 7 = 196? No.
        # The 7 scalar multiplications compute the 4 entries of the 2x2 result.
        # So for each super term, we have 7 scalar multiplications.
        # Total 7 * 7 = 49 scalar multiplications.
        
        # Let's implement this carefully.
        
        # For each super term index st (0..6):
        for st in range(7):
            p_blocks, q_blocks = super_terms[st]
            
            # We need to decompose the product of the 2x2 matrix P (formed by p_blocks)
            # and the 2x2 matrix Q (formed by q_blocks) into 7 scalar multiplications.
            
            # P is a 2x2 matrix. P[0,0] is the top-left 2x2 block of the sum.
            # P[0,0] = sum(p_blocks[b] * A_b[0,0]) for b in 0..3
            # P[0,1] = sum(p_blocks[b] * A_b[0,1])
            # P[1,0] = sum(p_blocks[b] * A_b[1,0])
            # P[1,1] = sum(p_blocks[b] * A_b[1,1])
            
            # Similarly for Q.
            
            # The 7 scalar multiplications are:
            # s1 = (P[0,0] + P[1,1]) * (Q[0,0] + Q[1,1])
            # s2 = (P[1,0] + P[1,1]) * Q[0,0]
            # s3 = P[0,0] * (Q[0,1] - Q[1,1])
            # s4 = P[1,1] * (Q[1,0] - Q[0,0])
            # s5 = (P[0,0] + P[0,1]) * Q[1,1]
            # s6 = (P[1,0] - P[0,0]) * (Q[0,0] + Q[0,1])
            # s7 = (P[0,1] - P[1,1]) * (Q[1,0] + Q[1,1])
            
            # We need to expand these into global indices.
            
            # Helper to get the list of (i, j, coeff) for a P element P[r,c]
            def get_p_elem(r, c):
                res = []
                for b_idx, coeff in enumerate(p_blocks):
                    if coeff == 0: continue
                    # Block b_idx is A00, A01, A10, A11
                    # A00: i in [0,1], j in [0,1]. Local (r,c) -> Global (r, c)
                    # A01: i in [0,1], j in [2,3]. Local (r,c) -> Global (r, 2+c)
                    # A10: i in [2,3], j in [0,1]. Local (r,c) -> Global (2+r, c)
                    # A11: i in [2,3], j in [2,3]. Local (r,c) -> Global (2+r, 2+c)
                    
                    if b_idx == 0:
                        gi, gj = r, c
                    elif b_idx == 1:
                        gi, gj = r, 2+c
                    elif b_idx == 2:
                        gi, gj = 2+r, c
                    else:
                        gi, gj = 2+r, 2+c
                    
                    res.append((gi, gj, coeff))
                return res
            
            # Helper to get the list of (j, k, coeff) for a Q element Q[r,c]
            def get_q_elem(r, c):
                res = []
                for b_idx, coeff in enumerate(q_blocks):
                    if coeff == 0: continue
                    # Block b_idx is B00, B01, B10, B11
                    # B00: j in [0,1], k in [0,1]. Local (r,c) -> Global (r, c)
                    # B01: j in [0,1], k in [2,3]. Local (r,c) -> Global (r, 2+c)
                    # B10: j in [2,3], k in [0,1]. Local (r,c) -> Global (2+r, c)
                    # B11: j in [2,3], k in [2,3]. Local (r,c) -> Global (2+r, 2+c)
                    
                    if b_idx == 0:
                        gj, gk = r, c
                    elif b_idx == 1:
                        gj, gk = r, 2+c
                    elif b_idx == 2:
                        gj, gk = 2+r, c
                    else:
                        gj, gk = 2+r, 2+c
                    
                    res.append((gj, gk, coeff))
                return res
            
            # Helper to get the list of (k, i, coeff) for a C element C[r,c]
            # C is the result block.
            # The result of P*Q is a 2x2 matrix R.
            # R[0,0] contributes to C00 (k in [0,1], i in [0,1])
            # R[0,1] contributes to C01 (k in [0,1], i in [2,3])
            # R[1,0] contributes to C10 (k in [2,3], i in [0,1])
            # R[1,1] contributes to C11 (k in [2,3], i in [2,3])
            
            # We need to know which C block each scalar multiplication contributes to.
            # The scalar multiplications s1..s7 are combined to form R[0,0], R[0,1], R[1,0], R[1,1].
            # R00 = s1 + s4 - s5 + s7
            # R01 = s3 + s5
            # R10 = s2 + s4
            # R11 = s1 - s2 + s3 + s6
            
            # So for each scalar multiplication s_idx (0..6), we need to add its contribution to the appropriate W entries.
            # The U and V entries are the same for all s_idx in this super term? No, U and V depend on the specific linear combination in s_idx.
            
            # Let's define the 7 scalar multiplications explicitly.
            # Each is a product of a linear combo of P elements and a linear combo of Q elements.
            
            # s1: (P00+P11) * (Q00+Q11)
            # s2: (P10+P11) * Q00
            # s3: P00 * (Q01-Q11)
            # s4: P11 * (Q10-Q00)
            # s5: (P00+P01) * Q11
            # s6: (P10-P00) * (Q00+Q01)
            # s7: (P01-P11) * (Q10+Q11)
            
            # For each s_idx, we determine the u_list and v_list.
            # Then we determine which C blocks (r,c) this s_idx contributes to, and with what coefficient.
            
            # Coefficients for C blocks:
            # s1: C00 (+1), C11 (+1)
            # s2: C10 (+1), C11 (-1)
            # s3: C01 (+1), C11 (+1)
            # s4: C00 (+1), C10 (+1)
            # s5: C00 (-1), C01 (+1)
            # s6: C11 (+1)
            # s7: C00 (+1)
            
            # Wait, let's re-verify the Strassen reconstruction.
            # R00 = s1 + s4 - s5 + s7
            # R01 = s3 + s5
            # R10 = s2 + s4
            # R11 = s1 - s2 + s3 + s6
            
            # So:
            # s1 contributes to R00 (+1) and R11 (+1)
            # s2 contributes to R10 (+1) and R11 (-1)
            # s3 contributes to R01 (+1) and R11 (+1)
            # s4 contributes to R00 (+1) and R10 (+1)
            # s5 contributes to R00 (-1) and R01 (+1)
            # s6 contributes to R11 (+1)
            # s7 contributes to R00 (+1)
            
            # Now, R00 is C00, R01 is C01, R10 is C10, R11 is C11.
            # C00: k in [0,1], i in [0,1]
            # C01: k in [0,1], i in [2,3]
            # C10: k in [2,3], i in [0,1]
            # C11: k in [2,3], i in [2,3]
            
            # We need to expand the u_list and v_list for each s_idx.
            
            def get_u_v_for_s(s_idx):
                if s_idx == 0: # s1
                    p_elems = [(0, 1), (3, 1)] # P00, P11
                    q_elems = [(0, 1), (3, 1)] # Q00, Q11
                elif s_idx == 1: # s2
                    p_elems = [(2, 1), (3, 1)] # P10, P11
                    q_elems = [(0, 1)] # Q00
                elif s_idx == 2: # s3
                    p_elems = [(0, 1)] # P00
                    q_elems = [(1, 1), (3, -1)] # Q01, -Q11
                elif s_idx == 3: # s4
                    p_elems = [(3, 1)] # P11
                    q_elems = [(2, 1), (0, -1)] # Q10, -Q00
                elif s_idx == 4: # s5
                    p_elems = [(0, 1), (1, 1)] # P00, P01
                    q_elems = [(3, 1)] # Q11
                elif s_idx == 5: # s6
                    p_elems = [(2, 1), (0, -1)] # P10, -P00
                    q_elems = [(0, 1), (1, 1)] # Q00, Q01
                else: # s7
                    p_elems = [(1, 1), (3, -1)] # P01, -P11
                    q_elems = [(2, 1), (3, 1)] # Q10, Q11
                
                u_list = []
                v_list = []
                
                for (pr, pc), pc_coeff in p_elems:
                    # Get the global A indices for P[pr, pc]
                    elems = get_p_elem(pr, pc)
                    for (gi, gj, coeff) in elems:
                        u_list.append((gi, gj, coeff * pc_coeff))
                
                for (qr, qc), qc_coeff in q_elems:
                    # Get the global B indices for Q[qr, qc]
                    elems = get_q_elem(qr, qc)
                    for (gj, gk, coeff) in elems:
                        v_list.append((gj, gk, coeff * qc_coeff))
                        
                return u_list, v_list
            
            # Define the C block contributions for each s_idx
            # s_idx: list of (c_block_idx, coeff)
            # c_block_idx: 0:C00, 1:C01, 2:C10, 3:C11
            c_contribs = [
                [(0, 1), (3, 1)], # s1
                [(2, 1), (3, -1)], # s2
                [(1, 1), (3, 1)], # s3
                [(0, 1), (2, 1)], # s4
                [(0, -1), (1, 1)], # s5
                [(3, 1)], # s6
                [(0, 1)] # s7
            ]
            
            for s_idx in range(7):
                u_list, v_list = get_u_v_for_s(s_idx)
                
                # We need to add a term for each C block contribution.
                # But wait, a single rank-1 term has a single W vector.
                # The W vector must be zero for all C entries except those in the contributing blocks.
                # And the U and V vectors are the same for all C entries in this term?
                # Yes, because the scalar multiplication is a single product.
                # The result is distributed to multiple C entries.
                # So we add ONE rank-1 term for this s_idx.
                # The W vector will have non-zero entries in all contributing C blocks.
                
                w_list = []
                for c_block_idx, coeff in c_contribs[s_idx]:
                    # c_block_idx 0: C00 (k in [0,1], i in [0,1])
                    # c_block_idx 1: C01 (k in [0,1], i in [2,3])
                    # c_block_idx 2: C10 (k in [2,3], i in [0,1])
                    # c_block_idx 3: C11 (k in [2,3], i in [2,3])
                    
                    if c_block_idx == 0:
                        for k in range(2):
                            for i in range(2):
                                w_list.append((k, i, coeff))
                    elif c_block_idx == 1:
                        for k in range(2):
                            for i in range(2, 4):
                                w_list.append((k, i, coeff))
                    elif c_block_idx == 2:
                        for k in range(2, 4):
                            for i in range(2):
                                w_list.append((k, i, coeff))
                    else:
                        for k in range(2, 4):
                            for i in range(2, 4):
                                w_list.append((k, i, coeff))
                
                add_term(col, u_list, v_list, w_list)
                col += 1
        
        # Part 2: 4x4 * 4x1 (col 4 of B, col 4 of C)
        # This is just standard matrix-vector multiplication.
        # C[:, 4] = A * B[:, 4]
        # C[k, i] = sum_j A[i, j] * B[j, 4]
        # This requires 16 scalar multiplications.
        
        for i in range(4):
            for j in range(4):
                a = i * m + j
                b = j * p + 4
                c = 4 * n + i # k=4, i in [0,3]
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
        
        return U, V, W

    U_65, V_65, W_65 = build_rank_65_decomposition()
    
    # Verify the rank-65 decomposition
    if check_decomposition(U_65, V_65, W_65, tol=1e-9):
        best_rank = 65
        best_result = {'U': U_65.tolist(), 'V': V_65.tolist(), 'W': W_65.tolist()}
        return best_result
    else:
        # Fallback to schoolbook if Strassen implementation is flawed
        pass

    # If we have time, try to improve further or just return the best found
    # Since 65 is a significant improvement over 80, we return it immediately if valid.
    
    return best_result
