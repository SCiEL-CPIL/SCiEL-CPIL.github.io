"""Rewritten program: rank-34 exact decomposition for M_{2,3,6} using Strassen on a 2x2x2 block."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_strassen_222():
    """Return (U, V, W) for M_{2,2,2} with rank 7 using Strassen.
    U has shape (4, 7), V (4, 7), W (4, 7).
    Indexing: U[i*2 + j, c], V[j*2 + k, c], W[k*2 + i, c].
    """
    # Strassen 7 products for multiplying 2x2 by 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # Each M is a sum of products of linear combinations.
    # In tensor decomposition, each "product" of a linear form in A and a linear form in B
    # contributes to the decomposition.
    #
    # For M_{2,2,2}, the standard Strassen rank-7 decomposition:
    # We need to express each C_{i,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    # But actually the tensor is T[i,j,k] = delta (the multiplication tensor).
    # The decomposition says: T = sum_c U_c \otimes V_c \otimes W_c
    # where (U_c)_{i,j} is coefficient of A_{i,j} in the i-th linear form... 
    #
    # Actually let me think more carefully. The matrix multiplication tensor T_{ijk} has T_{ijk}=1 
    # for all valid indices. A rank-r decomposition writes:
    # T_{ijk} = sum_{c=1}^r U_{i, c} V_{j, c} W_{k, c}
    # where U has shape (n*m, r) indexed by (i,j) pair -> row i*m+j, V shape (m*p, r) indexed by (j,k) -> row j*p+k, W shape (p*n, r) indexed by (k,i) -> row k*n+i.
    #
    # For Strassen on 2x2x2:
    # The 7 terms correspond to bilinear forms. Each bilinear form M_c = a_c^T A b_c where a_c, b_c are vectors.
    # But that's not quite right either. Let me use the standard formulation.
    #
    # Standard: C = A*B where A is 2x2, B is 2x2.
    # Strassen computes 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    #
    # Each P is a product of a linear combination of A entries and a linear combination of B entries.
    # P = (sum_ia a_i A_i) * (sum_kb b_k B_k) = sum_ia sum_kb a_i b_k A_i B_k
    # In the tensor decomposition, each (i, k) pair in a P contributes to the (i, j, k) tensor entry...
    # Wait, A_i is A_{i,j} for some j, B_k is B_{j,k} for some j. The j must match.
    #
    # Let me be explicit. A is 2x2 with entries A_{i,j}, i=0,1, j=0,1.
    # B is 2x2 with entries B_{j,k}, j=0,1, k=0,1.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #
    # P1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   = (i=0,j=0,k=0) + (i=0,j=1,k=1) + (i=1,j=0,k=0) + (i=1,j=1,k=1)
    #   Coefficients: all 1.
    #
    # In the decomposition, for P1, we have contributions to T_{i,j,k}:
    # T_{0,0,0} += 1*1 = 1
    # T_{0,1,1} += 1*1 = 1
    # T_{1,0,0} += 1*1 = 1
    # T_{1,1,1} += 1*1 = 1
    #
    # So for column c corresponding to P1:
    # U row (i*m+j) = coefficient of A_{i,j} in the A-part of P1
    # V row (j*p+k) = coefficient of B_{j,k} in the B-part of P1
    # W row (k*n+i) = coefficient needed so that sum_j U_{i,j,c} * V_{j,k,c} * W_{k,i,c} gives the right T_{i,j,k}
    #
    # Hmm, this is getting confusing. Let me use a different approach.
    #
    # The tensor T has entries T[i,j,k] for i in range(n), j in range(m), k in range(p).
    # A rank-r decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For a single "term" that corresponds to a product of linear forms:
    # L_A = sum_{i,j} alpha_{i,j} * A_{i,j}
    # L_B = sum_{j,k} beta_{j,k} * B_{j,k}
    # Product = L_A * L_B = sum_{i,j} sum_{j',k} alpha_{i,j} * beta_{j',k} * A_{i,j} * B_{j',k}
    # This contributes to C_{i,k} only when j=j', i.e., the sum over j of alpha_{i,j} * beta_{j,k} * A_{i,j} * B_{j,k}
    #
    # So the contribution to T[i,j,k] from this term is alpha_{i,j} * beta_{j,k}.
    #
    # In the decomposition, if we have just this one term (rank 1), then:
    # U[i*m+j, c] = alpha_{i,j}
    # V[j*p+k, c] = beta_{j,k}
    # W[k*n+i, c] = 1
    # Then sum_c U[...] * V[...] * W[...] = alpha_{i,j} * beta_{j,k} * 1 = alpha_{i,j} * beta_{j,k}
    # Which matches T[i,j,k] for this term.
    #
    # So for a single bilinear term with coefficients alpha and beta:
    # U[i*m+j, c] = alpha[i,j]
    # V[j*p+k, c] = beta[j,k]
    # W[k*n+i, c] = 1
    #
    # For Strassen, we have 7 such terms, and the final C is a linear combination.
    # The W matrix will contain the coefficients from the linear combination.
    #
    # Let me define the 7 terms for 2x2x2:
    
    # Term 1: P1 = (A00+A11)(B00+B11)
    # alpha: A00=1, A11=1, others 0
    # beta: B00=1, B11=1, others 0
    # Contribution to T: T[0,0,0]+=1, T[0,1,1]+=1, T[1,0,0]+=1, T[1,1,1]+=1
    # W coefficient: +1 for C00, -1 for C00 (wait, no)
    
    # Let me just compute it directly by building the tensor and decomposing.
    # Actually, let me just hardcode the rank-7 Strassen decomposition for M_{2,2,2}.
    
    # The 7 products and their contributions:
    # P1: A-part = [1,0; 0,1] (A00 + A11), B-part = [1,0; 0,1] (B00 + B11)
    #     T contributions: (0,0,0):1, (0,1,1):1, (1,0,0):1, (1,1,1):1
    #     C00 += P1, C11 -= P1... wait let me re-derive.
    
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # Each P_c contributes to multiple (i,k) entries of C with coefficient 1.
    # In the tensor T[i,j,k], the entry for a given (i,j,k) is the sum over c of [coeff of (i,j) in A-part of P_c] * [coeff of (j,k) in B-part of P_c] * [W_{k*n+i, c}]
    
    # Let me define the A-parts and B-parts for each P_c:
    # P1: A-part = A00 + A11 => alpha = {(0,0):1, (1,1):1}
    #     B-part = B00 + B11 => beta = {(0,0):1, (1,1):1}
    # P2: A-part = A10 + A11 => alpha = {(1,0):1, (1,1):1}
    #     B-part = B00 => beta = {(0,0):1}
    # P3: A-part = A00 => alpha = {(0,0):1}
    #     B-part = B10 - B11 => beta = {(1,0):1, (1,1):-1}
    # P4: A-part = A11 => alpha = {(1,1):1}
    #     B-part = B01 - B00 => beta = {(0,1):1, (0,0):-1}
    # P5: A-part = A00 + A01 => alpha = {(0,0):1, (0,1):1}
    #     B-part = B11 => beta = {(1,1):1}
    # P6: A-part = A10 - A00 => alpha = {(1,0):1, (0,0):-1}
    #     B-part = B00 + B01 => beta = {(0,0):1, (0,1):1}
    # P7: A-part = A01 - A11 => alpha = {(0,1):1, (1,1):-1}
    #     B-part = B10 + B11 => beta = {(1,0):1, (1,1):1}
    
    # Now, for each P_c and each (i,k), the contribution to C_{i,k} from P_c is:
    # sum_j alpha_{i,j} * beta_{j,k}
    # And the W_{k*n+i, c} should be the coefficient of P_c in the expression for C_{i,k}.
    
    # Let me compute W for each (i,k,c):
    # C00 = P1 + P4 - P5 - P7
    #   P1: sum_j alpha1_{0,j} * beta1_{j,0} = alpha1_{0,0}*beta1_{0,0} + alpha1_{0,1}*beta1_{1,0} = 1*1 + 0*0 = 1
    #   P4: sum_j alpha4_{0,j} * beta4_{j,0} = alpha4_{0,0}*beta4_{0,0} + alpha4_{0,1}*beta4_{1,0} = 0*(-1) + 0*0 = 0
    #   Hmm, this doesn't seem right. Let me re-check.
    
    # Actually, the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For a single P_c with alpha and beta:
    # U[i*m+j, c] = alpha_{i,j}
    # V[j*p+k, c] = beta_{j,k}
    # W[k*n+i, c] = gamma_{i,k} (to be determined)
    
    # Then T[i,j,k] contribution from P_c = alpha_{i,j} * beta_{j,k} * gamma_{i,k}
    
    # For the full Strassen, T[i,j,k] = sum_c alpha_c_{i,j} * beta_c_{j,k} * gamma_c_{i,k}
    
    # And we know that C_{i,k} = sum_j T[i,j,k] * (the actual multiplication)
    # Wait, no. T[i,j,k] IS the multiplication tensor, so T[i,j,k] = 1 for all i,j,k.
    
    # I think I'm overcomplicating this. Let me just directly construct the decomposition
    # by verifying that the sum of outer products equals the identity tensor.
    
    # For M_{2,2,2}, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need: 1 = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] for all i,j,k.
    
    # Let me just use a known rank-7 decomposition. Here's one:
    
    n2, m2, p2 = 2, 2, 2
    rank2 = 7
    U2 = [[0]*rank2 for _ in range(n2*m2)]
    V2 = [[0]*rank2 for _ in range(m2*p2)]
    W2 = [[0]*rank2 for _ in range(p2*n2)]
    
    # Define the 7 terms
    # Each term: (alpha dict, beta dict, gamma dict)
    # alpha: {(i,j): coeff}, beta: {(j,k): coeff}, gamma: {(i,k): coeff}
    
    terms = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):-1, (0,1):1}),
        # P4
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1, (1,1):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
    ]
    
    for c, (alpha, beta, gamma) in enumerate(terms):
        for (i,j), a in alpha.items():
            U2[i*m2+j][c] = a
        for (j,k), b in beta.items():
            V2[j*p2+k][c] = b
        for (i,k), g in gamma.items():
            W2[k*n2+i][c] = g
    
    # Verify: T[i,j,k] should equal 1 for all i,j,k
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                s = 0
                for c in range(rank2):
                    s += U2[i*m2+j][c] * V2[j*p2+k][c] * W2[k*n2+i][c]
                if s != 1:
                    raise ValueError(f"Strassen verification failed at ({i},{j},{k}): got {s}")
    
    return U2, V2, W2


def _build_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy:
    # Split A (2x3) into A1 (2x1, col 0) and A2 (2x2, cols 1,2)
    # Split B (3x6) into B1 (1x6, row 0) and B2 (2x6, rows 1,2)
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x1 times 1x6. Rank = 2*1*6 = 12 (schoolbook)
    # A2*B2: 2x2 times 2x6. Split B2 into B2a (2x3, cols 0-2) and B2b (2x3, cols 3-5)
    #   A2*B2a: 2x2 times 2x3. Use Strassen for first 2 cols (2x2x2, rank 7) + schoolbook for 3rd col (2x1*1x1=... wait)
    #   Actually, 2x2 times 2x3: split into 2x2*2x2 (rank 7 Strassen) + 2x2*2x1 (rank 4 schoolbook) = 11
    #   A2*B2b: similarly 11
    #
    # Total: 12 + 11 + 11 = 34
    
    rank_total = 34
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    col_offset = 0
    
    # Part 1: A1 * B1 (2x1 times 1x6), schoolbook, rank 12
    # A1: rows i=0,1, col j=0
    # B1: row j=0, cols k=0..5
    # U row i*m+0, V row 0*p+k, W row k*n+i
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0..5
            U[i*m+0][col_offset] = 1
            V[0*p+k][col_offset] = 1
            W[k*n+i][col_offset] = 1
            col_offset += 1
    
    # Part 2: A2 * B2a (2x2 times 2x3), where A2 uses j=1,2 and B2a uses j=1,2, k=0,1,2
    # Split into: A2 * B2a_1 (2x2 times 2x2, k=0,1) using Strassen (rank 7)
    #            + A2 * B2a_2 (2x2 times 2x1, k=2) using schoolbook (rank 4)
    
    # Get Strassen decomposition for 2x2x2
    U2, V2, W2 = _build_strassen_222()
    
    # Map local indices (i_loc, j_loc, k_loc) to global (i, j, k)
    # i_loc = 0 -> i=0, i_loc = 1 -> i=1
    # j_loc = 0 -> j=1, j_loc = 1 -> j=2
    # k_loc = 0 -> k=0, k_loc = 1 -> k=1
    
    for c in range(7):
        for i_loc in range(2):
            for j_loc in range(2):
                if U2[i_loc*2+j_loc][c] != 0:
                    i = i_loc
                    j = 1 + j_loc
                    U[i*m+j][col_offset] = U2[i_loc*2+j_loc][c]
        for j_loc in range(2):
            for k_loc in range(2):
                if V2[j_loc*2+k_loc][c] != 0:
                    j = 1 + j_loc
                    k = k_loc
                    V[j*p+k][col_offset] = V2[j_loc*2+k_loc][c]
        for k_loc in range(2):
            for i_loc in range(2):
                if W2[k_loc*2+i_loc][c] != 0:
                    k = k_loc
                    i = i_loc
                    W[k*n+i][col_offset] = W2[k_loc*2+i_loc][c]
        col_offset += 1
    
    # Schoolbook for A2 * B2a_2 (k=2): 2x2 times 2x1, rank 4
    for i in range(2):
        for j_loc in range(2):
            j = 1 + j_loc
            k = 2
            U[i*m+j][col_offset] = 1
            V[j*p+k][col_offset] = 1
            W[k*n+i][col_offset] = 1
            col_offset += 1
    
    # Part 3: A2 * B2b (2x2 times 2x3), where B2b uses j=1,2, k=3,4,5
    # Split into: A2 * B2b_1 (2x2 times 2x2, k=3,4) using Strassen (rank 7)
    #            + A2 * B2b_2 (2x2 times 2x1, k=5) using schoolbook (rank 4)
    
    for c in range(7):
        for i_loc in range(2):
            for j_loc in range(2):
                if U2[i_loc*2+j_loc][c] != 0:
                    i = i_loc
                    j = 1 + j_loc
                    U[i*m+j][col_offset] = U2[i_loc*2+j_loc][c]
        for j_loc in range(2):
            for k_loc in range(2):
                if V2[j_loc*2+k_loc][c] != 0:
                    j = 1 + j_loc
                    k = 3 + k_loc
                    V[j*p+k][col_offset] = V2[j_loc*2+k_loc][c]
        for k_loc in range(2):
            for i_loc in range(2):
                if W2[k_loc*2+i_loc][c] != 0:
                    k = 3 + k_loc
                    i = i_loc
                    W[k*n+i][col_offset] = W2[k_loc*2+i_loc][c]
        col_offset += 1
    
    # Schoolbook for A2 * B2b_2 (k=5): 2x2 times 2x1, rank 4
    for i in range(2):
        for j_loc in range(2):
            j = 1 + j_loc
            k = 5
            U[i*m+j][col_offset] = 1
            V[j*p+k][col_offset] = 1
            W[k*n+i][col_offset] = 1
            col_offset += 1
    
    # Verify decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank_total):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
