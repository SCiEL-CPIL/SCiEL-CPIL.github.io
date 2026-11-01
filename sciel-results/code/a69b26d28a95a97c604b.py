"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 3, 7]

def construct():
    # Rank 39 decomposition
    # A is 2x3, B is 3x7
    # Split A into A1 (2x2, cols 0-1) and A2 (2x1, col 2)
    # Split B into B1 (2x7, rows 0-1) and B2 (1x7, row 2)
    # C = A1 B1 + A2 B2
    
    # A2 B2 is outer product of A2 (2x1) and B2 (1x7). Rank 14.
    # A1 B1 is 2x2 times 2x7.
    # Split B1 into B1a (2x3, cols 0-2) and B1b (2x4, cols 3-6).
    # A1 B1a: Split B1a into B1a1 (2x2, cols 0-1) and B1a2 (2x1, col 2).
    #   A1 B1a1: Rank 7 (Strassen)
    #   A1 B1a2: Rank 4 (Schoolbook)
    # A1 B1b: Split B1b into B1b1 (2x2, cols 0-1) and B1b2 (2x2, cols 2-3).
    #   A1 B1b1: Rank 7 (Strassen)
    #   A1 B1b2: Rank 7 (Strassen)
    
    # Total Rank: 14 + (7 + 4) + (7 + 7) = 39.

    rank = 39
    U = [[0] * rank for _ in range(6)] # 2*3
    V = [[0] * rank for _ in range(21)] # 3*7
    W = [[0] * rank for _ in range(14)] # 7*2

    col = 0
    
    # Helper to add a rank-1 term
    # u_idx is index in U (0..5)
    # v_idx is index in V (0..20)
    # w_idx is index in W (0..13)
    # u_val, v_val, w_val are coefficients
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # A2 B2 part: A2 is A[0,2], A[1,2]. B2 is B[2,0..6].
    # Products: A[0,2]*B[2,j] -> C[0,j]
    #           A[1,2]*B[2,j] -> C[1,j]
    for j in range(7):
        # C[0,j] gets A[0,2] * B[2,j]
        # U index for A[0,2] is 0*3 + 2 = 2
        # V index for B[2,j] is 2*7 + j = 14 + j
        # W index for C[0,j] is j*2 + 0 = 2j
        add_term(2, 14 + j, 2 * j, 1, 1, 1)
        
        # C[1,j] gets A[1,2] * B[2,j]
        # U index for A[1,2] is 1*3 + 2 = 5
        # V index for B[2,j] is 14 + j
        # W index for C[1,j] is j*2 + 1 = 2j + 1
        add_term(5, 14 + j, 2 * j + 1, 1, 1, 1)
        
    # A1 B1 part
    # A1 is A[0,0..1], A[1,0..1].
    # U indices: 0,1 for row 0; 3,4 for row 1.
    
    # Sub-part 1: A1 B1a1 (cols 0-1 of B1, i.e., B[0..1, 0..1])
    # Strassen 2x2 x 2x2
    # A1 = [[a00, a01], [a10, a11]]
    # B1a1 = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = (a00+a01)b11
    # M4 = a11(b01+b11)
    # M5 = a10(b10+b01)
    # M6 = (a00-a11)(b00+b10)
    # M7 = (a01-a10)(b01+b11)
    
    # Indices in U:
    # a00: 0, a01: 1, a10: 3, a11: 4
    # Indices in V (for B1a1):
    # b00: 0, b01: 1, b10: 7, b11: 8
    # Indices in W (for C1a1):
    # C[0,0]: 0, C[0,1]: 1, C[1,0]: 2, C[1,1]: 3
    
    # M1
    # U: a00(1), a11(1) -> U[0]+=1, U[4]+=1
    # V: b00(1), b11(1) -> V[0]+=1, V[8]+=1
    # W: C[0,0] += 1, C[0,1] += 1, C[1,0] += 1, C[1,1] -= 1 ?
    # Strassen formula:
    # C00 = M1 + M4 - M3 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M6
    # Wait, let's verify Strassen.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = (A00+A01)B11
    # M4 = A11(B01+B11)
    # M5 = A10(B10+B01)
    # M6 = (A00-A11)(B00+B10)
    # M7 = (A01-A10)(B01+B11)
    
    # C00 = M1 - M3 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M6
    
    # Let's implement these 7 terms.
    # Term M1
    add_term(0, 0, 0, 1, 1, 1) # C00
    add_term(0, 0, 1, 1, 1, 0) # C01? No, C01 doesn't use M1 directly in this form?
    # Let's map carefully.
    # M1 contributes to C00 (+1) and C11 (-1).
    # My add_term adds one scalar product.
    # For M1, U vector is e_a00 + e_a11. V vector is e_b00 + e_b11.
    # This is one rank-1 term in the tensor decomposition?
    # No, the tensor decomposition requires U, V, W to be vectors of length rank.
    # Each column c corresponds to one product $u_c^T a \cdot v_c^T b \cdot w_c^T$.
    # So for M1, we need to construct u, v, w such that $u^T a v^T b w^T$ is not a scalar, but adds to the matrix C.
    # Wait. The standard form is $C = \sum_{r=1}^R u_r a \otimes v_r b \otimes w_r^T$?
    # No, $C_{ij} = \sum_{r=1}^R U_{i?, r} V_{j?, r} W_{k?, r}$?
    # The interface is:
    # U: (n*m, rank). U[i*m+j, r]
    # V: (m*p, rank). V[j*p+k, r]
    # W: (p*n, rank). W[k*n+i, r]
    # $C_{i,k} = \sum_{j=0}^{m-1} A_{i,j} B_{j,k} = \sum_{r=1}^R (\sum_j U_{i*m+j, r} A_{i,j}) (\sum_j V_{j*p+k, r} B_{j,k}) W_{k*n+i, r}$?
    # No, the standard tensor decomposition of matrix multiplication is:
    # $T_{i,j,k} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}$? No.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # This is a 3-tensor $T$ with indices $i, j, k$?
    # Actually, the bilinear map is defined by a tensor $T \in \mathbb{R}^{n \times m \times p \times n \times p}$? No.
    # The tensor is $T \in \mathbb{R}^{(n \times m) \times (m \times p) \times (n \times p)}$.
    # $T_{(i,j), (j',k), (i',k')} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}$.
    # We want $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
    # $u_r$ is a vector of length $nm$. $u_r[i*m+j] = U_{i*m+j, r}$.
    # $v_r$ is a vector of length $mp$. $v_r[j*p+k] = V_{j*p+k, r}$.
    # $w_r$ is a vector of length $np$. $w_r[k*n+i] = W_{k*n+i, r}$.
    # The condition is:
    # $\sum_{r=1}^R U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = \delta_{i,i'} \delta_{j,j'} \delta_{k,k'}$?
    # No, the output is $C_{i,k}$.
    # $C_{i,k} = \sum_{r=1}^R W_{k*n+i, r} \sum_{j=0}^{m-1} U_{i*m+j, r} A_{i,j} V_{j*p+k, r} B_{j,k}$?
    # This formulation implies that for a fixed r, the contribution to $C_{i,k}$ is $W_{k*n+i, r} \times (\text{dot product of some A and B})$.
    # But $U_{i*m+j, r}$ depends on i. $V_{j*p+k, r}$ depends on k.
    # So for a fixed r, the term added to $C_{i,k}$ is $W_{k*n+i, r} \sum_{j} U_{i*m+j, r} A_{i,j} V_{j*p+k, r} B_{j,k}$.
    # This is not a simple product of linear forms in A and B unless U and V are sparse in a specific way.
    # Specifically, if for a given r, $U_{i*m+j, r}$ is non-zero for only one i and one j, say $i_0, j_0$, and $V_{j*p+k, r}$ is non-zero for only one j and one k, say $j_1, k_0$, and $i_0=i_1$?
    # In the schoolbook case:
    # r corresponds to (i, j, k).
    # U[i*m+j, r] = 1.
    # V[j*p+k, r] = 1.
    # W[k*n+i, r] = 1.
    # Sum over r: $\sum_{i,j,k} \delta_{U, i,j} \delta_{V, j,k} \delta_{W, k,i} A_{i,j} B_{j,k}$.
    # For a fixed output (i, k), the term is non-zero only if r matches (i, j, k).
    # Then $U_{i*m+j, r} = 1$ (only if i matches), $V_{j*p+k, r} = 1$ (only if k matches).
    # Then the sum over j is just $A_{i,j} B_{j,k}$.
    # Multiplied by $W_{k*n+i, r} = 1$.
    # So it works.
    
    # In Strassen, the U and V vectors have multiple non-zeros.
    # For M1 = (A00+A11)(B00+B11):
    # We need a term r such that:
    # $U_{0, r}=1, U_{3, r}=1$ (for A00, A11) -- wait, A indices are (0,0), (1,1).
    # U index for A00 is 0. U index for A11 is 1*2+1 = 3? No, m=2.
    # A is 2x2. U size 4.
    # A00: 0. A01: 1. A10: 2. A11: 3.
    # So for M1, U[0]=1, U[3]=1.
    # V: B00 (0,0) -> 0. B11 (1,1) -> 1*2+1 = 3.
    # V[0]=1, V[3]=1.
    # W: C00 (0,0) -> 0. C11 (1,1) -> 1*2+1 = 3.
    # Contribution to C00: W[0] * (A00+A11) * (B00+B11).
    # Contribution to C11: W[3] * (A00+A11) * (B00+B11).
    # We want M1 to contribute +1 to C00 and -1 to C11 (based on C00 = M1 - M3 + M4 + M7, C11 = M1 - M2 + M5 + M6).
    # So W[0] = 1, W[3] = -1.
    # Other W entries 0.
    
    # So for M1:
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, -1] (for C00, C01, C10, C11 indices 0,1,2,3)
    
    # I need to map these to the global U, V, W arrays.
    # The sub-problem $2 \times 2 \times 2$ uses specific indices of the global A, B, C.
    
    # Let's write a helper to add a Strassen block.
    # Inputs: 
    # a_indices: list of 4 U indices for A00, A01, A10, A11.
    # b_indices: list of 4 V indices for B00, B01, B10, B11.
    # c_indices: list of 4 W indices for C00, C01, C10, C11.
    # Note: W index order is k*n + i. For 2x2, C00(0,0)->0, C01(0,1)->1, C10(1,0)->2, C11(1,1)->3.
    
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # M1
        U[a_idx[0]][col] = 1; U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1; V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[3]][col] = -1
        col += 1
        # M2
        U[a_idx[2]][col] = 1; U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[2]][col] = 1; W[c_idx[3]][col] = -1
        col += 1
        # M3
        U[a_idx[0]][col] = 1; U[a_idx[1]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = -1; W[c_idx[1]][col] = 1
        col += 1
        # M4
        U[a_idx[3]][col] = 1
        V[b_idx[1]][col] = 1; V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[2]][col] = 1
        col += 1
        # M5
        U[a_idx[2]][col] = 1
        V[b_idx[2]][col] = 1; V[b_idx[1]][col] = 1
        W[c_idx[1]][col] = 1; W[c_idx[3]][col] = 1
        col += 1
        # M6
        U[a_idx[0]][col] = 1; U[a_idx[3]][col] = -1
        V[b_idx[0]][col] = 1; V[b_idx[2]][col] = 1
        W[c_idx[3]][col] = 1
        col += 1
        # M7
        U[a_idx[1]][col] = 1; U[a_idx[2]][col] = -1
        V[b_idx[1]][col] = 1; V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1; W[c_idx[3]][col] = 1
        col += 1

    # Schoolbook for 2x2 x 1
    def add_schoolbook_2x2_1(a_idx, b_idx, c_idx):
        # a_idx: [A00, A01, A10, A11]
        # b_idx: [B00, B01, B10, B11] (only B00, B10 used? No, B is 2x1. So B0, B1)
        # b_idx should be [B0, B1]
        # c_idx: [C00, C10]
        nonlocal col
        # C00 = A00 B0 + A10 B1
        # C10 = A00 B1? No.
        # A = [[a00, a01], [a10, a11]]. B = [[b0], [b1]].
        # C = [[a00 b0 + a10 b1], [a00 b1? No.
        # C00 = a00 b0 + a10 b1? No.
        # C is 2x1.
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0].
        # So:
        # Term 1: A00 B0 -> C00
        U[a_idx[0]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        # Term 2: A01 B1 -> C00
        U[a_idx[1]][col] = 1
        V[b_idx[1]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1
        # Term 3: A10 B0 -> C10
        U[a_idx[2]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[1]][col] = 1
        col += 1
        # Term 4: A11 B1 -> C10
        U[a_idx[3]][col] = 1
        V[b_idx[1]][col] = 1
        W[c_idx[1]][col] = 1
        col += 1

    # Global Indices
    # A is 2x3.
    # A1 (2x2) uses cols 0,1.
    # A1 A00: 0, A01: 1, A10: 3, A11: 4.
    # A2 (2x1) uses col 2.
    # A2 A00: 2, A10: 5.
    
    # B is 3x7.
    # B1 (2x7) uses rows 0,1.
    # B2 (1x7) uses row 2.
    
    # B1a (2x3) uses cols 0,1,2 of B1.
    # B1a1 (2x2) uses cols 0,1 of B1.
    # B00: 0, B01: 1, B10: 7, B11: 8.
    # B1a2 (2x1) uses col 2 of B1.
    # B00: 2, B10: 9.
    
    # B1b (2x4) uses cols 3,4,5,6 of B1.
    # B1b1 (2x2) uses cols 3,4.
    # B00: 3, B01: 4, B10: 10, B11: 11.
    # B1b2 (2x2) uses cols 5,6.
    # B00: 5, B01: 6, B10: 12, B11: 13.
    
    # C is 7x2.
    # W index: k*n + i = k*2 + i.
    # C1a1 (2x2) uses cols 0,1.
    # C00: 0, C01: 1, C10: 2, C11: 3.
    # C1a2 (2x1) uses col 2.
    # C00: 4, C10: 5.
    # C1b1 (2x2) uses cols 3,4.
    # C00: 6, C01: 7, C10: 8, C11: 9.
    # C1b2 (2x2) uses cols 5,6.
    # C00: 10, C01: 11, C10: 12, C11: 13.

    # Execute
    # 1. A2 B2 (14 terms)
    for j in range(7):
        add_term(2, 14 + j, 2 * j, 1, 1, 1)
        add_term(5, 14 + j, 2 * j + 1, 1, 1, 1)
        
    # 2. A1 B1a1 (7 terms)
    a_idx_1 = [0, 1, 3, 4]
    b_idx_1 = [0, 1, 7, 8]
    c_idx_1 = [0, 1, 2, 3]
    add_strassen(a_idx_1, b_idx_1, c_idx_1)
    
    # 3. A1 B1a2 (4 terms)
    b_idx_2 = [2, 9]
    c_idx_2 = [4, 5]
    add_schoolbook_2x2_1(a_idx_1, b_idx_2, c_idx_2)
    
    # 4. A1 B1b1 (7 terms)
    b_idx_3 = [3, 4, 10, 11]
    c_idx_3 = [6, 7, 8, 9]
    add_strassen(a_idx_1, b_idx_3, c_idx_3)
    
    # 5. A1 B1b2 (7 terms)
    b_idx_4 = [5, 6, 12, 13]
    c_idx_4 = [10, 11, 12, 13]
    add_strassen(a_idx_1, b_idx_4, c_idx_4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
