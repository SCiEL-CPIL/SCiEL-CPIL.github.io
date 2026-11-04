"""Program for matrix multiplication tensor rank minimization (3x4 by 4x6)."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the multiplication into 2x2 blocks of A (rows) and B (cols) where possible.
    # A is 3x4, B is 4x6.
    # We can view A as a block matrix with row blocks of size 2 and 1.
    # We can view B as a block matrix with col blocks of size 2 and 2.
    # Specifically, let's partition A into rows [0:2] and [2:3], and B into cols [0:2], [2:4], [4:6].
    # This gives 3 row-blocks of A (sizes 2, 1) and 3 col-blocks of B (sizes 2, 2, 2).
    # For each pair of row-block $i$ and col-block $j$, we compute the subproduct $C_{ij}$.
    # $C_{ij}$ has size $n_i \times p_j$.
    # The intermediate dimension is $m=4$.
    # So for each pair $(i, j)$, we are multiplying an $n_i \times 4$ matrix by a $4 \times p_j$ matrix.
    # We can use Strassen's algorithm for the $2 \times 2$ cases if $n_i=2$ and $p_j=2$.
    # If $n_i=1$ or $p_j=2$ but $n_i \neq 2$, we might use trivial or other methods.
    
    # Let's refine the partition.
    # A rows: 0,1,2. B cols: 0,1,2,3,4,5.
    # Let's group A rows into $R_0=\{0,1\}$ and $R_1=\{2\}$.
    # Let's group B cols into $C_0=\{0,1\}$, $C_1=\{2,3\}$, $C_2=\{4,5\}$.
    # This creates 6 subproblems:
    # 1. $R_0 \times C_0$: $2 \times 4$ by $4 \times 2$. Rank 7 (Strassen).
    # 2. $R_0 \times C_1$: $2 \times 4$ by $4 \times 2$. Rank 7 (Strassen).
    # 3. $R_0 \times C_2$: $2 \times 4$ by $4 \times 2$. Rank 7 (Strassen).
    # 4. $R_1 \times C_0$: $1 \times 4$ by $4 \times 2$. Rank 4 (Trivial: $1 \cdot 2 \cdot 4 = 8$? No, $1 \times 2$ output, $4$ intermediates. Rank is $1 \cdot 2 \cdot 4$? No.
    #    Wait, for $1 \times m$ by $m \times p$, the rank is $m \cdot p$? No.
    #    $C_{ij} = \sum_k A_{ik} B_{kj}$. For fixed $i$, $C_{i0}, C_{i1}$.
    #    $C_{i0} = A_{i0}B_{00} + A_{i1}B_{10} + A_{i2}B_{20} + A_{i3}B_{30}$.
    #    $C_{i1} = A_{i0}B_{01} + A_{i1}B_{11} + A_{i2}B_{21} + A_{i3}B_{31}$.
    #    This is equivalent to multiplying a $1 \times 4$ vector by a $4 \times 2$ matrix.
    #    The tensor is $\sum_{k=0}^3 \sum_{l=0}^1 a_k b_{kl} c_{l}$.
    #    This can be done with rank 4?
    #    Let's check. We need to compute 2 outputs from 4 intermediates.
    #    Actually, we can just do the standard dot products.
    #    For $C_{i0}$, we need $A_{i0}B_{00}, A_{i1}B_{10}, A_{i2}B_{20}, A_{i3}B_{30}$.
    #    For $C_{i1}$, we need $A_{i0}B_{01}, A_{i1}B_{11}, A_{i2}B_{21}, A_{i3}B_{31}$.
    #    We can reuse $A_{ik}$ terms.
    #    Let's just use the trivial decomposition for these small blocks to be safe, or find a better one.
    #    Trivial rank for $1 \times 4 \times 2$ is $1 \cdot 4 \cdot 2 = 8$? No, the tensor is $\sum_{k,l} A_k B_{kl} C_l$.
    #    The number of terms is $m \cdot p = 4 \cdot 2 = 8$.
    #    Can we do better?
    #    $C_0 = A_0 B_{00} + A_1 B_{10} + A_2 B_{20} + A_3 B_{30}$
    #    $C_1 = A_0 B_{01} + A_1 B_{11} + A_2 B_{21} + A_3 B_{31}$
    #    This is essentially two independent dot products of length 4.
    #    The rank is at least 4?
    #    Actually, we can compute $M_k = A_k (B_{k0} + B_{k1})$? No.
    #    Let's stick to trivial for now to ensure correctness, then optimize if needed.
    #    Wait, $1 \times m \times p$ multiplication.
    #    The tensor is $\sum_{k=1}^m \sum_{l=1}^p a_k b_{kl} c_l$.
    #    This is a sum of $m \cdot p$ rank-1 tensors.
    #    Is the rank exactly $m \cdot p$?
    #    For $m=1, p=1$, rank 1.
    #    For $m=1, p=2$, rank 2.
    #    For $m=2, p=1$, rank 2.
    #    For $m=2, p=2$, rank 4? No, $2 \times 2 \times 2$ is 7. But here $n=1$.
    #    If $n=1$, the output is a row vector.
    #    $C_l = \sum_k A_k B_{kl}$.
    #    This is just matrix-vector multiplication.
    #    The rank of the tensor for $1 \times m \times p$ is $m \cdot p$?
    #    Let's check $m=2, p=2$.
    #    $C_0 = A_0 B_{00} + A_1 B_{10}$
    #    $C_1 = A_0 B_{01} + A_1 B_{11}$
    #    We can compute $M_1 = A_0 (B_{00}+B_{01})$, $M_2 = A_1 (B_{10}+B_{11})$? No, that doesn't help separate $B_{00}$ and $B_{01}$.
    #    Actually, we can compute $M_1 = A_0 B_{00}$, $M_2 = A_0 B_{01}$, $M_3 = A_1 B_{10}$, $M_4 = A_1 B_{11}$. Rank 4.
    #    Can we do it in 3?
    #    If we use Strassen-like ideas?
    #    $M_1 = A_0 (B_{00}+B_{01})$, $M_2 = A_1 (B_{10}+B_{11})$, $M_3 = (A_0+A_1) (B_{00}+B_{10})$? No.
    #    It turns out the rank of $1 \times m \times p$ is $m \cdot p$.
    #    So for $R_1 \times C_j$ ($1 \times 4$ by $4 \times 2$), rank is $4 \cdot 2 = 8$.
    #    There are 3 such blocks ($R_1 \times C_0, R_1 \times C_1, R_1 \times C_2$).
    #    Total rank for $R_1$ blocks: $3 \times 8 = 24$.
    #    Total rank for $R_0$ blocks: $3 \times 7 = 21$.
    #    Total rank = $24 + 21 = 45$.
    #    The trivial rank is $3 \cdot 4 \cdot 6 = 72$.
    #    45 is a significant improvement.
    
    # Let's implement this.
    # We will create the U, V, W matrices.
    # U has shape (n*m, rank) = (12, 45)
    # V has shape (m*p, rank) = (24, 45)
    # W has shape (p*n, rank) = (18, 45)
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a trivial term
    def add_trivial(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1
        
    # Helper to add a Strassen term
    # u_vec: list of (idx, coeff) for U
    # v_vec: list of (idx, coeff) for V
    # w_vec: list of (idx, coeff) for W
    def add_strassen(u_vec, v_vec, w_vec):
        nonlocal col
        for idx, coeff in u_vec:
            U[idx][col] += coeff
        for idx, coeff in v_vec:
            V[idx][col] += coeff
        for idx, coeff in w_vec:
            W[idx][col] += coeff
        col += 1
        
    # Define Strassen 2x2x2 decomposition
    # Indices for U: 0,1,2,3 corresponding to A00, A01, A10, A11
    # Indices for V: 0,1,2,3 corresponding to B00, B01, B10, B11
    # Indices for W: 0,1,2,3 corresponding to C00, C01, C10, C11
    
    # M1 = A00 * (B10 - B00) -> C00 += M1, C10 += M1
    # M2 = (A00 + A01) * B00 -> C00 -= M2, C01 += M2
    # M3 = A11 * (B01 - B11) -> C01 += M3, C11 -= M3
    # M4 = (A10 + A11) * B11 -> C10 -= M4, C11 += M4
    # M5 = (A00 + A10) * (B00 + B01) -> C11 += M5
    # M6 = (A01 - A11) * (B10 + B11) -> C00 += M6
    # M7 = (A00 - A10) * (B00 - B10) -> C11 += M7
    
    # Let's re-verify the standard Strassen formulas to ensure signs are correct.
    # C = AB
    # M1 = (A11 - A21)(B21 - B11)
    # M2 = (A11 + A21)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M7
    
    # My previous mapping was slightly different. Let's stick to the standard one.
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11 (using 0-based, so A00 is (0,0))
    # Wait, A is 2x2.
    # A00 is row 0 col 0. A01 is row 0 col 1.
    # A10 is row 1 col 0. A11 is row 1 col 1.
    # So:
    # M1: (A11 - A21) in standard 1-based is (A[1][1] - A[2][1])? No.
    # Standard 1-based: A11 is top-left. A21 is bottom-left.
    # In 0-based: A[0][0] is top-left. A[1][0] is bottom-left.
    # So M1 uses A[1][1] and A[0][1]? No.
    # Let's map carefully.
    # A11 (std) = A[0][0]
    # A12 (std) = A[0][1]
    # A21 (std) = A[1][0]
    # A22 (std) = A[1][1]
    
    # M1 = (A22 - A21)(B21 - B11) -> (A[1][1] - A[1][0]) (B[1][0] - B[0][0])
    # M2 = (A11 + A21)B11 -> (A[0][0] + A[1][0]) B[0][0]
    # M3 = A11(B12 - B22) -> A[0][0] (B[0][1] - B[1][1])
    # M4 = A22(B21 - B11) -> A[1][1] (B[1][0] - B[0][0])
    # M5 = (A11 + A12)B22 -> (A[0][0] + A[0][1]) B[1][1]
    # M6 = (A21 - A11)(B11 + B12) -> (A[1][0] - A[0][0]) (B[0][0] + B[0][1])
    # M7 = (A12 - A22)(B21 + B22) -> (A[0][1] - A[1][1]) (B[1][0] + B[1][1])
    
    # C11 (std) = C[0][0] = M1 + M4 - M5 + M7
    # C12 (std) = C[0][1] = M3 + M5
    # C21 (std) = C[1][0] = M2 + M6
    # C22 (std) = C[1][1] = M1 - M2 + M3 + M7
    
    # Let's implement this specific set of 7 terms.
    
    def get_strassen_terms(u_base, v_base, w_base):
        # u_base: index of A[0][0] in U
        # v_base: index of B[0][0] in V
        # w_base: index of C[0][0] in W
        
        # Term 1
        u1 = [(u_base + 3, 1), (u_base + 2, -1)] # A[1][1], A[1][0]
        v1 = [(v_base + 2, 1), (v_base + 0, -1)] # B[1][0], B[0][0]
        w1 = [(w_base + 0, 1), (w_base + 3, 1)]  # C[0][0], C[1][1]
        
        # Term 2
        u2 = [(u_base + 0, 1), (u_base + 2, 1)] # A[0][0], A[1][0]
        v2 = [(v_base + 0, 1)]                  # B[0][0]
        w2 = [(w_base + 1, 1), (w_base + 2, 1)] # C[0][1], C[1][0]
        
        # Term 3
        u3 = [(u_base + 0, 1)]                  # A[0][0]
        v3 = [(v_base + 1, 1), (v_base + 3, -1)] # B[0][1], B[1][1]
        w3 = [(w_base + 1, 1), (w_base + 3, 1)] # C[0][1], C[1][1]
        
        # Term 4
        u4 = [(u_base + 3, 1)]                  # A[1][1]
        v4 = [(v_base + 2, 1), (v_base + 0, -1)] # B[1][0], B[0][0]
        w4 = [(w_base + 0, 1), (w_base + 3, -1)] # C[0][0], C[1][1]
        
        # Term 5
        u5 = [(u_base + 0, 1), (u_base + 1, 1)] # A[0][0], A[0][1]
        v5 = [(v_base + 3, 1)]                  # B[1][1]
        w5 = [(w_base + 0, -1), (w_base + 1, 1)] # C[0][0], C[0][1]
        
        # Term 6
        u6 = [(u_base + 2, 1), (u_base + 0, -1)] # A[1][0], A[0][0]
        v6 = [(v_base + 0, 1), (v_base + 1, 1)] # B[0][0], B[0][1]
        w6 = [(w_base + 2, 1)]                  # C[1][0]
        
        # Term 7
        u7 = [(u_base + 1, 1), (u_base + 3, -1)] # A[0][1], A[1][1]
        v7 = [(v_base + 2, 1), (v_base + 3, 1)] # B[1][0], B[1][1]
        w7 = [(w_base + 0, 1), (w_base + 3, 1)] # C[0][0], C[1][1]
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]

    # Process R0 x C0 (2x4 by 4x2)
    # A rows 0,1. B cols 0,1.
    # u_base = 0 * 4 + 0 = 0
    # v_base = 0 * 2 + 0 = 0
    # w_base = 0 * 3 + 0 = 0
    terms = get_strassen_terms(0, 0, 0)
    for u, v, w in terms:
        add_strassen(u, v, w)
        
    # Process R0 x C1 (2x4 by 4x2)
    # A rows 0,1. B cols 2,3.
    # u_base = 0
    # v_base = 2 * 2 + 0 = 4
    # w_base = 2 * 3 + 0 = 6
    terms = get_strassen_terms(0, 4, 6)
    for u, v, w in terms:
        add_strassen(u, v, w)
        
    # Process R0 x C2 (2x4 by 4x2)
    # A rows 0,1. B cols 4,5.
    # u_base = 0
    # v_base = 4 * 2 + 0 = 8
    # w_base = 4 * 3 + 0 = 12
    terms = get_strassen_terms(0, 8, 12)
    for u, v, w in terms:
        add_strassen(u, v, w)
        
    # Process R1 x C0 (1x4 by 4x2)
    # A row 2. B cols 0,1.
    # u_base = 2 * 4 + 0 = 8
    # v_base = 0
    # w_base = 0 * 3 + 2 = 2 (since C index is k*n + i, i=2, k=0,1 -> 0*3+2=2, 1*3+2=5)
    # Trivial: 8 terms
    for k in range(4):
        for l in range(2):
            u_idx = 2 * 4 + k
            v_idx = 0 * 2 + l
            w_idx = l * 3 + 2
            add_trivial(u_idx, v_idx, w_idx)
            
    # Process R1 x C1 (1x4 by 4x2)
    # A row 2. B cols 2,3.
    # u_base = 8
    # v_base = 4
    # w_base = 2 * 3 + 2 = 8
    for k in range(4):
        for l in range(2):
            u_idx = 2 * 4 + k
            v_idx = 2 * 2 + l
            w_idx = (2 + l) * 3 + 2
            add_trivial(u_idx, v_idx, w_idx)
            
    # Process R1 x C2 (1x4 by 4x2)
    # A row 2. B cols 4,5.
    # u_base = 8
    # v_base = 8
    # w_base = 4 * 3 + 2 = 14
    for k in range(4):
        for l in range(2):
            u_idx = 2 * 4 + k
            v_idx = 4 * 2 + l
            w_idx = (4 + l) * 3 + 2
            add_trivial(u_idx, v_idx, w_idx)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
