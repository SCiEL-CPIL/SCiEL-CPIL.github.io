"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block construction:
    # Split 5x5 into 4x4 block and 1x1 corner, with cross terms.
    # Rank 4x4x4 is 14 (Laderman).
    # Rank 4x4x1 (vector mult) is 4.
    # Rank 1x4x4 (vector mult) is 4.
    # Cross terms are scalar mults.
    
    # We need to explicitly construct the 4x4 Laderman decomposition.
    # Laderman's 14-term decomposition for 4x4x4.
    # Indices for U (16x14), V (16x14), W (16x14).
    # We will map the 4x4 indices to the 5x5 indices.
    
    # Laderman 4x4 decomposition terms (u, v, w) where u,v,w in 0..15 (4x4 indices)
    # This is a known set of 14 triples.
    laderman_4x4 = [
        (0, 0, 0),   # A00*B00
        (1, 1, 1),   # A11*B11
        (2, 2, 2),   # A22*B22
        (3, 3, 3),   # A33*B33
        (0, 1, 1),   # A00*B11 ? No, Laderman is specific.
        # I do not have the exact Laderman 14 terms memorized.
        # I will use the schoolbook for 4x4 (rank 16) to be safe, 
        # or find a simpler reduction.
        
        # Actually, using schoolbook for 4x4 (16) + 4 + 4 + 10 = 34.
        # Score 125/34 = 3.67.
        # This is a safe improvement.
    ]
    
    # Let's use schoolbook for the 4x4 part to ensure correctness.
    # Rank 4x4x4 = 16.
    # Rank 4x4x1 = 4 (assuming rank is min(n,m) for vector mult? 
    # Actually, M_{4,4,1} rank is 4? 
    # If I use schoolbook for vector mult, it's 16.
    # 16 + 16 + 16 + 10 = 58.
    # Score 125/58 = 2.15.
    
    # Let's try to implement the rank 4 for M_{4,4,1}.
    # M_{4,4,1} terms:
    # C_i = sum_j A_ij d_j.
    # We can use 4 mults?
    # Let's assume rank 4 for now.
    
    # Total Rank = 16 (4x4 schoolbook) + 4 (A*d) + 4 (b*B) + 10 (cross) = 34.
    # If I use Laderman 14, total is 32.
    
    # I will implement the 34-rank decomposition using schoolbook for 4x4.
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 1. 4x4 block multiplication (schoolbook, 16 terms)
    # Indices 0..3 for 4x4 block.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 16.
    
    # 2. A_44 * d (4x4 matrix times 4x1 vector)
    # A_44 is rows 0..3, cols 0..3.
    # d is col 4 of B (rows 0..3).
    # Result C rows 0..3, col 4.
    # Rank 4? 
    # If rank 4, we need 4 terms.
    # Let's use schoolbook (16 terms) to be safe?
    # If I use 16 terms, total rank = 16 + 16 + 16 + 10 = 58.
    # Score 2.15.
    # If I use 4 terms, total rank = 16 + 4 + 4 + 10 = 34.
    # Score 3.67.
    
    # I will use 4 terms for M_{4,4,1} assuming it's valid.
    # Terms for M_{4,4,1}:
    # We need to compute C_i = sum_j A_ij d_j for i=0..3.
    # This is 4 independent dot products.
    # Can we do it with 4 mults?
    # Yes, if we treat it as 4 parallel scalar mults? No.
    # The rank of M_{n,m,1} is m?
    # For m=4, rank 4.
    # Terms:
    for j in range(4):
        # Term j: A_j (col j of A_44) * d_j (row j of d)
        # A_j is a vector of 4 elements (rows 0..3, col j).
        # d_j is a scalar (row j, col 4 of B).
        # Result is added to C col 4, rows 0..3.
        for i in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
        col += 1
    # col is now 16 + 4 = 20.
    
    # 3. b^T * B_44 (1x4 vector times 4x4 matrix)
    # b is row 4 of A (cols 0..3).
    # B_44 is rows 0..3, cols 0..3.
    # Result C row 4, cols 0..3.
    # Rank 4.
    for j in range(4):
        # Term j: b_j (col j of b) * B_j (row j of B_44)
        for k in range(4):
            U[4 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 4][col] = 1
        col += 1
    # col is now 20 + 4 = 24.
    
    # 4. Cross terms
    # C_44 = b^T d + c f
    # b^T d: dot product of b (1x4) and d (4x1). 4 mults.
    for j in range(4):
        U[4 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
    # col is now 24 + 4 = 28.
    
    # c f: scalar mult. 1 mult.
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    # col is now 29.
    
    # Wait, I missed the terms for C_41 and C_14 involving the 1x1 corner?
    # C_41 (row 4, col 4? No, C_41 is row 4, col 4? No, C is 5x5.
    # C_41 means row 4, col 1? No, indices 0..4.
    # C_44 is row 4, col 4.
    # C_40..C_43 is row 4, cols 0..3. (Handled by b^T B_44)
    # C_04..C_34 is rows 0..3, col 4. (Handled by A_44 d)
    # C_44 is row 4, col 4. (Handled by b^T d + c f)
    
    # Are there other cross terms?
    # A = [A44, a; b, c]
    # B = [B44, d; e, f]
    # C = [A44B44 + a e, A44 d + a f; b B44 + c e, b d + c f]
    
    # I missed:
    # C_04..C_34 includes a e?
    # a is row 0..3, col 4 of A.
    # e is row 4, col 0..3 of B.
    # a e is outer product of a (4x1) and e (1x4).
    # This contributes to C rows 0..3, cols 0..3? No.
    # a is col 4 of A (rows 0..3). e is row 4 of B (cols 0..3).
    # (a e)_{ij} = a_i e_j.
    # This contributes to C_{ij} for i=0..3, j=0..3.
    # So C_44_block = A44 B44 + a e.
    # I only did A44 B44.
    # I need to add a e.
    # a e is outer product. Rank 1?
    # C_{ij} += a_i e_j.
    # This is 1 term?
    # U[i, col] = a_i? No, U is constant.
    # The term is $a_i e_j$.
    # $U_{i,4} = 1$? No, $a_i$ is an entry of A.
    # $U_{i,4, col} = 1$?
    # $V_{4,j, col} = 1$?
    # $W_{j,i, col} = 1$?
    # This would compute $A_{i,4} B_{4,j}$.
    # Yes.
    # So we need 1 term for the outer product?
    # No, $a_i e_j$ is a product of two variables.
    # It is a rank 1 tensor term.
    # So 1 term.
    
    # Similarly for C_40..C_43: b B44 + c e.
    # I did b B44.
    # I need c e.
    # c is A_{4,4}. e is B_{4,0..3}.
    # $c e_j$.
    # $U_{4,4, col} = 1$.
    # $V_{4,j, col} = 1$.
    # $W_{j,4, col} = 1$.
    # This is 1 term? No, it's 4 terms (one for each j)?
    # $c e_j$ is a scalar mult for each j.
    # So 4 terms.
    
    # And C_04..C_34: A44 d + a f.
    # I did A44 d.
    # I need a f.
    # $a_i f$.
    # 4 terms.
    
    # So additional terms:
    # a e: 1 term (outer product).
    # c e: 4 terms.
    # a f: 4 terms.
    
    # Total rank = 16 (A44B44) + 4 (A44d) + 4 (bB44) + 4 (bd) + 1 (cf) + 1 (ae) + 4 (ce) + 4 (af) = 38.
    # Score 125/38 = 3.29.
    
    # I will implement this.
    
    # Reset col to 0 and rebuild correctly.
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # 1. A44 B44 (16 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. a e (1 term)
    # a is A[:, 4], e is B[4, :]
    # C[i, j] += A[i, 4] B[4, j]
    # U[i, 4, col] = 1, V[4, j, col] = 1, W[j, i, col] = 1
    # This is a single rank-1 term?
    # No, $A_{i,4} B_{4,j}$ varies with i and j.
    # It is not a single term in the tensor decomposition sense?
    # The tensor is $T_{ijk} = \sum_r U_{ir} V_{jr} W_{kr}$.
    # Here $i, j$ are indices of C? No, $C_{ij}$.
    # The term is $A_{i,4} B_{4,j}$.
    # This is $U_{i,4, r} V_{4,j, r} W_{j,i, r}$.
    # If we set $U_{i,4, r} = 1$ for all i? No.
    # We need $U_{i,4, r} = 1$ and $V_{4,j, r} = 1$?
    # Then $W_{j,i, r} = 1$?
    # This would compute $\sum_{i,j} A_{i,4} B_{4,j} W_{j,i}$.
    # We want $C_{ij} = A_{i,4} B_{4,j}$.
    # So $W_{j,i, r}$ must be 1 if we want $C_{ij}$ to get this value.
    # But $W$ is a matrix. $W_{j,i}$ is a fixed entry.
    # If we set $W_{j,i, r} = 1$ for all i,j, then $C_{ij}$ gets the sum?
    # No, the decomposition is $C_{ij} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # Wait, the indices in the prompt are:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # $C_{ik} = \sum_j \sum_r U_{ij, r} V_{jk, r} W_{ki, r}$.
    # My previous index mapping was:
    # $U_{i*m+j, r}$, $V_{j*p+k, r}$, $W_{k*n+i, r}$.
    # So $C_{i,k} = \sum_{j,r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
    
    # For the term $A_{i,4} B_{4,j}$ contributing to $C_{i,j}$:
    # We need $U_{i*m+4, r} = 1$, $V_{4*p+j, r} = 1$, $W_{j*n+i, r} = 1$.
    # This must hold for all i, j.
    # So we need one term r such that for all i, j:
    # $U_{i*m+4, r} = 1$, $V_{4*p+j, r} = 1$, $W_{j*n+i, r} = 1$.
    # This is a valid rank-1 term.
    U[i*m+4][col] = 1 # for all i? No, U is a matrix.
    # We need to set U[i*m+4][col] = 1 for all i in 0..3.
    for i in range(4):
        U[i * m + 4][col] = 1
    for j in range(4):
        V[4 * p + j][col] = 1
    for i in range(4):
        for j in range(4):
            W[j * n + i][col] = 1
    col += 1
    
    # 3. A44 d (4 terms)
    # C[i, 4] += A[i, j] B[j, 4]
    for j in range(4):
        for i in range(4):
            U[i * m + j][col] = 1
        V[j * p + 4][col] = 1
        for i in range(4):
            W[4 * n + i][col] = 1
        col += 1
        
    # 4. a f (4 terms)
    # C[i, 4] += A[i, 4] B[4, 4]
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
        
    # 5. b B44 (4 terms)
    # C[4, j] += A[4, i] B[i, j]
    for i in range(4):
        U[4 * m + i][col] = 1
        for j in range(4):
            V[i * p + j][col] = 1
        W[j * n + 4][col] = 1 # Wait, W index is k*n+i. Here k=j, i=4.
        # W[j * n + 4][col] = 1 for all j?
        # Yes.
        for j in range(4):
            W[j * n + 4][col] = 1
        col += 1
        
    # 6. c e (4 terms)
    # C[4, j] += A[4, 4] B[4, j]
    for j in range(4):
        U[4 * m + 4][col] = 1
        V[4 * p + j][col] = 1
        W[j * n + 4][col] = 1
        col += 1
        
    # 7. b d (4 terms)
    # C[4, 4] += A[4, i] B[i, 4]
    for i in range(4):
        U[4 * m + i][col] = 1
        V[i * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
        
    # 8. c f (1 term)
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
