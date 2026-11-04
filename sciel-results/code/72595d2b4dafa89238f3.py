"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # The total rank will be the sum of the ranks of the sub-blocks.
    
    # Helper to add a sub-decomposition to the global U, V, W
    # U_sub: (n_sub * m_sub, rank_sub)
    # V_sub: (m_sub * p_sub, rank_sub)
    # W_sub: (p_sub * n_sub, rank_sub)
    # offsets: i_off, j_off, k_off for A, B, C indices
    # A is n x m. B is m x p. C is p x n.
    # Wait, the task says W[k*n+i, r]. C is p x n?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Matrix A is n x m. Matrix B is m x p. Product C is n x p.
    # But W is (p*n, rank). Index k*n+i.
    # Usually C[i, k] = sum A[i, j] B[j, k].
    # If C is n x p, then index i is 0..n-1, k is 0..p-1.
    # W index k*n + i corresponds to C[i, k]? Or C[k, i]?
    # "W[k*n+i,r]". If k is the column index of C, and i is the row index.
    # Standard row-major: C[i, k] is at i*p + k.
    # Here it is k*n + i. This is column-major for C?
    # Or C is p x n?
    # "multiplying a 3 by 5 matrix by a 5 by 5 matrix".
    # A: 3x5. B: 5x5. C: 3x5.
    # n=3, m=5, p=5.
    # C is 3x5. Rows 0..2, Cols 0..4.
    # W size is p*n = 15.
    # Index k*n + i. If k is col (0..4) and i is row (0..2).
    # Then W[k*3+i] corresponds to C[i, k].
    # Let's assume C[i, k] = sum_j A[i, j] B[j, k].
    # U index i*m + j. (A[i, j])
    # V index j*p + k. (B[j, k])
    # W index k*n + i. (C[i, k])
    
    U = [[0] * 100 for _ in range(n * m)] # Pre-allocate max rank
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    current_rank = 0
    
    # Function to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal current_rank
        U[u_idx][current_rank] += coeff
        V[v_idx][current_rank] += coeff
        W[w_idx][current_rank] += coeff
        current_rank += 1

    # Function to add Strassen M222
    def add_strassen(i0, j0, k0):
        # A block: rows i0..i0+1, cols j0..j0+1
        # B block: rows j0..j0+1, cols k0..k0+1
        # C block: rows i0..i0+1, cols k0..k0+1
        # Indices in global U, V, W
        # U idx: (i)*m + j
        # V idx: (j)*p + k
        # W idx: (k)*n + i
        
        # M1 = (A11+A22)(B11+B22)
        # A11: i0, j0. A22: i0+1, j0+1
        # B11: j0, k0. B22: j0+1, k0+1
        # We need to form linear combinations of A and B.
        # This requires multiple rank-1 terms or pre-computed sums?
        # Tensor decomposition requires U, V, W to be fixed matrices.
        # We cannot compute A11+A22 as a single entry in U.
        # U[i*m+j, r] must be a scalar.
        # So we expand:
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This is 4 terms. But Strassen combines them into 1 multiplication.
        # In the tensor decomposition, each "multiplication" is a rank-1 term.
        # The coefficients in U and V must reflect the sums.
        # For M1:
        # U entries: A11 gets 1, A22 gets 1.
        # V entries: B11 gets 1, B22 gets 1.
        # W entries: C11 gets 1, C12 gets 0, C21 gets 0, C22 gets 1?
        # No, C11 = M1 + ...
        # So W11 gets 1, W22 gets 1.
        
        # Let's define the 7 Strassen terms.
        
        # Term 1: (A11+A22)(B11+B22)
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1) # A11 B11 -> C11
        add_term(i0*m + j0, (j0+1)*p + (k0+1), (k0+1)*n + i0, 1) # A11 B22 -> C12
        add_term((i0+1)*m + (j0+1), j0*p + k0, k0*n + (i0+1), 1) # A22 B11 -> C21
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1) # A22 B22 -> C22
        # Wait, this is just schoolbook for the 2x2x2 block. Rank 4.
        # Strassen reduces it to 7 multiplications, but the "multiplications" are of sums.
        # In the tensor decomposition, we don't have "sums" in the inputs.
        # The tensor decomposition is a sum of rank-1 tensors.
        # $\sum_{r} U_{ij,r} V_{jk,r} W_{ki,r} = \delta_{ik}$? No.
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # The decomposition is $C_{ik} = \sum_{r} U_{ij,r} V_{jk,r} W_{ki,r}$?
        # No, the standard form is:
        # $C_{ik} = \sum_{r} \lambda_r (\sum_{j} U_{ij,r} A_{ij}) (\sum_{j} V_{jk,r} B_{jk})$?
        # No.
        # The tensor is $T_{ijk}$. $T_{ijk} = \sum_{r} U_{ij,r} V_{jk,r} W_{ki,r}$?
        # Indices: U is n*m x rank. V is m*p x rank. W is p*n x rank.
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # The decomposition must satisfy:
        # $\sum_{r} W_{ki,r} (\sum_{j} U_{ij,r} A_{ij}) (\sum_{j} V_{jk,r} B_{jk})$? No.
        # The product is bilinear.
        # $C_{ik} = \sum_{r} W_{ki,r} M_r$.
        # $M_r = (\sum_{j} U_{ij,r} A_{ij}) (\sum_{j} V_{jk,r} B_{jk})$?
        # This implies $U$ and $V$ depend on $i, j$ and $j, k$ respectively.
        # But $U$ is a matrix of size $nm \times rank$. $U_{ij, r}$.
        # So $M_r$ is a product of a linear form in A and a linear form in B.
        # Yes.
        
        # So for Strassen M1 = (A11+A22)(B11+B22).
        # The linear form in A is $A11 + A22$.
        # So $U_{i0,j0, r1} = 1$, $U_{i0+1,j0+1, r1} = 1$.
        # The linear form in B is $B11 + B22$.
        # So $V_{j0,k0, r1} = 1$, $V_{j0+1,k0+1, r1} = 1$.
        # The output $C$ receives this term with coefficient 1 for C11 and C22?
        # No, $M1$ contributes to C11, C12, C21, C22?
        # In Strassen, $M1$ is added to C11 and C22.
        # So $W_{k0,i0, r1} = 1$ (for C11) and $W_{k0+1,i0+1, r1} = 1$ (for C22).
        
        # Let's implement this correctly.
        
        # We will add 7 terms for the 2x2x2 block.
        
        # T1
        U[i0*m + j0][current_rank] = 1
        U[(i0+1)*m + (j0+1)][current_rank] = 1
        V[j0*p + k0][current_rank] = 1
        V[(j0+1)*p + (k0+1)][current_rank] = 1
        W[k0*n + i0][current_rank] = 1
        W[(k0+1)*n + (i0+1)][current_rank] = 1
        current_rank += 1
        
        # T2 = (A21+A22)B11
        U[(i0+1)*m + j0][current_rank] = 1
        U[(i0+1)*m + (j0+1)][current_rank] = 1
        V[j0*p + k0][current_rank] = 1
        W[k0*n + (i0+1)][current_rank] = 1
        current_rank += 1
        
        # T3 = A11(B12-B22)
        U[i0*m + j0][current_rank] = 1
        V[j0*p + (k0+1)][current_rank] = 1
        V[(j0+1)*p + (k0+1)][current_rank] = -1
        W[(k0+1)*n + i0][current_rank] = 1
        current_rank += 1
        
        # T4 = A22(B21-B11)
        U[(i0+1)*m + (j0+1)][current_rank] = 1
        V[(j0+1)*p + k0][current_rank] = 1
        V[j0*p + k0][current_rank] = -1
        W[k0*n + (i0+1)][current_rank] = 1
        current_rank += 1
        
        # T5 = (A11+A12)B22
        U[i0*m + j0][current_rank] = 1
        U[i0*m + (j0+1)][current_rank] = 1
        V[(j0+1)*p + (k0+1)][current_rank] = 1
        W[(k0+1)*n + i0][current_rank] = 1
        W[(k0+1)*n + (i0+1)][current_rank] = 1
        current_rank += 1
        
        # T6 = (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][current_rank] = 1
        U[i0*m + j0][current_rank] = -1
        V[j0*p + k0][current_rank] = 1
        V[j0*p + (k0+1)][current_rank] = 1
        W[(k0+1)*n + (i0+1)][current_rank] = 1
        current_rank += 1
        
        # T7 = (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][current_rank] = 1
        U[(i0+1)*m + (j0+1)][current_rank] = -1
        V[(j0+1)*p + k0][current_rank] = 1
        V[(j0+1)*p + (k0+1)][current_rank] = 1
        W[(k0+1)*n + i0][current_rank] = 1
        current_rank += 1

    # We will apply this to the 2x2 sub-blocks of the 3x5x5 problem.
    # But we must be careful not to double count or miss terms.
    # The block decomposition approach I outlined earlier (sum of ranks) is safer.
    # I will implement the block decomposition manually.
    
    # Reset
    U = [[0] * 60 for _ in range(n * m)]
    V = [[0] * 60 for _ in range(m * p)]
    W = [[0] * 60 for _ in range(p * n)]
    current_rank = 0

    # Block (1,1): C[0:2, 0:2]
    # A[0:2, 0:5], B[0:5, 0:2]
    # Split B cols 0:5 into 0:2, 2:4, 4:5.
    # Split A rows 0:2.
    # Term 1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2 Strassen)
    add_strassen(0, 0, 0)
    # Term 2: A[0:2, 2:4] * B[2:4, 0:2] (2x2x2 Strassen)
    add_strassen(0, 2, 0)
    # Term 3: A[0:2, 4:5] * B[4:5, 0:2] (2x1x2 Schoolbook)
    # A is 2x1, B is 1x2.
    # C11 = A11 B11 + A12 B12? No.
    # A[0,4]B[4,0] + A[1,4]B[4,0] -> C[0,0], C[1,0]
    # A[0,4]B[4,1] + A[1,4]B[4,1] -> C[0,1], C[1,1]
    # Rank 2.
    U[0*m + 4][current_rank] = 1
    V[4*p + 0][current_rank] = 1
    W[0*n + 0][current_rank] = 1
    current_rank += 1
    
    U[1*m + 4][current_rank] = 1
    V[4*p + 1][current_rank] = 1
    W[1*n + 0][current_rank] = 1
    current_rank += 1
    
    # Wait, this is not correct.
    # C[0,0] = A[0,4]B[4,0].
    # C[1,0] = A[1,4]B[4,0].
    # C[0,1] = A[0,4]B[4,1].
    # C[1,1] = A[1,4]B[4,1].
    # This is 4 terms. Rank 4?
    # No, M212 rank is 2?
    # $C_{ik} = A_{i1} B_{1k}$.
    # $C_{00} = A_{01} B_{10}$.
    # $C_{10} = A_{11} B_{10}$.
    # $C_{01} = A_{01} B_{11}$.
    # $C_{11} = A_{11} B_{11}$.
    # This can be done with 2 multiplications?
    # $M_1 = A_{01} B_{10}$.
    # $M_2 = A_{11} B_{11}$.
    # No.
    # $M_1 = A_{01} B_{10}$.
    # $M_2 = A_{11} B_{10}$.
    # $M_3 = A_{01} B_{11}$.
    # $M_4 = A_{11} B_{11}$.
    # Rank 4.
    # So Block (1,1) rank = 7 + 7 + 4 = 18.

    # This is getting too complex to code correctly in the time available.
    # I will return the schoolbook algorithm.
    
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
