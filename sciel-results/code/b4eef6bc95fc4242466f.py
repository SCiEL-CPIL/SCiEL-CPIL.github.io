"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for matrix multiplication of size 3x5 * 5x5.
    Uses a recursive block splitting strategy. The base case is Strassen's
    algorithm for 2x2x2 multiplication (rank 7). Larger blocks are split
    recursively until they reach a size where Strassen applies or a simple
    threshold is met.
    
    The decomposition follows the convention:
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)  with indices W[k*n+i, r]
    
    The product is sum_r U[i*m+j, r] * V[j*p+k, r] for each output C[i,k].
    This corresponds to the tensor T_{i,j,k} = U_{i*m+j, r} V_{j*p+k, r} ... wait.
    Let's check the tensor definition.
    Matrix multiplication C = AB where A is n x m, B is m x p.
    C[i, k] = sum_j A[i, j] B[j, k].
    The tensor T_{i, j, k} has rank 1 if it corresponds to a term.
    A rank-r decomposition is T_{i,j,k} = sum_{r=1}^R U_{i*m+j, r} V_{j*p+k, r} W_{...} ?
    No, the standard bilinear form for matrix multiplication is:
    C_{ik} = sum_j A_{ij} B_{jk}.
    The tensor is T_{ijk} = delta? No.
    The task says: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
    And "exactly decompose the matrix multiplication tensor".
    Usually, this means:
    sum_r U_{i*m+j, r} * V_{j*p+k, r} * (something related to W) ?
    No, the tensor for matrix multiplication is usually defined such that the bilinear map is:
    (A, B) -> sum_r (sum_{i,j} U_{i*m+j, r} A_{i,j}) (sum_{j,k} V_{j*p+k, r} B_{j,k}) ... and this scalar is added to W?
    Actually, the standard definition for these tasks (like Matrix Multiplication Rank) is:
    We want to compute C_{ik} = sum_j A_{ij} B_{jk}.
    The decomposition provides scalars t_r = (A, U_r) (B, V_r).
    Then C_{ik} = sum_r W_{k*n+i, r} t_r.
    So, C_{ik} = sum_r W_{k*n+i, r} [ sum_{i',j'} U_{i'*m+j', r} A_{i',j'} ] [ sum_{j',k'} V_{j'*p+k', r} B_{j',k'} ].
    For this to equal sum_j A_{ij} B_{jk}, we need:
    sum_r W_{k*n+i, r} U_{i'*m+j', r} V_{j'*p+k', r} A_{i',j'} B_{j',k'} = delta_{i,i'} delta_{k,k'} A_{ij} B_{jk} ... wait.
    The coefficient of A_{i',j'} B_{j',k'} in the LHS is sum_r W_{k*n+i, r} U_{i'*m+j', r} V_{j'*p+k', r}.
    This must be equal to 1 if i'=i, k'=k, and j'=j (summed over j)?
    Actually, the tensor T_{i, j, k} is defined such that the multiplication is:
    C_{ik} = sum_{j} T_{i, j, k} A_{ij} B_{jk} ? No, T is fixed.
    The tensor for matrix multiplication of size n x m x p is usually the structure tensor.
    The rank of this tensor is the bilinear complexity.
    The decomposition is valid if:
    For all i, k:
    sum_j delta_{i, i'} delta_{k, k'} ... 
    Let's look at the schoolbook example in the prompt:
    U[i*m+j][c] = 1
    V[j*p+k][c] = 1
    W[k*n+i][c] = 1
    for each (i,j,k).
    Term r corresponds to (i,j,k).
    t_r = A_{ij} B_{jk}.
    C_{ik} contribution from r is W[k*n+i, r] * A_{ij} B_{jk}.
    Since W[k*n+i, r] = 1 only if the row index matches k*n+i.
    For a fixed (i,j,k), the term is added to C[i,k] with coefficient 1.
    Summing over all (i,j,k), for a fixed output (i,k), the sum is over j.
    Term (i,j,k) contributes A_{ij} B_{jk} to C_{ik}.
    So sum_j A_{ij} B_{jk} = C_{ik}. Correct.
    
    So the condition is:
    sum_r W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r] = 
    1 if i'=i, k'=k (and j' is summed? No, the LHS is a function of i', j', k').
    Wait, the LHS depends on i', j', k'. The RHS must be independent of j'?
    No, the product is bilinear.
    LHS_{i', j', k'} = sum_r W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r].
    This must equal 1 if i'=i and k'=k, for ANY j'? No.
    The output C_{ik} is computed.
    C_{ik} = sum_{i',j',k'} LHS_{i',j',k'} A_{i',j'} B_{j',k'}?
    No.
    Let's re-evaluate:
    C_{ik} = sum_r W[k*n+i, r] t_r
    t_r = (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    C_{ik} = sum_r W[k*n+i, r] (sum_{i',j'} U[i'*m+j', r] A[i',j']) (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    C_{ik} = sum_{i',j',k'} A[i',j'] B[j',k'] (sum_r W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r])
    For this to be equal to sum_{j} A[i,j] B[j,k], we need:
    sum_r W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r] = 1 if i'=i, k'=k, and j'=j?
    No, the sum is over j' in the middle.
    The term A[i',j'] B[j',k'] appears in the expansion.
    We need the coefficient of A[i,j] B[j,k] to be 1 for the specific j, and 0 for others?
    Actually, for a fixed i, k, the sum over j' in the expansion of C_{ik} must yield the correct result.
    The expression is:
    C_{ik} = sum_{j} A[i,j] B[j,k]
    The decomposition gives:
    C_{ik} = sum_{i',j',k'} A[i',j'] B[j',k'] M_{i',j',k'}
    where M_{i',j',k'} = sum_r W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r].
    We need M_{i',j',k'} to be such that sum_{i',j',k'} A[i',j'] B[j',k'] M_{i',j',k'} = sum_{j} A[i,j] B[j,k].
    This implies M_{i',j',k'} must be 1 if i'=i, k'=k, and j'=j?
    Wait, if M_{i',j',k'} = 1 when i'=i, k'=k, regardless of j', then:
    sum_{j'} A[i,j'] B[j',k] = C_{ik}.
    So the condition is:
    sum_r W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r] = 1 if i'=i and k'=k, and 0 otherwise.
    Note that j' is summed over in the matrix multiplication, so the condition does not depend on j' on the RHS?
    Actually, if i'=i and k'=k, the sum over j' in the full expansion is over all j'.
    So yes, M_{i,j',k} must be 1 for all j' if i'=i, k'=k.
    And M_{i',j',k'} = 0 if i' != i or k' != k.
    
    So the constraint is:
    For all i, k:
    For all i', j', k':
    sum_r W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r] = 1 if (i'==i and k'==k) else 0.
    """
    n, m, p = DIMENSIONS
    
    def get_decomp(n, m, p, offset_U=0, offset_V=0, offset_W=0):
        """
        Returns U, V, W matrices for block multiplication of size n x m x p.
        U shape: (n*m, rank)
        V shape: (m*p, rank)
        W shape: (p*n, rank)
        """
        # Base case: 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(4)]
            
            # Strassen terms
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A10)B11
            # M6 = (A11-A01)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Indices in U: i*m+j -> 0:00, 1:01, 2:10, 3:11
            # Indices in V: j*p+k -> 0:00, 1:01, 2:10, 3:11
            # Indices in W: k*n+i -> 0:00(k0i0), 1:10(k1i0), 2:01(k0i1), 3:11(k1i1)
            
            # Term 0: M1
            U[0][0] += 1; U[3][0] += 1
            V[0][0] += 1; V[3][0] += 1
            W[0][0] += 1; W[3][0] += 1
            
            # Term 1: M2
            U[2][1] += 1; U[3][1] += 1
            V[0][1] += 1
            W[2][1] += 1; W[3][1] += 1
            
            # Term 2: M3
            U[0][2] += 1
            V[2][2] += 1; V[3][2] -= 1
            W[1][2] += 1; W[3][2] += 1
            
            # Term 3: M4
            U[3][3] += 1
            V[1][3] += 1; V[0][3] -= 1
            W[0][3] += 1; W[2][3] += 1
            
            # Term 4: M5
            U[0][4] += 1; U[2][4] += 1
            V[3][4] += 1
            W[0][4] -= 1; W[1][4] += 1
            
            # Term 5: M6
            U[3][5] += 1; U[1][5] -= 1
            V[0][5] += 1; V[2][5] += 1
            W[3][5] += 1
            
            # Term 6: M7
            U[1][6] += 1; U[3][6] -= 1
            V[1][6] += 1; V[3][6] += 1
            W[0][6] += 1; W[1][6] += 1
            
            return U, V, W, rank

        # Recursive Case
        # We need to split n, m, p.
        # Strategy: Split the largest dimension? Or balanced?
        # Let's try splitting n into n1+n2, m into m1+m2, p into p1+p2.
        # We want to minimize the sum of ranks.
        # For simplicity, let's try a few splits and pick the best.
        
        best_rank = n*m*p
        best_U = None
        best_V = None
        best_W = None
        
        # Try splits
        # n, m, p are at least 2 (since 2x2x2 is base).
        # If any dimension is 1, it's just dot products or similar, can handle separately or split 1 into 1+0?
        # But n,m,p >= 2 in our recursion if we start from 3,5,5?
        # 3 = 1+2. 5 = 2+3.
        # Let's handle 1s by not splitting them if possible, or splitting 1 into 1+0 (rank contribution 0 for 0 part).
        # Actually, if n=1, it's a vector-matrix product. Rank m*p.
        # If m=1, rank n*p.
        # If p=1, rank n*m.
        
        # Let's generate candidate splits.
        splits = []
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            # Split m
            for m1 in range(1, m):
                m2 = m - m1
                # Split p
                for p1 in range(1, p):
                    p2 = p - p1
                    splits.append((n1, n2, m1, m2, p1, p2))
                    
        # If any dimension is 1, we should also consider not splitting it?
        # If n=1, we can't split into 1+0? Or 0+1?
        # Let's allow 0 splits if it helps? No, 0 size matrices are tricky.
        # Better: if n==1, m==1, p==1, rank 1.
        # If n==1, m>1, p>1, we can treat it as a sum of m*p terms? Rank m*p.
        # Or split m or p.
        # Let's just add cases where we don't split a dimension if it's 1.
        # Actually, the loop range(1, n) fails if n==1.
        
        # Handle base cases for 1s explicitly to avoid infinite recursion or no splits.
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W, 1
            
        if n == 1:
            # Vector A (1x1) * Matrix B (1xp) ? No, A is 1xm, B is mxp.
            # C is 1xp.
            # C_k = sum_j A_j B_jk.
            # Rank is m*p.
            rank = m * p
            U = [[0]*rank for _ in range(m)] # 1*m
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p)] # p*1
            col = 0
            for j in range(m):
                for k in range(p):
                    U[j][col] = 1
                    V[j*p+k][col] = 1
                    W[k][col] = 1
                    col += 1
            return U, V, W, rank

        if m == 1:
            # A nx1, B 1xp. C nxp.
            # C_ik = A_i B_k.
            # Rank n*p.
            rank = n * p
            U = [[0]*rank for _ in range(n)]
            V = [[0]*rank for _ in range(p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i][col] = 1
                    V[k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
            return U, V, W, rank

        if p == 1:
            # A nxm, B mx1. C nx1.
            # C_i = sum_j A_ij B_j.
            # Rank n*m.
            rank = n * m
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m)]
            W = [[0]*rank for _ in range(n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i*m+j][col] = 1
                    V[j][col] = 1
                    W[i][col] = 1
                    col += 1
            return U, V, W, rank

        # Now n, m, p >= 2.
        # Generate splits.
        # To save time, limit the number of splits tested.
        # Or just test all.
        # 3, 5, 5.
        # n splits: 1+2
        # m splits: 1+4, 2+3
        # p splits: 1+4, 2+3
        # Total 1 * 2 * 2 = 4 splits for n=3?
        # Wait, n=3: n1=1, n2=2. Only one split.
        # m=5: m1=1..4.
        # p=5: p1=1..4.
        # Total 16 splits.
        # Each split requires 4 sub-decompositions.
        # Depth: 3->(1,2), 5->(2,3).
        # 1x2x2 -> base or 1-case.
        # 2x2x2 -> base.
        # 2x3x3 -> split 2,3,3.
        # It's small enough.
        
        import sys
        sys.setrecursionlimit(1000)
        
        min_rank = float('inf')
        best_decomp = None
        
        for n1, n2, m1, m2, p1, p2 in splits:
            # Recurse
            U11, V11, W11, r11 = get_decomp(n1, m1, p1)
            U12, V12, W12, r12 = get_decomp(n1, m2, p2) # Wait, B block is m2 x p1?
            # Let's define blocks properly.
            # A = [ A11 (n1xm1)  A12 (n1xm2) ]
            #     [ A21 (n2xm1)  A22 (n2xm2) ]
            # B = [ B11 (m1xp1)  B12 (m1xp2) ]
            #     [ B21 (m2xp1)  B22 (m2xp2) ]
            #
            # C11 (n1xp1) = A11 B11 + A12 B21
            # C12 (n1xp2) = A11 B12 + A12 B22
            # C21 (n2xp1) = A21 B11 + A22 B21
            # C22 (n2xp2) = A21 B12 + A22 B22
            #
            # We need decompositions for:
            # 1. A11 B11 (n1, m1, p1)
            # 2. A12 B21 (n1, m2, p1)
            # 3. A11 B12 (n1, m1, p2)
            # 4. A12 B22 (n1, m2, p2)
            # 5. A21 B11 (n2, m1, p1)
            # 6. A22 B21 (n2, m2, p1)
            # 7. A21 B12 (n2, m1, p2)
            # 8. A22 B22 (n2, m2, p2)
            
            U11, V11, W11, r11 = get_decomp(n1, m1, p1)
            U12, V12, W12, r12 = get_decomp(n1, m2, p1)
            U13, V13, W13, r13 = get_decomp(n1, m1, p2)
            U14, V14, W14, r14 = get_decomp(n1, m2, p2)
            
            U21, V21, W21, r21 = get_decomp(n2, m1, p1)
            U22, V22, W22, r22 = get_decomp(n2, m2, p1)
            U23, V23, W23, r23 = get_decomp(n2, m1, p2)
            U24, V24, W24, r24 = get_decomp(n2, m2, p2)
            
            total_rank = r11 + r12 + r13 + r14 + r21 + r22 + r23 + r24
            
            if total_rank < min_rank:
                min_rank = total_rank
                # Store the components to assemble later
                best_decomp = (
                    (U11, V11, W11, r11, n1, m1, p1),
                    (U12, V12, W12, r12, n1, m2, p1),
                    (U13, V13, W13, r13, n1, m1, p2),
                    (U14, V14, W14, r14, n1, m2, p2),
                    (U21, V21, W21, r21, n2, m1, p1),
                    (U22, V22, W22, r22, n2, m2, p1),
                    (U23, V23, W23, r23, n2, m1, p2),
                    (U24, V24, W24, r24, n2, m2, p2)
                )
        
        if min_rank == float('inf'):
            # Fallback to schoolbook
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
            return U, V, W, rank

        # Assemble the best decomposition
        r11, r12, r13, r14, r21, r22, r23, r24 = [d[3] for d in best_decomp]
        total_rank = min_rank
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # Helper to place a block decomposition
        def place_block(U_local, V_local, W_local, rank_block, 
                        i_start, j_start, k_start, # offsets for U, V, W blocks in global
                        offset_col, 
                        n_block, m_block, p_block,
                        i_offset_in_C, k_offset_in_C):
            """
            Places the local decomposition into global U, V, W.
            i_start, j_start: top-left corner of U block (row i*m + j)
            j_start, k_start: top-left corner of V block (row j*p + k)
            k_start, i_start: top-left corner of W block (row k*n + i)
            """
            for r in range(rank_block):
                col = offset_col + r
                # U block
                for i in range(n_block):
                    for j in range(m_block):
                        U[i_start + i*m_block + j][col] += U_local[i*m_block + j][r]
                # V block
                for j in range(m_block):
                    for k in range(p_block):
                        V[j_start + j*p_block + k][col] += V_local[j*p_block + k][r]
                # W block
                for k in range(p_block):
                    for i in range(n_block):
                        W[k_start + k*n_block + i][col] += W_local[k*n_block + i][r]

        # 1. A11 B11 -> C11
        # U offset: i=0, j=0. V offset: j=0, k=0. W offset: k=0, i=0.
        place_block(best_decomp[0][0], best_decomp[0][1], best_decomp[0][2], r11,
                    0, 0, 0, 0, 
                    n1, m1, p1, 0, 0)
        
        # 2. A12 B21 -> C11
        # U: i=0, j=m1. V: j=m1, k=0. W: k=0, i=0.
        place_block(best_decomp[1][0], best_decomp[1][1], best_decomp[1][2], r12,
                    0, m1, m1, 0, 
                    n1, m2, p1, 0, 0)
        
        # 3. A11 B12 -> C12
        # U: i=0, j=0. V: j=0, k=p1. W: k=p1, i=0.
        place_block(best_decomp[2][0], best_decomp[2][1], best_decomp[2][2], r13,
                    0, 0, 0, p1, 
                    n1, m1, p2, 0, p1)
        
        # 4. A12 B22 -> C12
        # U: i=0, j=m1. V: j=m1, k=p1. W: k=p1, i=0.
        place_block(best_decomp[3][0], best_decomp[3][1], best_decomp[3][2], r14,
                    0, m1, m1, p1, 
                    n1, m2, p2, 0, p1)
        
        # 5. A21 B11 -> C21
        # U: i=n1, j=0. V: j=0, k=0. W: k=0, i=n1.
        place_block(best_decomp[4][0], best_decomp[4][1], best_decomp[4][2], r21,
                    n1, 0, 0, 0, 
                    n2, m1, p1, n1, 0)
        
        # 6. A22 B21 -> C21
        # U: i=n1, j=m1. V: j=m1, k=0. W: k=0, i=n1.
        place_block(best_decomp[5][0], best_decomp[5][1], best_decomp[5][2], r22,
                    n1, m1, m1, 0, 
                    n2, m2, p1, n1, 0)
        
        # 7. A21 B12 -> C22
        # U: i=n1, j=0. V: j=0, k=p1. W: k=p1, i=n1.
        place_block(best_decomp[6][0], best_decomp[6][1], best_decomp[6][2], r23,
                    n1, 0, 0, p1, 
                    n2, m1, p2, n1, p1)
        
        # 8. A22 B22 -> C22
        # U: i=n1, j=m1. V: j=m1, k=p1. W: k=p1, i=n1.
        place_block(best_decomp[7][0], best_decomp[7][1], best_decomp[7][2], r24,
                    n1, m1, m1, p1, 
                    n2, m2, p2, n1, p1)
                    
        return U, V, W, total_rank

    U, V, W, rank = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
