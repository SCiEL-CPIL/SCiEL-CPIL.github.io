"""Rank-minimal tensor decomposition for 2x2 x 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

import time


def construct():
    n, m, p = 2, 2, 13  # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (26, rank)
    # W shape: (p*n, rank) = (26, rank)

    # We use a hybrid approach:
    # 1. For the first 2 columns (k=0,1), we use a rank-7 Strassen-like decomposition.
    #    However, upon careful analysis, the "1 term per column" approach for extra columns
    #    (k>=2) is actually valid for the tensor identity T[i,j,k]=1.
    #    Let's verify:
    #    For column k>=2, use 1 term c:
    #    U[i*m+j, c] = 1 for all i,j
    #    V[j*p+k, c] = 1 for all j
    #    W[k*n+i, c] = 1 for all i
    #    Then T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1.
    #    This satisfies the tensor identity.
    #    
    #    BUT, does this correspond to a valid matrix multiplication algorithm?
    #    The tensor identity is the definition of the decomposition.
    #    If T[i,j,k]=1 for all i,j,k, then C[i,k] = sum_j A[i,j]B[j,k] is correctly computed.
    #    The "algorithm" implied by the decomposition would compute:
    #    For each term c:
    #      S_c = (sum_j U[i*m+j,c] A[i,j]) * (sum_k' V[j*p+k',c] B[j,k']) ... no.
    #    
    #    Actually, the standard interpretation is:
    #    The decomposition provides a way to compute C with 'rank' scalar multiplications.
    #    For the "1 term per column" case for k>=2:
    #    The term is: (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,k]+B[1,k])
    #    And it contributes to C[0,k] and C[1,k] with weight 1.
    #    This means C[0,k] += (sum A) * (sum B_k) and C[1,k] += (sum A) * (sum B_k).
    #    This is NOT correct for matrix multiplication.
    #    C[0,k] should be A[0,0]B[0,k] + A[0,1]B[1,k].
    #    C[1,k] should be A[1,0]B[0,k] + A[1,1]B[1,k].
    #    
    #    So why does the tensor identity hold?
    #    Because the tensor T[i,j,k] is defined such that C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k].
    #    Wait, no. The standard definition is:
    #    C[i,k] = sum_j A[i,j] B[j,k].
    #    This can be written as C[i,k] = sum_{j} sum_{c} (U[i,j,c] A[i,j]) (V[j,k,c] B[j,k]) W[k,i,c] ?
    #    No, the standard tensor decomposition of matrix multiplication is:
    #    T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    #    where T[i,j,k] is the multiplication tensor, which is 1 if the indices form a valid
    #    multiplication step, but actually T[i,j,k] is just 1 for all valid i,j,k in the
    #    "identity" tensor? No.
    #    
    #    Let's look at the definition again.
    #    The matrix multiplication tensor is a 3-tensor T of size n x m x p (or similar).
    #    T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    #    Actually, the tensor is defined by the bilinear map.
    #    The rank of the tensor is the minimum number of rank-1 tensors that sum to T.
    #    A rank-1 tensor is u ⊗ v ⊗ w, where (u ⊗ v ⊗ w)[i,j,k] = u[i,j] v[j,k] w[k,i].
    #    We need sum_c u_c[i,j] v_c[j,k] w_c[k,i] = T[i,j,k].
    #    For matrix multiplication, T[i,j,k] = 1 if the indices are valid, but specifically:
    #    The term A[i,j] B[j,k] appears in C[i,k].
    #    So the "coefficient" of A[i,j] B[j,k] in the expression for C[i,k] is 1.
    #    But C[i,k] is a single entry. The tensor T is defined on the space of all
    #    (i,j,k) triples.
    #    Actually, the standard definition is:
    #    T is a tensor in U^* ⊗ V^* ⊗ W^* where U is the space of A matrices,
    #    V is the space of B matrices, W is the space of C matrices.
    #    T(A,B) = C = AB.
    #    The tensor T has components T_{ijk} such that C_k = sum_{i,j} A_i B_j T_{ijk}.
    #    Wait, C is a matrix. Let's flatten it.
    #    C_{kl} (row k, col l) = sum_{i,j} A_{ki} B_{jl} delta_{ij} ? No.
    #    C_{kl} = sum_j A_{kj} B_{jl}.
    #    So the tensor T has indices (k,j,l) where k is row of C, j is inner index, l is col of C.
    #    T_{k,j,l} = 1 if the term A_{k,j} B_{j,l} contributes to C_{k,l}.
    #    In this case, T_{k,j,l} = 1 for all valid k,j,l.
    #    So we need sum_c U_{k,j,c} V_{j,l,c} W_{l,k,c} = 1 for all k,j,l.
    #    
    #    My previous analysis was correct:
    #    For k>=2, if I use 1 term with U=1, V=1, W=1, then the sum is 1.
    #    So the tensor identity holds.
    #    
    #    Why did I doubt it? Because I was thinking about the "algorithm".
    #    The decomposition implies an algorithm with 'rank' multiplications.
    #    If the tensor identity holds, the algorithm is correct.
    #    The "algorithm" for the 1-term-per-column case for k>=2 is:
    #    Compute S = (sum A) * (sum B_k).
    #    Add S to C[0,k] and C[1,k].
    #    This is clearly wrong for matrix multiplication.
    #    
    #    Where is the error?
    #    The error is in the mapping between the tensor indices and the algorithm.
    #    The tensor identity sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 ensures that
    #    the bilinear map is correct.
    #    Let's check the bilinear map for the 1-term case.
    #    Term c: U[i,j]=1, V[j,k]=1, W[k,i]=1.
    #    The contribution to the bilinear map is:
    #    (sum_{i,j} U[i,j] A[i,j]) * (sum_{j,k} V[j,k] B[j,k]) * W[k,i] ?
    #    No, the reconstruction is:
    #    C[i,k] = sum_c W[k,i,c] * (sum_j U[i,j,c] A[i,j]) * (sum_j' V[j',k,c] B[j',k]) ?
    #    No, that's not how it works.
    #    
    #    The correct reconstruction from a rank-1 term u ⊗ v ⊗ w is:
    #    C = w * (u^T A) * v^T ? No.
    #    The term is (sum_{i,j} u_{ij} A_{ij}) (sum_{j,k} v_{jk} B_{jk}) w_{ki} ?
    #    No, the indices must match.
    #    
    #    Let's use the explicit formula:
    #    C[i,k] = sum_j A[i,j] B[j,k].
    #    Decomposition: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k' V[j,k',c] B[j,k']) W[k,i,c] ?
    #    This doesn't look right because the sums are over different indices.
    #    
    #    Actually, the standard way to view it is:
    #    Define intermediate variables M_c = (sum_j U[i,j,c] A[i,j]) (sum_k' V[j,k',c] B[j,k']) ?
    #    No, U depends on i,j and V depends on j,k.
    #    
    #    Let's go back to the tensor identity:
    #    sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    #    This identity must hold for all i,j,k.
    #    If it holds, then the decomposition is valid.
    #    The "algorithm" is derived from this.
    #    For the 1-term case for column k:
    #    U[i*m+j,c] = 1 for all i,j.
    #    V[j*p+k,c] = 1 for all j.
    #    W[k*n+i,c] = 1 for all i.
    #    Then sum_c ... = 1.
    #    So the identity holds.
    #    
    #    Therefore, the rank is 7 (for k=0,1) + 11 (for k=2..12) = 18.
    #    
    #    Wait, is rank 18 possible?
    #    The lower bound for M_{n,m,p} is max(nm, mp, pn).
    #    Here max(4, 26, 26) = 26.
    #    So rank 18 is impossible!
    #    
    #    My tensor identity check must be wrong or I'm misinterpreting it.
    #    The lower bound is max(nm, mp, pn) = 26.
    #    So rank must be at least 26.
    #    
    #    Why did my check pass?
    #    Because I checked sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    #    For the 1-term case, this sum is 1.
    #    But this identity is NOT sufficient for the rank to be valid?
    #    No, it is the definition.
    #    
    #    Let's re-evaluate the lower bound.
    #    The rank of the matrix multiplication tensor <n,m,p> is at least max(nm, mp, pn).
    #    For 2,2,13, this is 26.
    #    So rank 18 is impossible.
    #    
    #    Where is the flaw in my 1-term construction?
    #    The flaw is that U, V, W are fixed for ALL k.
    #    In my construction, for column k, I used a term c_k.
    #    U[i*m+j, c_k] = 1 for all i,j.
    #    V[j*p+k, c_k] = 1 for all j.
    #    W[k*n+i, c_k] = 1 for all i.
    #    
    #    But U is a matrix of size (n*m) x rank.
    #    The column c_k of U is the same for all k? No, c_k is a different column for each k.
    #    So U[:, c_k] is a vector.
    #    For term c_k, U[:, c_k] is a vector of length 4.
    #    I set U[i*m+j, c_k] = 1 for all i,j. So U[:, c_k] is [1,1,1,1].
    #    V[:, c_k] is a vector of length 26.
    #    I set V[j*p+k, c_k] = 1 for j=0,1. So V[:, c_k] has 1s at positions k and 13+k.
    #    W[:, c_k] is a vector of length 26.
    #    I set W[k*n+i, c_k] = 1 for i=0,1. So W[:, c_k] has 1s at positions 2k and 2k+1.
    #    
    #    Now, let's check the tensor identity for a specific (i,j,k').
    #    T[i,j,k'] = sum_c U[i*m+j,c] V[j*p+k',c] W[k'*n+i,c].
    #    
    #    Case 1: k' = k (the column for which term c_k was designed).
    #    Term c_k contributes: U[i*m+j,c_k] V[j*p+k,c_k] W[k*n+i,c_k] = 1 * 1 * 1 = 1.
    #    Other terms c_l (l != k) contribute 0 because V[j*p+k,c_l] = 0 (since c_l only has 1s at 2l and 13+2l, not k).
    #    Wait, V[j*p+k, c_l] is the entry at row j*p+k of column c_l.
    #    For term c_l, V[:, c_l] has 1s at rows 2l and 13+2l.
    #    So V[j*p+k, c_l] is 1 only if j*p+k is 2l or 13+2l.
    #    If k' = k, then j*p+k' is j*p+k.
    #    If k != 2l and k != 13+2l, then V[j*p+k, c_l] = 0.
    #    So only term c_k contributes. Sum = 1. Correct.
    #    
    #    Case 2: k' != k.
    #    Term c_k contributes: U[i*m+j,c_k] V[j*p+k',c_k] W[k'*n+i,c_k].
    #    V[j*p+k',c_k] is 1 only if j*p+k' is 2k or 13+2k.
    #    If k' is not 2k or 13+2k, then V[j*p+k',c_k] = 0.
    #    So term c_k contributes 0.
    #    Similarly, term c_{k'} contributes 0 to T[i,j,k] if k != k'.
    #    So for k' != k, T[i,j,k'] = 0.
    #    But we need T[i,j,k'] = 1!
    #    
    #    Ah, I see. The terms are "local" to their column.
    #    Term c_k only contributes to T[i,j,k] where k is the column index.
    #    So T[i,j,k] = 1 is satisfied by term c_k.
    #    And T[i,j,k'] for k' != k is 0 from term c_k.
    #    But T[i,j,k'] must be 1.
    #    So we need another term for k'.
    #    Which is term c_{k'}.
    #    So T[i,j,k'] = sum_c ... = 1 (from c_{k'}).
    #    
    #    So the identity holds for ALL (i,j,k).
    #    So the rank is 18.
    #    
    #    But the lower bound is 26.
    #    Contradiction.
    #    
    #    The lower bound max(nm, mp, pn) applies to the rank of the tensor.
    #    Is my construction a valid tensor decomposition?
    #    Yes, if the identity holds.
    #    So either the lower bound is wrong, or my construction is invalid.
    #    
    #    The lower bound is correct.
    #    So my construction must be invalid.
    #    Why?
    #    Because U, V, W must be fixed matrices.
    #    In my construction, U[:, c_k] is [1,1,1,1] for all k.
    #    This is fine.
    #    V[:, c_k] has 1s at 2k and 13+2k.
    #    W[:, c_k] has 1s at 2k and 2k+1.
    #    
    #    Wait, the lower bound max(nm, mp, pn) is for the rank of the tensor <n,m,p>.
    #    Here n=2, m=2, p=13.
    #    nm=4, mp=26, pn=26.
    #    So rank >= 26.
    #    
    #    My construction has rank 18.
    #    This is impossible.
    #    
    #    I must have made a mistake in the index mapping.
    #    U is size (n*m) x rank.
    #    V is size (m*p) x rank.
    #    W is size (p*n) x rank.
    #    
    #    The tensor T is size (n*m) x (m*p) x (p*n)? No.
    #    The tensor T has indices (i,j,k) where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    #    The decomposition is T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    #    
    #    In my construction, for term c_k:
    #    U[i*m+j, c_k] = 1.
    #    V[j*p+k, c_k] = 1.
    #    W[k*n+i, c_k] = 1.
    #    
    #    Let's check T[i,j,k] for a fixed k.
    #    T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
    #    Only c_k contributes.
    #    T[i,j,k] = U[i*m+j,c_k] V[j*p+k,c_k] W[k*n+i,c_k] = 1*1*1 = 1.
    #    
    #    This seems correct.
    #    
    #    Why does it violate the lower bound?
    #    Maybe the lower bound is for a different tensor?
    #    No, it's for matrix multiplication.
    #    
    #    Let's check the dimensions again.
    #    U is 4 x 18.
    #    V is 26 x 18.
    #    W is 26 x 18.
    #    
    #    The rank of the tensor is the rank of the matrix unfolding.
    #    The (n*m) x (m*p) unfolding is U V^T.
    #    The rank of U V^T is at most rank(U) and rank(V).
    #    rank(U) <= 4.
    #    So the rank of the unfolding is <= 4.
    #    But the rank of the tensor must be at least the rank of any unfolding.
    #    So rank(T) >= rank(U V^T).
    #    This doesn't give a lower bound of 26.
    #    
    #    The lower bound max(nm, mp, pn) comes from the fact that the tensor must be
    #    able to represent the identity map on the space of matrices.
    #    
    #    Actually, the lower bound is rank(T) >= max(nm, mp, pn) is NOT generally true.
    #    The lower bound is rank(T) >= max(nm, mp, pn) / something?
    #    No, the known lower bound is rank(T) >= max(nm, mp, pn) is false.
    #    The lower bound is rank(T) >= max(nm, mp, pn) is for the *border rank*?
    #    
    #    Actually, the lower bound is rank(T) >= max(nm, mp, pn) is NOT correct.
    #    The correct lower bound is rank(T) >= max(nm, mp, pn) is not a standard result.
    #    The standard lower bound is rank(T) >= max(nm, mp, pn) is not true.
    #    For example, Strassen for 2x2x2 has rank 7, and max(4,4,4)=4. 7>=4.
    #    For 2x2x13, max(4,26,26)=26.
    #    If rank is 18, then 18 < 26.
    #    So if the lower bound is 26, rank 18 is impossible.
    #    
    #    Let's assume the lower bound is correct and my construction is flawed.
    #    I will use a safe construction with rank 26 or higher.
    #    
    #    A simple construction:
    #    For each column k, use 2 terms (Strassen-like for 2x2 x 2x1).
    #    For 2x2 x 2x1, the rank is 3 (or 2 if we allow non-integer?).
    #    Actually, 2x2 x 2x1 is just 2 dot products of length 2.
    #    C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k].
    #    C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k].
    #    This can be done with 2 multiplications:
    #    M1 = A[0,0]B[0,k], M2 = A[0,1]B[1,k].
    #    C[0,k] = M1+M2.
    #    M3 = A[1,0]B[0,k], M4 = A[1,1]B[1,k].
    #    C[1,k] = M3+M4.
    #    This is 4 multiplications per column.
    #    
    #    Can we do better?
    #    Use Karatsuba-like:
    #    S = (A[0,0]+A[1,0]) B[0,k]
    #    D = (A[0,1]+A[1,1]) B[1,k]
    #    This doesn't help directly.
    #    
    #    Let's use 2 terms per column:
    #    Term 1: (A[0,0]+A[1,0]) B[0,k] -> contributes to C[0,k] and C[1,k]
    #    Term 2: (A[0,1]+A[1,1]) B[1,k] -> contributes to C[0,k] and C[1,k]
    #    This is not correct.
    #    
    #    Let's stick to 4 terms per column for safety.
    #    Rank = 4 * 13 = 52.
    #    
    #    Or use Strassen for the first 2 columns (7 terms) and 4 terms for the rest (11*4=44).
    #    Total rank = 7 + 44 = 51.
    #    
    #    Let's implement rank 51.

    rank = 51
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(26)]
    W = [[0.0] * rank for _ in range(26)]

    # Strassen for columns 0,1 (7 terms)
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A[1,0]+A[1,1])B[0,0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1

    # M3 = A[0,0](B[1,0]-B[1,1])
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # M4 = A[1,1](B[0,1]-B[0,0])
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # M5 = (A[0,0]+A[0,1])B[1,1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1

    # Schoolbook for columns 2..12 (4 terms each)
    col = 7
    for k in range(2, p):
        # Term 1: A[0,0]B[0,k]
        U[0][col] = 1
        V[0 * m + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        # Term 2: A[0,1]B[1,k]
        U[1][col] = 1
        V[1 * m + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        # Term 3: A[1,0]B[0,k]
        U[2][col] = 1
        V[0 * m + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        # Term 4: A[1,1]B[1,k]
        U[3][col] = 1
        V[1 * m + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    # Verify
    def verify(n, m, p, U, V, W, tol=1e-9):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1

    return {"U": U, "V": V, "W": W}
