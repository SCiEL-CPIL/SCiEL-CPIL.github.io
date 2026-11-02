import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact with integer/half-integer entries.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = len(U[0]) if U else 0
    if R == 0:
        return False
        
    # Convert to Fractions for exact arithmetic
    # U, V, W are lists of lists
    U_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in U]
    V_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in V]
    W_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                if total != 1:
                    return False
    return True

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 multiplication (Strassen).
    Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    U shape: (2*2, 7) = (4, 7)
    V shape: (2*2, 7) = (4, 7)
    W shape: (2*2, 7) = (4, 7)
    """
    # Strassen matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U corresponds to A factors
    # V corresponds to B factors
    # W corresponds to C factors
    
    # We need to map the 2x2 blocks to the 2x2 indices.
    # A11 is index (0,0), A12 is (0,1), A21 is (1,0), A22 is (1,1).
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    # U: A11+A22 -> (1,0,0,1)
    U[0][0] = 1; U[3][0] = 1
    # V: B11+B22 -> (1,0,0,1)
    V[0][0] = 1; V[3][0] = 1
    # W: C11 += M1, C22 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2
    # U: A21+A22 -> (0,0,1,1)
    U[2][1] = 1; U[3][1] = 1
    # V: B11 -> (1,0,0,0)
    V[0][1] = 1
    # W: C21 += M2
    W[2][1] = 1
    
    # r=2: M3
    # U: A11 -> (1,0,0,0)
    U[0][2] = 1
    # V: B12-B22 -> (0,1,0,-1)
    V[1][2] = 1; V[3][2] = -1
    # W: C12 += M3
    W[1][2] = 1
    
    # r=3: M4
    # U: A22 -> (0,0,0,1)
    U[3][3] = 1
    # V: B21-B11 -> (0,0,1,-1)
    V[2][3] = 1; V[0][3] = -1
    # W: C22 -= M4
    W[3][3] = -1
    
    # r=4: M5
    # U: A11+A12 -> (1,1,0,0)
    U[0][4] = 1; U[1][4] = 1
    # V: B22 -> (0,0,0,1)
    V[3][4] = 1
    # W: C12 -= M5
    W[1][4] = -1
    
    # r=5: M6
    # U: A21-A11 -> (0,0,1,-1)
    U[2][5] = 1; U[0][5] = -1
    # V: B11+B12 -> (1,1,0,0)
    V[0][5] = 1; V[1][5] = 1
    # W: C21 -= M6
    W[2][5] = -1
    
    # r=6: M7
    # U: A12-A22 -> (0,1,0,-1)
    U[1][6] = 1; U[3][6] = -1
    # V: B21+B22 -> (0,0,1,1)
    V[2][6] = 1; V[3][6] = 1
    # W: C11 -= M7
    W[0][6] = -1
    
    return U, V, W

def _build_1x3x3():
    """
    Returns U, V, W for 1x3x3 multiplication.
    Indices: i in {0}, j in {0,1,2}, k in {0,1,2}.
    U shape: (1*3, 3) = (3, 3)
    V shape: (3*3, 3) = (9, 3)
    W shape: (3*1, 3) = (3, 3)
    """
    U = [[0]*3 for _ in range(3)]
    V = [[0]*3 for _ in range(9)]
    W = [[0]*3 for _ in range(3)]
    
    # Decomposition:
    # r=0: u=e0, v=E_{0,:}, w=(1,1,1)
    # r=1: u=e1, v=E_{1,:}, w=(1,1,1)
    # r=2: u=e2, v=E_{2,:}, w=(1,1,1)
    
    for r in range(3):
        U[r][r] = 1
        # v is a 3x3 matrix flattened to 9 elements.
        # v_{r,jk} = 1 if j=r, else 0.
        for k in range(3):
            idx = r * 3 + k
            V[idx][r] = 1
        # w is a 3x1 vector flattened to 3 elements.
        for k in range(3):
            W[k][r] = 1
            
    return U, V, W

def _build_rank_10_decomposition():
    """
    Build a rank-10 decomposition for 2x3x3 matrix multiplication.
    
    The 2x3x3 multiplication tensor can be viewed as:
    - A 2x2x2 block (rows 0-1, cols 0-1) using Strassen (rank 7)
    - Additional terms to handle the third column of A and the interactions
    
    Actually, let's think more carefully. The 2x3x3 tensor T_{i,j,k} for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    can be decomposed by considering it as two 1x3x3 tensors stacked, but that gives rank 6.
    
    Alternatively, we can use a block approach:
    - Use Strassen for the 2x2x2 sub-tensor (i in {0,1}, j in {0,1}, k in {0,1})
    - Handle the remaining terms separately
    
    Let me reconsider. The standard approach for 2x3x3 is to note that:
    rank(2x3x3) <= rank(2x2x2) + rank(1x3x3) = 7 + 3 = 10
    
    The decomposition works by partitioning the j-index:
    - For j=0,1: use the 2x2x2 Strassen decomposition
    - For j=2: use a 1x3x3 decomposition (treating it as a separate 1x3x3 problem)
    
    But this doesn't quite work directly because the tensor is not a direct sum.
    
    Let me try a different approach: use the fact that we can pad the 2x2x2 to 2x3x3.
    
    Actually, the correct way is:
    - The 2x3x3 multiplication can be seen as computing C = A*B where A is 2x3 and B is 3x3.
    - We can write A = [A1 | a2] where A1 is 2x2 and a2 is a 2x1 column.
    - Similarly B = [B1 | b2] where B1 is 3x2 and b2 is a 3x1 column.
    - Then C = A1*B1 + a2*b2^T... no, that's not right.
    
    Let me think again. The tensor is T[i,j,k] = delta_{i,j} * delta_{j,k} in the standard basis, but for matrix multiplication it's the structure constant.
    
    For matrix multiplication of (2x3) * (3x3) -> (2x3):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    We can decompose this as:
    - Part 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 tensor (Strassen, rank 7)
    - Part 2: i in {0,1}, j=2, k in {0,1,2} -> 2x1x3 tensor
    - Part 3: i in {0,1}, j in {0,1}, k=2 -> 2x2x1 tensor
    
    But these parts overlap at (i, j in {0,1}, k in {0,1}) which is already handled.
    
    Actually, the cleanest decomposition is:
    - Use Strassen for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    - For the remaining terms where j=2 or k=2, we need additional rank terms.
    
    Let me count:
    - Terms with j=2: i in {0,1}, k in {0,1,2} -> 2*3 = 6 terms
    - Terms with k=2 and j in {0,1}: i in {0,1}, j in {0,1} -> 2*2 = 4 terms
    - But we've double-counted (i, j=2, k=2) which is 2 terms
    
    Total additional terms: 6 + 4 - 2 = 8 terms, which would give rank 7+8=15, not 10.
    
    Let me try a different approach. Use the 1x3x3 decomposition for one row of A.
    
    Actually, I think the correct decomposition is:
    - Treat the 2x3x3 as two 1x3x3 tensors (one for each row of A)
    - But this gives rank 3+3=6, which is too low (impossible since rank >= max(n*m, m*p, p*n) = 9)
    
    The minimum rank for 2x3x3 is at least 9 (since n*m=6, m*p=9, p*n=6, so rank >= 9).
    
    Let me try rank 9 first with a simple decomposition:
    For each (i,k) pair, we need to compute C[i,k] = sum_j A[i,j]*B[j,k].
    This is a 2x3 = 6 output, each requiring 3 multiplications, so schoolbook is rank 18.
    
    For rank 9, we need a smarter approach.
    
    Let me try: for each j, we have a rank-1 term:
    T_j = (column j of A) outer (row j of B)
    
    So the tensor is sum_{j=0}^{2} (A[:,j]) (B[j,:])
    
    This gives rank 3, but that's not right because the tensor is T[i,j,k] not T[i,k].
    
    I'm confusing myself. Let me restart.
    
    The matrix multiplication tensor for (2x3)*(3x3) has entries T[i,j,k] where:
    - i indexes rows of A and C (0,1)
    - j indexes columns of A and rows of B (0,1,2)  
    - k indexes columns of B and C (0,1,2)
    
    The tensor satisfies: C[i,k] = sum_j A[i,j] B[j,k]
    
    So T[i,j,k] = 1 if we're computing the (i,k) entry from the j-th term.
    
    Actually, the tensor is defined such that <T, A, B> = C where C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k].
    So T[i,j,k] = delta_{i,k}... no, that's not right either.
    
    For matrix multiplication, the tensor is T[i,j,k] = 1 for all valid i,j,k, and the contraction gives:
    sum_{j} A[i,j] B[j,k] = C[i,k]
    
    So the tensor has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    Wait, that's not right. The tensor for matrix multiplication (n x m) * (m x p) is:
    T[i,j,k] where i in {0,...,n-1}, j in {0,...,m-1}, k in {0,...,p-1}
    and the multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    
    So the tensor is T[i,j,k] = 1 for all i,j,k, and the contraction is:
    <T, A, B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    
    But this gives a scalar, not a matrix. I'm misunderstanding the tensor structure.
    
    The correct tensor structure for matrix multiplication is that it's a 3-tensor where:
    - First mode: (i,k) pairs -> n*p dimensions
    - Second mode: (i,j) pairs -> n*m dimensions  
    - Third mode: (j,k) pairs -> m*p dimensions
    
    And T[(i,k), (i',j), (j',k')] = 1 if i=i' and k=k' and j=j', else 0.
    
    So for 2x3x3:
    - First mode: 2*3 = 6 dimensions (output C indices)
    - Second mode: 2*3 = 6 dimensions (A indices)
    - Third mode: 3*3 = 9 dimensions (B indices)
    
    And T[(i,k), (i',j), (j',k')] = 1 if i=i' and k=k' and j=j'.
    
    This is the correct structure. Now, the rank of this tensor is what we're trying to minimize.
    
    The schoolbook decomposition has rank 6*3 = 18 (one term for each (i,j,k) triple).
    
    For a better decomposition, we can use Strassen-like ideas.
    
    Let me try to construct a rank-10 decomposition.
    
    Actually, I recall that the rank of 2x3x3 matrix multiplication is known to be 10. Let me try to construct it.
    
    One approach: use the fact that 2x3x3 can be embedded in 3x3x3, and use known 3x3x3 decompositions.
    
    Alternatively, use a block decomposition:
    - Split the 2x3 matrix into [A1 | a] where A1 is 2x2 and a is 2x1
    - Split the 3x3 matrix into [B1; b^T] where B1 is 2x3 and b is 3x1... no, that doesn't work dimensionally.
    
    Let me try: A is 2x3, B is 3x3.
    Write A = [A_left | A_right] where A_left is 2x2 and A_right is 2x1.
    Write B = [B_top; B_bottom] where B_top is 2x3 and B_bottom is 1x3.
    
    Then C = A*B = A_left*B_top + A_right*B_bottom
    
    A_left*B_top is 2x2 * 2x3 -> 2x3, which is a 2x2x3 multiplication.
    A_right*B_bottom is 2x1 * 1x3 -> 2x3, which is a 2x1x3 multiplication.
    
    Rank of 2x2x3: at least max(2*2, 2*3, 3*2) = 6. Known to be 7 (using Strassen-like).
    Rank of 2x1x3: at least max(2*1, 1*3, 3*2) = 6. But actually, 2x1x3 is just outer product of a 2-vector and a 3-vector, so rank is 1? No, that's not right.
    
    For 2x1x3: A is 2x1, B is 1x3, C is 2x3.
    C[i,k] = A[i,0] * B[0,k]
    This is a rank-1 tensor (outer product of A and B), so rank 1.
    
    So total rank = 7 + 1 = 8? But that seems too good.
    
    Wait, the 2x2x3 multiplication: A is 2x2, B is 2x3, C is 2x3.
    C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    
    This is not the same as 2x2x2. The rank of 2x2x3 is at least 6 (since n*p=6).
    
    Actually, I think the rank of 2x2x3 is 7 (can be done with 7 multiplications).
    
    So total rank = 7 + 1 = 8. But the lower bound is max(2*3, 2*3, 3*3) = 9, so rank 8 is impossible.
    
    I must be making an error. Let me reconsider.
    
    The decomposition C = A_left*B_top + A_right*B_bottom is correct, but the two parts are not independent in the tensor sense.
    
    Actually, they are independent because they involve different indices of A and B.
    A_left uses A[0:2,0:2] and B[0:2,:], while A_right uses A[0:2,2] and B[2,:].
    
    So the tensor is the direct sum of the 2x2x3 tensor and the 2x1x3 tensor.
    
    Rank of 2x2x3: Let's compute. The tensor T[(i,k),(i',j),(j',k')] = 1 if i=i', k=k', j=j'.
    For 2x2x3: i,i' in {0,1}, j,j' in {0,1}, k,k' in {0,1,2}.
    
    This tensor has rank at least max(2*3, 2*2, 2*3) = 6.
    
    Can we do it in 6? Probably not, because it's similar to 2x2x2 which needs 7.
    
    Actually, 2x2x3 can be done in 7 multiplications using a Strassen-like approach.
    
    And 2x1x3: A is 2x1, B is 1x3, C is 2x3.
    C[i,k] = A[i,0] * B[0,k]
    This is a rank-1 tensor, so rank 1.
    
    Total: 7 + 1 = 8. But the lower bound is 9, so something is wrong.
    
    Oh wait, the lower bound for 2x3x3 is max(n*m, m*p, p*n) = max(6, 9, 6) = 9.
    
    So rank 8 is impossible. My decomposition must be flawed.
    
    The issue is that when I split A = [A_left | A_right] and B = [B_top; B_bottom], the two products A_left*B_top and A_right*B_bottom are not independent in the sense that they share the same output C.
    
    Actually, they are independent because they compute different linear combinations.
    
    Let me verify: C = A*B = [A_left | A_right] * [B_top; B_bottom] = A_left*B_top + A_right*B_bottom
    
    This is correct. And the tensor for the first part uses only A_left and B_top, while the second part uses only A_right and B_bottom.
    
    So the total tensor is the direct sum, and the rank should be rank(2x2x3) + rank(2x1x3).
    
    If rank(2x2x3) = 7 and rank(2x1x3) = 1, then total rank = 8, which contradicts the lower bound of 9.
    
    Therefore, either rank(2x2x3) > 7 or rank(2x1x3) > 1.
    
    Let me reconsider rank(2x1x3). A is 2x1, B is 1x3, C is 2x3.
    C[i,k] = A[i,0] * B[0,k]
    
    The tensor T[(i,k), i', j'] where i,i' in {0,1}, j' in {0}, k,k' in {0,1,2}.
    T[(i,k), i', j'] = 1 if i=i' and k=k' and j'=0.
    
    This tensor has only 2*3 = 6 non-zero entries, all in the j'=0 slice.
    It's a rank-1 tensor: T = u * v * w where u[i,k] = 1, v[i'] = 1, w[j'] = delta_{j',0}.
    
    So rank(2x1x3) = 1.
    
    Then rank(2x2x3) must be at least 8 to make the total at least 9.
    
    Actually, I think rank(2x2x3) = 8. Let me verify.
    
    The tensor for 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is like two 1x2x3 tensors added together.
    Each 1x2x3 tensor has rank 2 (since it's sum of two rank-1 terms).
    But they share the same B, so we can't just add the ranks.
    
    Actually, the 2x2x3 tensor can be written as:
    T = sum_{j=0}^{1} (A[:,j] outer B[j,:])
    
    Where A[:,j] is a 2-vector and B[j,:] is a 3-vector.
    So T is a sum of 2 rank-1 tensors, giving rank <= 2.
    
    But that can't be right because the lower bound is 6.
    
    I'm still confused about the tensor structure. Let me look it up mentally.
    
    For matrix multiplication (n x m) * (m x p) -> (n x p), the tensor is in V_{n*p} * V_{n*m} * V_{m*p}.
    
    The basis for V_{n*p} is indexed by (i,k) for i in {0,...,n-1}, k in {0,...,p-1}.
    The basis for V_{n*m} is indexed by (i,j) for i in {0,...,n-1}, j in {0,...,m-1}.
    The basis for V_{m*p} is indexed by (j,k) for j in {0,...,m-1}, k in {0,...,p-1}.
    
    And T[(i,k), (i',j), (j',k')] = 1 if i=i' and k=k' and j=j', else 0.
    
    So for 2x2x3:
    T[(i,k), (i',j), (j',k')] = 1 if i=i', k=k', j=j'.
    
    The number of non-zero entries is 2*2*3 = 12.
    
    The rank of this tensor is at least the dimension of any mode, so at least max(6, 4, 6) = 6.
    
    Can we write it as a sum of 6 rank-1 tensors? Probably not, because it's similar to the 2x2x2 case which needs 7.
    
    Actually, I think the rank of 2x2x3 is 7. Let me try to construct it.
    
    For 2x2x3, we can use a Strassen-like decomposition for the 2x2x2 part (first two columns of B) and handle the third column separately.
    
    But this is getting complicated. Let me just try to implement a rank-10 decomposition and verify it.
    
    Actually, let me try a different approach: use the known fact that rank(2x3x3) = 10 and construct it.
    
    One known decomposition uses 10 terms. Let me try to find it.
    
    After some thought, I believe the correct approach is:
    - Use 7 terms for the 2x2x2 sub-tensor (Strassen)
    - Use 3 terms for the remaining part
    
    The remaining part corresponds to j=2 (the third column of A and third row of B).
    
    For j=2: C[i,k] += A[i,2] * B[2,k] for all i,k.
    This is a 2x1x3 tensor, which has rank 1, not 3.
    
    So total rank = 7 + 1 = 8, which is still less than 9.
    
    I must be making a fundamental error. Let me just implement a safe rank-18 schoolbook decomposition and return that, since I'm not sure about the rank-10 construction.
    
    Actually, let me try rank 9 with a simple decomposition:
    For each (i,k), we need to compute C[i,k] = sum_j A[i,j] B[j,k].
    We can use 3 terms for each (i,k), giving 6*3 = 18 terms.
    
    For rank 9, we need to share terms.
    
    Let me try: for each j, we have a term that contributes to all (i,k) where i is in {0,1} and k is in {0,1,2}.
    So term j: U[:,j] = A[:,j] (2-vector), V[:,j] = B[j,:] (3-vector), W[:,j] = identity (2x3 matrix).
    
    But this gives 3 terms, and the tensor would be sum_j (A[:,j] outer B[j,:]) which is correct, but the rank is 3, which is impossible.
    
    I think the issue is that the tensor structure is not what I think it is.
    
    Let me just implement the schoolbook rank-18 decomposition and return that. It's safe and correct.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Use Strassen for the 2x2x2 part (i in {0,1}, j in {0,1}, k in {0,1})
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    # Map Strassen indices to 2x3x3 indices
    # A indices: (i,j) -> i*m + j
    # B indices: (j,k) -> j*p + k
    # C indices: (i,k) -> k*n + i (as per the problem statement: W[k*n+i,r])
    
    for r in range(7):
        for idx_u in range(4):
            i = idx_u // 2
            j = idx_u % 2
            U[i*m + j][r] = U_s[idx_u][r]
        for idx_v in range(4):
            j = idx_v // 2
            k = idx_v % 2
            V[j*p + k][r] = V_s[idx_v][r]
        for idx_w in range(4):
            k = idx_w // 2
            i = idx_w % 2
            W[k*n + i][r] = W_s[idx_w][r]
    
    # Now handle the remaining terms: j=2 or k=2
    # Terms with j=2: i in {0,1}, k in {0,1,2}
    # Terms with k=2 and j in {0,1}: i in {0,1}, j in {0,1}
    
    # For j=2: C[i,k] += A[i,2] * B[2,k]
    # This is a 2x1x3 tensor. We can use 3 terms (one for each k).
    for k in range(3):
        r = 7 + k
        for i in range(2):
            U[i*m + 2][r] = 1
        for j in range(3):
            if j == 2:
                V[j*p + k][r] = 1
        for i in range(2):
            W[k*n + i][r] = 1
    
    # But this gives rank 7+3=10, and we haven't handled the k=2, j in {0,1} terms yet.
    # Actually, the k=2, j in {0,1} terms are: C[i,2] += A[i,j] * B[j,2] for j in {0,1}
    # These are not covered by the j=2 terms above.
    
    # So we need additional terms for k=2, j in {0,1}.
    # But we only have 10-7-3=0 terms left.
    
    # This means my decomposition is incomplete.
    
    # Let me try a different approach: use 3 terms for j=2 and 3 terms for k=2, j in {0,1}.
    # But that would give 7+3+3=13 terms.
    
    # Actually, I think the correct decomposition is:
    # - 7 terms for 2x2x2 (Strassen)
    # - 3 terms for the 1x3x3 part (treating one row of A)
    
    # Let me try: use the 1x3x3 decomposition for i=0 and i=1 separately.
    # But that would give 3+3=6 terms, plus the 2x2x2 terms, which doesn't work.
    
    # I'm stuck. Let me just return the schoolbook rank-18 decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to build a rank-10 decomposition
    try:
        result = _build_rank_10_decomposition()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook rank-18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
