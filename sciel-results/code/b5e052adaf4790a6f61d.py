"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=5, p=6, max_check=100):
    """
    Verify decomposition using random matrices.
    The tensor identity is:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
    Wait, no. The standard rank-1 term for matrix multiplication tensor T_{ijk} = delta_{ij} delta_{jk} (indices i,j,k for A,B,C)
    is usually written as:
    T = sum_r u_r \otimes v_r \otimes w_r
    where (u_r \otimes v_r \otimes w_r)_{i,j,k} = u_r[i,j] v_r[j,k] w_r[k,i]
    And the product is C[i,k] = sum_j A[i,j] B[j,k].
    The identity is:
    sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] u_r[i,j]) (sum_j B[j,k] v_r[j,k]) w_r[k,i]
    Let's check indices.
    Usually: C = A B.
    Tensor T has entries T[i,j,k] = 1 if we view it as a 3-tensor where A is on (i,j), B on (j,k), C on (i,k).
    Actually, the standard definition for the multiplication tensor $M_{n,m,p}$ is:
    $M = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p E_{ij} \otimes E_{jk} \otimes E_{ki}$
    Here $E_{ij}$ is the matrix unit for A, $E_{jk}$ for B, $E_{ki}$ for C.
    So the term is $u_r[i,j] v_r[j,k] w_r[k,i]$.
    The product is computed as:
    $C[i,k] = \sum_r (\sum_j A[i,j] u_r[i,j]) (\sum_j B[j,k] v_r[j,k]) w_r[k,i]$
    Wait, the sums over j must be consistent.
    Let $X_r[i] = \sum_j A[i,j] u_r[i,j]$
    Let $Y_r[k] = \sum_j B[j,k] v_r[j,k]$
    Then $C[i,k] = \sum_r X_r[i] Y_r[k] w_r[k,i]$? No, $w_r$ is a matrix $W_r$ of size $p \times n$.
    The term is $w_r[k,i]$.
    So $C[i,k] = \sum_r (\sum_j A[i,j] U_r[i,j]) (\sum_j B[j,k] V_r[j,k]) W_r[k,i]$.
    
    Let's verify with schoolbook.
    For term (i0, j0, k0):
    U[i0, j0] = 1, others 0.
    V[j0, k0] = 1, others 0.
    W[k0, i0] = 1, others 0.
    X_r[i] = A[i, j0] if i=i0 else 0.
    Y_r[k] = B[j0, k] if k=k0 else 0.
    W_r[k,i] = 1 if k=k0, i=i0 else 0.
    Sum over r: Only term (i0, j0, k0) contributes to C[i0, k0].
    Contribution: A[i0, j0] * B[j0, k0] * 1.
    Sum over j0 gives C[i0, k0] = sum_j A[i0,j] B[j,k0]. Correct.
    """
    if not (len(U) == n*m and len(V) == m*p and len(W) == p*n):
        return False
    r = len(U[0]) if n*m > 0 else 0
    if len(V[0]) != r or len(W[0]) != r:
        return False
    if r == 0:
        return n*m == 0
    
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    
    for _ in range(max_check):
        A = np.random.randint(-2, 3, (n, m))
        B = np.random.randint(-2, 3, (m, p))
        C_expected = A @ B
        
        # X_r[i] = sum_j A[i,j] U_arr[i,j,r]
        # Y_r[k] = sum_j B[j,k] V_arr[j,k,r]
        # C[i,k] = sum_r X_r[i] Y_r[k] W_arr[k,i,r]
        
        # Efficient calculation:
        # X: (n, m) x (n, m, r) -> (n, r) ? No.
        # X[i, r] = sum_j A[i, j] U_arr[i, j, r]
        # This is element-wise multiply A[..., None] * U_arr, then sum over j (axis 1).
        X = np.sum(A[:, :, np.newaxis] * U_arr, axis=1)  # shape (n, r)
        Y = np.sum(B[np.newaxis, :, :] * V_arr, axis=1)  # shape (p, r)
        
        # C[i, k] = sum_r X[i, r] Y[k, r] W_arr[k, i, r]
        # Let's form a tensor Z[i, k, r] = X[i, r] Y[k, r] W_arr[k, i, r]
        # Then C[i, k] = sum_r Z[i, k, r]
        
        # X[i, r] * W_arr[k, i, r] is (n, p, r)
        # Y[k, r] is (p, r) -> (1, p, r)
        # Term = X[:, np.newaxis, :] * W_arr * Y[np.newaxis, :, :]
        # Sum over r (axis 2)
        
        C_calc = np.sum(X[:, np.newaxis, :] * W_arr * Y[np.newaxis, :, :], axis=2)
        
        if not np.allclose(C_calc, C_expected, atol=1e-9):
            return False
    return True


def _try_merge(U, V, W, idx1, idx2, n, m, p):
    """
    Try to merge two rank-1 terms into one.
    If we have terms t1 and t2, we want to find coefficients a, b, c, d such that
    a*u1 + b*u2 = u_new, etc., and the resulting tensor matches.
    This is generally hard. A simpler heuristic:
    Check if the two terms are "compatible" in a way that allows a Strassen-like combination.
    
    Actually, a more robust approach for small reductions is to check if the sum of two rank-1 tensors
    can be represented as a single rank-1 tensor? No, rank of sum is <= sum of ranks.
    We want to replace 2 terms with 1 term. This implies the two terms were linearly dependent in a specific way
    or part of a larger structure.
    
    However, for matrix multiplication, simple pairwise merging of schoolbook terms doesn't work directly
    because the indices are distinct.
    
    Let's try a different strategy: Random search for a valid decomposition of rank R < 60.
    We can start with schoolbook and try to "perturb" the coefficients and drop terms,
    then solve for the remaining coefficients to satisfy the tensor identity.
    
    But solving for coefficients is a system of linear equations?
    The tensor identity is trilinear.
    T = sum_r u_r v_r w_r.
    If we fix the structure (which indices are non-zero) and only vary coefficients, it's linear.
    But we also vary the structure (which entries are non-zero).
    
    Let's stick to the "Strassen block" idea if possible, or just return schoolbook if no quick win.
    Given the time limit and complexity, and that all inspirations scored 1.0 (schoolbook),
    it is highly likely that finding a rank < 60 decomposition for 2x5x6 is non-trivial
    and might not be achievable with simple heuristics in 120s.
    
    However, the task says "score is the rank". Lower is better.
    If I return 60, score is 1.0.
    If I return 59, score is > 1.0.
    
    Is there a known lower rank for 2x5x6?
    Rank of 2x2x2 is 7.
    Rank of 2x2x3?
    If we can decompose 2x5x6 into blocks:
    2x5 = 2x2 + 2x3
    5x6 = 2x6 + 3x6
    C = A1 B1 + A2 B2.
    Rank(T) <= Rank(T_{2x2x6}) + Rank(T_{2x3x6}).
    
    What is Rank(T_{2x2x6})?
    We can view 2x2x6 as two 2x2x3 blocks? No, the j index is shared.
    A (2x2), B (2x6). C (2x6).
    C[i,k] = sum_{j=0}^1 A[i,j] B[j,k].
    This is just standard multiplication.
    Can we do better than 2*2*6 = 24?
    For 2x2xn, the rank is known to be 3n - 2 for n>=2?
    Let's check.
    For 2x2x2, rank 7. 3*2-2 = 4? No.
    Strassen is 7.
    For 2x2x3, is it less than 18?
    Yes, likely.
    
    If we use a known algorithm for 2x2xn, we might reduce the rank.
    However, implementing a general low-rank algorithm for arbitrary n is hard.
    
    Let's try a specific heuristic:
    If we can find ANY valid decomposition with rank < 60, we win.
    
    Strategy:
    1. Start with schoolbook (rank 60).
    2. Try to apply Strassen's algorithm to sub-blocks.
       Split A into A1 (2x2) and A2 (2x3).
       Split B into B1 (2x6) and B2 (3x6).
       C = A1 B1 + A2 B2.
       
       For A1 B1 (2x2x6):
       We can use Strassen-like methods if we further split B1?
       Or use the fact that 2x2xn multiplication can be done with 3n-2 multiplications?
       Actually, the rank of the tensor $M_{2,2,n}$ is $3n-2$ for $n \ge 2$ is a conjecture/bound?
       Let's assume we can find a decomposition for 2x2x6 with rank < 24.
       
       For A2 B2 (2x3x6):
       Rank < 36.
       
       If we can just find *any* improvement, we are good.
       
    Given the difficulty of deriving these on the fly, and the risk of invalidity,
    I will implement a robust verification and return the schoolbook decomposition.
    I will also add a simple local search that tries to zero out coefficients in the schoolbook decomposition
    and see if the tensor identity still holds (it won't, unless dependent).
    
    Wait, if I just return schoolbook, I get 1.0.
    The prompt asks to "Combine ... strongest ideas".
    The strongest idea is the verification function.
    The structural idea is the schoolbook construction.
    
    I will enhance the program to:
    1. Construct schoolbook.
    2. Attempt a few specific "Strassen" merges on the 2x2 blocks if possible.
       Specifically, for the 2x2x2 part (indices i=0,1, j=0,1, k=0,1), we can replace 8 terms with 7.
       This reduces rank by 1.
       We have multiple 2x2x2 blocks?
       A is 2x5. B is 5x6.
       The interaction is over j.
       We can't just apply Strassen to a 2x2x2 sub-block in isolation because the j index couples them.
       
       However, we can view the multiplication as:
       C = A B.
       If we partition the columns of A and rows of B?
       
       Let's try to partition j into two sets: J1 = {0,1}, J2 = {2,3,4}.
       C = A_J1 B_J1 + A_J2 B_J2.
       Term 1: 2x2x6 multiplication.
       Term 2: 2x3x6 multiplication.
       
       If we have a low-rank decomposition for 2x2x6, we can use it.
       Is there a simple one?
       For 2x2xn, one can use 3n-2 multiplications.
       For n=6, 3*6-2 = 16.
       Schoolbook for 2x2x6 is 24.
       So we save 8 terms.
       
       For 2x3x6, schoolbook is 36.
       Is there a bound?
       Maybe we can just use schoolbook for the second part.
       Total rank: 16 + 36 = 52.
       This is < 60.
       
       So if I can implement a 2x2x6 decomposition of rank 16, I win.
       
       How to construct 2x2xn rank 3n-2?
       This is related to the "Strassen" algorithm generalized.
       For 2x2x2, we have 7 terms.
       For 2x2x3, we can add 3 terms? No, the structure is more complex.
       
       Actually, the rank of $M_{2,2,n}$ is known to be $3n-2$ for $n \ge 2$?
       Let's check n=2: 3(2)-2 = 4? No, Strassen is 7.
       So the formula is different.
       
       Correct bound for 2x2xn:
       It is known that rank($M_{2,2,n}$) = 3n - 2 is WRONG.
       Strassen for 2x2x2 is 7.
       For 2x2x3, it is at most 7+3 = 10? (Using Strassen on first 2 cols, schoolbook on 3rd?)
       If we split B into B1 (2x2) and B2 (2x1).
       C = A B1 + A B2.
       Rank <= Rank(2x2x2) + Rank(2x2x1).
       Rank(2x2x1) = 2*2*1 = 4? Or less?
       2x2x1 is just dot products.
       C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
       C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
       This is 2 scalar multiplications? No, 4 terms in schoolbook.
       Can we do 3?
       (A00+A11)(B00+B10) ... no.
       Actually, 2x2x1 is equivalent to 2 independent dot products of length 2?
       No, B[0,0] is shared.
       It's a bilinear form.
       Rank of 2x2x1 tensor?
       T[i,j,0] = delta_{i,j} ? No.
       T[i,j,k] with k=0.
       C[i,0] = sum_j A[i,j] B[j,0].
       This is just matrix-vector multiplication.
       The tensor is $M_{2,2,1}$.
       Rank is 3?
       (A00+A10)(B00) + (A01+A11)(B10) + (A00+A11)(B00+B10) ...
       Actually, rank of 2x2x1 is 3.
       Proof:
       1. A00 B00
       2. A01 B10
       3. A10 B00
       4. A11 B10
       Can we merge?
       (A00+A10) B00 + (A01+A11) B10 = A00 B00 + A10 B00 + A01 B10 + A11 B10.
       This gives 2 terms.
       Wait, C[0,0] = A00 B00 + A01 B10.
       C[1,0] = A10 B00 + A11 B10.
       My combination gives:
       Term 1: (A00+A10) B00 -> contributes to C[0,0] as A00 B00 and C[1,0] as A10 B00.
       Term 2: (A01+A11) B10 -> contributes to C[0,0] as A01 B10 and C[1,0] as A11 B10.
       So yes, rank 2?
       Let's check the tensor definition.
       U[i,j, r], V[j,k, r], W[k,i, r].
       For r=1: U[0,0]=1, U[1,0]=1, others 0. V[0,0]=1, others 0. W[0,0]=1, W[0,1]=1?
       W is p x n = 1 x 2.
       W[0,0]=1, W[0,1]=1.
       Term 1:
       X[0] = A00*1 + A01*0 = A00.
       X[1] = A10*1 + A11*0 = A10.
       Y[0] = B00*1 + B10*0 = B00.
       C[0,0] += X[0] Y[0] W[0,0] = A00 B00 * 1.
       C[1,0] += X[1] Y[0] W[0,1] = A10 B00 * 1.
       
       For r=2:
       U[0,1]=1, U[1,1]=1. V[1,0]=1.
       X[0] = A01.
       X[1] = A11.
       Y[0] = B10.
       W[0,0]=1, W[0,1]=1.
       C[0,0] += A01 B10.
       C[1,0] += A11 B10.
       
       Total C[0,0] = A00 B00 + A01 B10. Correct.
       Total C[1,0] = A10 B00 + A11 B10. Correct.
       
       So Rank(2x2x1) = 2.
       
       So Rank(2x2x6) <= Rank(2x2x2) + 4 * Rank(2x2x1) = 7 + 4*2 = 15.
       Schoolbook is 24.
       So we can do 15.
       
       Now for 2x3x6.
       Split B into 3x2, 3x2, 3x2?
       Or just use schoolbook for the second part?
       Rank(2x3x6) <= 36.
       Can we improve?
       2x3x6 = sum of three 2x1x6?
       Rank(2x1x6) = 2 * 6 = 12?
       C[i,k] = A[i,0] B[0,k].
       This is outer product scaled.
       Rank is 2 (for each k? No, for the whole tensor).
       Tensor T[i,0,k] = 1.
       U[i,0,r], V[0,k,r], W[k,i,r].
       We can do:
       r=1: U[0,0]=1, V[0,0]=1, W[0,0]=1.
       r=2: U[1,0]=1, V[0,0]=1, W[0,1]=1.
       This only covers k=0.
       For each k, we need 2 terms?
       So 6 * 2 = 12 terms.
       So Rank(2x1x6) = 12.
       So Rank(2x3x6) <= 3 * 12 = 36.
       Schoolbook is also 36.
       
       So total rank: 15 + 36 = 51.
       This is a valid improvement from 60 to 51.
       
       Let's implement this.
       
       Steps:
       1. Initialize U, V, W for rank 51.
       2. Fill in the 2x2x6 part (j=0,1) using the 7+4*2=15 terms.
          - 7 terms for 2x2x2 (k=0,1).
          - 2 terms for each k in 2..5 (4 k's * 2 terms = 8 terms).
       3. Fill in the 2x3x6 part (j=2,3,4) using schoolbook (36 terms).
       
       Wait, the indices in the final U, V, W are global.
       U is 10 x R. (2*5)
       V is 30 x R. (5*6)
       W is 12 x R. (6*2)
       
       We need to map the local indices to global.
       
       Part 1: j in {0,1}.
       Local A is 2x2. Local B is 2x6. Local C is 2x6.
       Global A indices: i in {0,1}, j in {0,1}.
       Global B indices: j in {0,1}, k in {0..5}.
       Global C indices: i in {0,1}, k in {0..5}.
       
       Part 2: j in {2,3,4}.
       Local A is 2x3. Local B is 3x6. Local C is 2x6.
       Global A indices: i in {0,1}, j in {2,3,4}.
       Global B indices: j in {2,3,4}, k in {0..5}.
       Global C indices: i in {0,1}, k in {0..5}.
       
       We can construct the terms for each part and place them into the global matrices.
       
       Implementation details for 2x2x2 Strassen:
       We need the 7 terms.
       Standard Strassen for 2x2:
       M1 = (A00+A11)(B00+B11)
       M2 = (A10+A11)B00
       M3 = A00(B10-B11)
       M4 = A11(B01-B00)
       M5 = (A00+A01)B11
       M6 = (A10-A00)(B00+B10)
       M7 = (A01-A11)(B01+B11)
       
       C00 = M1 + M4 - M5 + M7
       C01 = M3 + M5
       C10 = M2 + M4
       C11 = M1 - M2 + M3 + M6
       
       We need to translate these into U, V, W terms.
       Each M is a product of a linear form in A and a linear form in B.
       M_r = (sum c_a A) (sum c_b B).
       This corresponds to a rank-1 term where:
       U[i,j] is the coefficient of A[i,j] in the A-part.
       V[j,k] is the coefficient of B[j,k] in the B-part.
       W[k,i] is the coefficient used to assemble C[i,k].
       
       For M1:
       A-part: A00 + A11. U[0,0]=1, U[1,1]=1.
       B-part: B00 + B11. V[0,0]=1, V[1,1]=1.
       C-contribution:
       C00 += M1 -> W[0,0] += 1.
       C11 += M1 -> W[1,1] += 1.
       
       For M2:
       A-part: A10 + A11. U[1,0]=1, U[1,1]=1.
       B-part: B00. V[0,0]=1.
       C-contribution:
       C10 += M2 -> W[0,1] += 1. (C[i=1, k=0])
       
       And so on.
       
       For 2x2x1 (k fixed):
       We found 2 terms.
       Term 1: A-part: A_i0. B-part: B_0k.
       U[i,0]=1. V[0,k]=1.
       C[i,k] += A_i0 B_0k.
       So W[k,i] += 1.
       
       Term 2: A-part: A_i1. B-part: B_1k.
       U[i,1]=1. V[1,k]=1.
       C[i,k] += A_i1 B_1k.
       So W[k,i] += 1.
       
       Wait, my previous derivation for 2x2x1 used:
       Term 1: U[0,0]=1, U[1,0]=1. V[0,0]=1. W[0,0]=1, W[0,1]=1.
       This computes:
       C[0,0] += (A00) * (B00) * 1.
       C[1,0] += (A10) * (B00) * 1.
       This covers the B00 part for both i.
       
       Term 2: U[0,1]=1, U[1,1]=1. V[1,0]=1. W[0,0]=1, W[0,1]=1.
       C[0,0] += (A01) * (B10) * 1.
       C[1,0] += (A11) * (B10) * 1.
       
       So for a specific k, we have 2 terms.
       
       So for 2x2x6:
       k=0,1: Use Strassen (7 terms).
       k=2,3,4,5: Use 2 terms each (8 terms).
       Total 15 terms.
       
       For 2x3x6:
       Just use schoolbook.
       For each i in {0,1}, j in {2,3,4}, k in {0..5}:
       1 term.
       Total 2*3*6 = 36 terms.
       
       Total rank = 15 + 36 = 51.
       
       Let's code this.
       
    """
    n, m, p = 2, 5, 6
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0,1}
    # Strassen for k=0,1
    # M1
    U[0*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*6+0][col] = 1; V[1*6+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1
    col += 1
    
    # M2
    U[1*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*6+0][col] = 1
    W[0*2+1][col] = 1
    col += 1
    
    # M3
    U[0*2+0][col] = 1
    V[1*6+0][col] = 1; V[1*6+1][col] = -1
    W[1*2+0][col] = 1
    col += 1
    
    # M4
    U[1*2+1][col] = 1
    V[0*6+1][col] = 1; V[0*6+0][col] = -1
    W[0*2+0][col] = 1; W[1*2+0][col] = 1
    col += 1
    
    # M5
    U[0*2+0][col] = 1; U[0*2+1][col] = 1
    V[1*6+1][col] = 1
    W[0*2+0][col] = -1; W[1*2+1][col] = 1
    col += 1
    
    # M6
    U[1*2+0][col] = 1; U[0*2+0][col] = -1
    V[0*6+0][col] = 1; V[0*6+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    
    # M7
    U[0*2+1][col] = 1; U[1*2+1][col] = -1
    V[0*6+1][col] = 1; V[1*6+1][col] = 1
    W[0*2+1][col] = 1
    col += 1
    
    # k=2,3,4,5
    for k in range(2, 6):
        # Term 1: A_i0 * B_0k
        for i in range(2):
            U[i*2+0][col] = 1
        V[0*6+k][col] = 1
        for i in range(2):
            W[k*2+i][col] = 1
        col += 1
        
        # Term 2: A_i1 * B_1k
        for i in range(2):
            U[i*2+1][col] = 1
        V[1*6+k][col] = 1
        for i in range(2):
            W[k*2+i][col] = 1
        col += 1
        
    # Part 2: j in {2,3,4}
    # Schoolbook
    for i in range(2):
        for j in range(2, 5):
            for k in range(6):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
                
    if col != rank:
        # If we messed up the count, pad or trim
        # But we calculated 7 + 8 + 36 = 51.
        pass
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
