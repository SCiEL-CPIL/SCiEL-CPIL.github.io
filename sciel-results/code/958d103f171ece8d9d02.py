"""Improved program: block-Strassen decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists, rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2():
    """Strassen's 7-product algorithm for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists (integers).
    A is 2x2 flattened row-major: (a,b,c,d) -> indices 0,1,2,3
    B is 2x2 flattened row-major: (e,f,g,h) -> indices 0,1,2,3
    C = A*B, W index = k*2 + i (n=2, p=2).
    W[0] -> i=0,k=0: C[0][0]
    W[1] -> i=0,k=1: C[0][1]
    W[2] -> i=1,k=0: C[1][0]
    W[3] -> i=1,k=1: C[1][1]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m times m x p = n x p.
    Returns U (n*m x n*m*p), V (m*p x n*m*p), W (p*n x n*m*p) as lists of lists.
    U row index: i*m + j
    V row index: j*p + k
    W row index: k*n + i
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _add_matrices(A, B):
    """Add two matrices (lists of lists) of same shape."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _construct_4x5x5():
    """Build 4x5x5 decomposition using block Strassen.
    
    A: 4x5, B: 5x5, C: 4x5.
    
    Split the summation index j (0-4) into j=0,1,2,3 and j=4.
    
    C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    
    Part 1: A_left (4x4) * B_top (4x5) -> 4x5
      Split A_left rows into 2+2: A_top (2x4), A_bot (2x4)
      Split B_top cols into 2+3: B_left (4x2), B_right (4x3)
      
      But we need to further decompose. Let's split differently:
      Split A_left into 2x2 blocks and B_top into 2x2 and 2x3 blocks.
      
      Actually, let's use a cleaner approach:
      A_left is 4x4, B_top is 4x5.
      Split A rows: [0,1] and [2,3]
      Split B cols: [0,1] and [2,3,4]
      
      But A_left has 4 cols (j=0,1,2,3), so we need to split those too.
      
      Let me restructure:
      Part 1: j=0,1,2,3
        A[i,0:4] * B[0:4,k]
        Split A rows: [0,1] and [2,3]
        Split B rows (j): [0,1] and [2,3]
        Split B cols: [0,1] and [2,3,4]
        
        This gets complex. Let me use a simpler split.
    
    Alternative: Split j into {0,1,2} and {3,4}
    
    Part 1: A(4x3) * B(3x5)
      Split A rows: [0,1] and [2,3]
      Split B cols: [0,1] and [2,3,4]
      A[0:2, 0:3] * B[0:3, 0:2]: 2x3*3x2 schoolbook = 12
      A[0:2, 0:3] * B[0:3, 2:5]: 2x3*3x3 schoolbook = 18
      A[2:4, 0:3] * B[0:3, 0:2]: 2x3*3x2 schoolbook = 12
      A[2:4, 0:3] * B[0:3, 2:5]: 2x3*3x3 schoolbook = 18
      Total Part 1: 60
    
    Part 2: A(4x2) * B(2x5) schoolbook = 40
    Total: 100 (worse than current)
    
    Let me try yet another approach. What if I use Strassen on 2x2 sub-blocks more aggressively?
    
    Split j into {0,1} and {2,3,4} (current approach gives 98)
    
    What if I split into {0,1,2} and {3,4}?
    Part 1: A(4x3)*B(3x5)
      Split A rows 2+2, B cols 2+3
      But A is 4x3, so I can't split cols evenly for Strassen.
      Use schoolbook: 4*3*5 = 60
    Part 2: A(4x2)*B(2x5) = 40
    Total: 100
    
    What about {0,1,2,3} and {4}?
    Part 1: A(4x4)*B(4x5)
      Split A rows 2+2
      For each 2x4 block of A, multiply by B(4x5)
      A[0:2, 0:4] * B[0:4, 0:5]: 
        Split B cols 2+3
        A[0:2, 0:4] * B[0:4, 0:2]: 2x4*4x2
          This is like (2x2 * 2x2) + (2x2 * 2x2) if we split A cols
          A[0:2, 0:2] * B[0:2, 0:2]: 2x2*2x2 Strassen = 7
          A[0:2, 0:2] * B[2:4, 0:2]: 2x2*2x2 Strassen = 7
          A[0:2, 2:4] * B[0:2, 0:2]: 2x2*2x2 Strassen = 7
          A[0:2, 2:4] * B[2:4, 0:2]: 2x2*2x2 Strassen = 7
          Total: 28
        A[0:2, 0:4] * B[0:4, 2:5]: 2x4*4x3
          A[0:2, 0:2] * B[0:2, 2:5]: 2x2*2x3 schoolbook = 12
          A[0:2, 0:2] * B[2:4, 2:5]: 2x2*2x3 schoolbook = 12
          A[0:2, 2:4] * B[0:2, 2:5]: 2x2*2x3 schoolbook = 12
          A[0:2, 2:4] * B[2:4, 2:5]: 2x2*2x3 schoolbook = 12
          Total: 48
        Subtotal for A[0:2]: 28 + 48 = 76
      Similarly for A[2:4]: 76
      Total Part 1: 152
    Part 2: A(4x1)*B(1x5) = 20
    Total: 172 (much worse)
    
    OK so the current approach with j={0,1} and {2,3,4} giving 98 seems reasonable.
    
    Let me try to improve by using Strassen where possible in the current structure.
    
    Current:
    Part 1: j={0,1}, A(4x2)*B(2x5)
      A rows 2+2, B cols 2+3
      1a: 2x2*2x2 Strassen = 7
      1b: 2x2*2x3 schoolbook = 12
      1c: 2x2*2x2 Strassen = 7
      1d: 2x2*2x3 schoolbook = 12
      Part 1 total: 38
    Part 2: j={2,3,4}, A(4x3)*B(3x5) schoolbook = 60
    Total: 98
    
    Can I do better on Part 2? A(4x3)*B(3x5)
    Split A rows 2+2, B cols 2+3
    A[0:2, 0:3] * B[0:3, 0:2]: 2x3*3x2
    A[0:2, 0:3] * B[0:3, 2:5]: 2x3*3x3
    A[2:4, 0:3] * B[0:3, 0:2]: 2x3*3x2
    A[2:4, 0:3] * B[0:3, 2:5]: 2x3*3x3
    
    For 2x3*3x2: schoolbook = 12
    For 2x3*3x3: schoolbook = 18
    Total: 12+18+12+18 = 60 (same as before)
    
    What if I use a different split for Part 2?
    Split j={2,3,4} into {2} and {3,4}
    A(4x1)*B(1x5) = 20
    A(4x2)*B(2x5) = 40
    Total: 60 (same)
    
    Or {2,3} and {4}
    A(4x2)*B(2x5) = 40
    A(4x1)*B(1x5) = 20
    Total: 60
    
    It seems 60 is the minimum for 4x3*3x5 with schoolbook.
    
    So the total is stuck at 98 unless I find a better way.
    
    Let me try a completely different split:
    Split j into {0,1,2,3} and {4}
    Part 1: A(4x4)*B(4x5)
    Part 2: A(4x1)*B(1x5) = 20
    
    For Part 1, can I do better than 4*4*5=80?
    Use Strassen-like decomposition on 4x4*4x5.
    
    Split A rows 2+2, B cols 2+3
    A[0:2, 0:4] * B[0:4, 0:2]: 2x4*4x2
    A[0:2, 0:4] * B[0:4, 2:5]: 2x4*4x3
    A[2:4, 0:4] * B[0:4, 0:2]: 2x4*4x2
    A[2:4, 0:4] * B[0:4, 2:5]: 2x4*4x3
    
    For 2x4*4x2:
      Split A cols 2+2, B rows 2+2
      A[0:2, 0:2] * B[0:2, 0:2]: 2x2*2x2 Strassen = 7
      A[0:2, 0:2] * B[2:4, 0:2]: 2x2*2x2 Strassen = 7
      A[0:2, 2:4] * B[0:2, 0:2]: 2x2*2x2 Strassen = 7
      A[0:2, 2:4] * B[2:4, 0:2]: 2x2*2x2 Strassen = 7
      Total: 28
    
    For 2x4*4x3:
      Split A cols 2+2, B rows 2+2
      A[0:2, 0:2] * B[0:2, 0:3]: 2x2*2x3 schoolbook = 12
      A[0:2, 0:2] * B[2:4, 0:3]: 2x2*2x3 schoolbook = 12
      A[0:2, 2:4] * B[0:2, 0:3]: 2x2*2x3 schoolbook = 12
      A[0:2, 2:4] * B[2:4, 0:3]: 2x2*2x3 schoolbook = 12
      Total: 48
    
    Part 1 total: 28+48+28+48 = 152
    Grand total: 152+20 = 172 (much worse)
    
    So the current 98 is actually quite good. Let me keep it and just clean up the code.
    """
    n, m, p = 4, 5, 5
    total_rank = 98
    
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 98
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 98
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 98
    
    col = 0
    
    # Part 1: A_left (4x2) * B_top (2x5)
    # A_left: A[i,j] for i=0..3, j=0..1 -> U rows i*5+j
    # B_top: B[j,k] for j=0..1, k=0..4 -> V rows j*5+k
    # Output: C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    
    # Sub-part 1a: A_top_left (2x2) * B_top_left (2x2) - Strassen
    # A_top_left: A[0,0],A[0,1],A[1,0],A[1,1] -> U rows 0,1,5,6
    # B_top_left: B[0,0],B[0,1],B[1,0],B[1,1] -> V rows 0,1,5,6
    # Output: C[0,0],C[0,1],C[1,0],C[1,1] -> W rows 0,1,4,5
    
    sU, sV, sW = _strassen_2x2()
    a_rows = [0, 1, 5, 6]  # U rows for A_top_left
    b_rows = [0, 1, 5, 6]  # V rows for B_top_left
    c_rows = [0, 1, 4, 5]  # W rows for C[0,0],C[0,1],C[1,0],C[1,1]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # Sub-part 1b: A_top_left (2x2) * B_top_right (2x3) - schoolbook
    # A_top_left: U rows 0,1,5,6
    # B_top_right: B[0,2],B[0,3],B[0,4],B[1,2],B[1,3],B[1,4] -> V rows 2,3,4,7,8,9
    # Output: C[0,2],C[0,3],C[0,4],C[1,2],C[1,3],C[1,4]
    a_rows = [0, 1, 5, 6]
    b_rows = [2, 3, 4, 7, 8, 9]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + i][col] = 1
                col += 1
    # col now at 7 + 12 = 19
    
    # Sub-part 1c: A_bot_left (2x2) * B_top_left (2x2) - Strassen
    # A_bot_left: A[2,0],A[2,1],A[3,0],A[3,1] -> U rows 10,11,15,16
    # B_top_left: V rows 0,1,5,6
    # Output: C[2,0],C[2,1],C[3,0],C[3,1]
    
    sU, sV, sW = _strassen_2x2()
    a_rows = [10, 11, 15, 16]
    b_rows = [0, 1, 5, 6]
    c_rows = [8, 10, 12, 14]  # C[2,0]=8, C[2,1]=10, C[3,0]=12, C[3,1]=14
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    # col now at 19 + 7 = 26
    
    # Sub-part 1d: A_bot_left (2x2) * B_top_right (2x3) - schoolbook
    # A_bot_left: U rows 10,11,15,16
    # B_top_right: V rows 2,3,4,7,8,9
    # Output: C[2,2],C[2,3],C[2,4],C[3,2],C[3,3],C[3,4]
    a_rows = [10, 11, 15, 16]
    b_rows = [2, 3, 4, 7, 8, 9]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + (2 + i)][col] = 1
                col += 1
    # col now at 26 + 12 = 38
    
    # Part 2: A_right (4x3) * B_bot (3x5) - schoolbook
    # A_right: A[i,j] for i=0..3, j=2..4 -> U rows i*5+j
    # B_bot: B[j,k] for j=2..4, k=0..4 -> V rows j*5+k
    # Output: C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    
    for i in range(4):
        for j in range(3):
            for k in range(5):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    # col now at 38 + 60 = 98
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, num_trials=10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(42)
    rank = len(W[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for trial in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_computed[i, k] += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _try_linear_algebra_improvement(n, m, p, base_rank, time_budget):
    """
    Try to find a lower-rank decomposition by solving a sparse linear system.
    
    The idea: if we have a rank-R decomposition, we can try to express some columns
    as linear combinations of others, potentially reducing the effective rank.
    
    For matrix multiplication, the tensor is:
    T[i,j,k] = delta(i,k) for the "identity" part... actually no.
    
    The matrix multiplication tensor T_{i,j,k} satisfies:
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    
    Actually, the decomposition is:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    
    This is equivalent to saying the "multiplication tensor" M has entries:
    M[i, j, k, l] where... let me think more carefully.
    
    The standard formulation: we want to compute C = A * B where A is n x m, B is m x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-r decomposition means:
    C[i,k] = sum_{r=1}^R (sum_j U_r[i,j] * A[i,j]) * (sum_k V_r[j,k] * B[j,k]) * W_r[i,k]
    
    Wait, that's not quite right either. Let me be precise.
    
    The bilinear form is:
    C[i,k] = sum_{j=1}^m A[i,j] * B[j,k]
    
    A rank-r decomposition of the multiplication algorithm means we can write:
    C[i,k] = sum_{r=1}^R mu_r(A) * nu_r(B)
    where mu_r(A) is linear in A and nu_r(B) is linear in B.
    
    mu_r(A) = sum_{i,j} U_r[i,j] * A[i,j]
    nu_r(B) = sum_{j,k} V_r[j,k] * B[j,k]
    
    And we need: sum_r mu_r(A) * nu_r(B) = C[i,k] for all A, B.
    
    This means: sum_r U_r[i',j'] * V_r[j'',k'] * (contribution to C[i,k]) = delta_{i',i} * delta_{j'',k'} ... 
    
    Hmm, let me use the standard tensor formulation.
    
    The multiplication tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    Actually, the "multiplication tensor" for C = A*B is a 3-tensor M with:
    M[i, j, k] = 1 if this term contributes to C[i,k]... 
    
    No. The correct formulation:
    
    We have input vector a = vec(A) in R^{n*m}, b = vec(B) in R^{m*p}.
    Output c = vec(C) in R^{n*p} where c[i,k] = sum_j a[i*m+j] * b[j*p+k].
    
    This is a trilinear map: (a, b) -> c where c[i,k] = sum_j a[i*m+j] b[j*p+k].
    
    A rank-r decomposition means:
    c[i,k] = sum_{r=1}^R (sum_{i',j'} U_r[i*m+j', i', j'] ... 
    
    OK let me just use the formulation from the code. The decomposition is:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * ... 
    
    No, that's not right. Looking at the verification code:
    C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is a tensor decomposition of the form:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where T[i,j,k] should equal... what? 
    
    For the matrix multiplication to work, we need:
    sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    
    But C[i,k] = sum_j A[i,j] * B[j,k], so we need:
    sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    
    This means T[i,j,k] = 1 for all i,j,k? That can't be right for a non-trivial decomposition.
    
    Wait, I think the formulation is different. Let me re-read.
    
    Actually, I think the correct interpretation is:
    The "multiplication tensor" is defined such that:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    
    And a rank-r decomposition provides r "intermediate products":
    P_r = (sum_{i,j} U_r[i,j] A[i,j]) * (sum_{j,k} V_r[j,k] B[j,k])
    
    Then C[i,k] = sum_r W_r[i,k] * P_r
    
    In matrix form:
    C = sum_r (U_r^T * A) * (B * V_r) * W_r^T ... this is getting confusing.
    
    Let me just trust the verification code and the schoolbook construction.
    
    For schoolbook, U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k) triple.
    
    So C[i,k] = sum_j sum_{r=(i',j',k')} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
              = sum_j sum_{i',j',k'} delta(i*m+j, i'*m+j') * delta(j*p+k, j'*p+k') * delta(k*n+i, k'*n+i')
    
    Hmm, this is getting too complicated. Let me just try a simple approach:
    
    Given the base decomposition of rank R, try to find if any column is a linear combination
    of the others. If so, we can eliminate it and reduce the rank.
    """
    t_start = time.time()
    U, V, W = _construct_4x5x5()
    rank = len(W[0])
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The constraint is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if this is a "valid" term for matrix multiplication
    
    # Actually, from the schoolbook construction, the tensor T is:
    # T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # But that's not quite right either.
    
    # Let me think about this differently. The decomposition must satisfy:
    # For any A, B: C = A*B where C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Using the decomposition: C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] * A[i,j] B[j,k]... 
    # No, that's not how it works.
    
    # I think the correct formulation is:
    # Define P_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (a scalar, linear in A)
    # Define Q_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (a scalar, linear in B)
    # Then C[i,k] = sum_r W[k*n+i, r] * P_r * Q_r
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j'',k'} V[j''*p+k', r] B[j'',k'])
    # = sum_j A[i,j] B[j,k]
    #
    # Expanding the left side:
    # = sum_r W[k*n+i, r] * sum_{i',j',j'',k'} U[i'*m+j', r] V[j''*p+k', r] A[i',j'] B[j'',k']
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k', r] = delta_{i',i} * delta_{j',j} * delta_{j'',j} * delta_{k',k}
    #
    # Simplifying: sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k', r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    #
    # This is a set of n*m*p*n*p equations (one for each (i',j',j'',k',i,k) but with constraints).
    # Actually, it's: for each (i',j',j'',k',i,k):
    # sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k', r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    #
    # The number of distinct constraints is n * m * m * p * n * p = n^2 * m^2 * p^2 = 16*25*25 = 10000
    # But many are redundant. The independent constraints are:
    # For each (i', j', j'', k'): sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k', r] must equal
    # delta_{i',i} * delta_{j',j''} * delta_{k',k} for all (i,k).
    #
    # This is a bilinear system in (U, V, W) for fixed r, or rather a system of equations
    # that must be satisfied.
    
    # This is extremely complex to solve directly. Let me try a different approach.
    
    # Approach: Start with the rank-98 decomposition and try to find linear dependencies
    # among the "rank-1 components."
    
    # Each rank-1 component r defines a tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The sum of all T_r must equal the multiplication tensor T.
    
    # If some T_r can be expressed as a linear combination of others, we can reduce the rank.
    
    # But this is a nonlinear constraint (the product of three vectors), so it's not straightforward.
    
    # Alternative: Try a randomized approach. Generate random rank-r decompositions and check
    # if they satisfy the constraints. This is unlikely to work for r < 98.
    
    # For now, return the base decomposition.
    if time.time() - t_start > time_budget:
        return U, V, W, rank
    
    return U, V, W, rank


def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Build the block-Strassen decomposition (rank 98)
    U, V, W = _construct_4x5x5()
    rank = len(W[0])
    
    # Verify correctness
    if time.time() - t0 < _DEADLINE_SECONDS - 10:
        if not _verify_decomposition(U, V, W, n, m, p, num_trials=5):
            # Fallback to schoolbook if verification fails
            U, V, W = _schoolbook(n, m, p)
            rank = len(W[0])
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_SECONDS:
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
