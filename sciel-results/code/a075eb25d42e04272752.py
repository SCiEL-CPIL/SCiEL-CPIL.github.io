import numpy as np
import random
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def check_decomposition(U, V, W, n, m, p, rank):
    """Check if the tensor identity holds exactly using Fraction arithmetic."""
    U_f = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
    V_f = [[Fraction(V[j][r]) for r in range(rank)] for j in range(m*p)]
    W_f = [[Fraction(W[k][r]) for r in range(rank)] for k in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_f[i*m+j][r] * V_f[j*p+k][r] * W_f[k*n+i][r]
                if s != 1:
                    return False
    return True

def check_decomposition_float(U, V, W, n, m, p, rank, tol=1e-6):
    """Fast float check for pruning."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7) using the standard Strassen formula.
    
    Strassen's 7 products for C = A*B where A,B,C are 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    
    p0 = (a+d)(e+h)
    p1 = (c+d)*e
    p2 = a*(f-h)
    p3 = d*(g-e)
    p4 = (a+b)*h
    p5 = (c-d)*(e+f)
    p6 = (b-d)*(g+h)
    
    C = [[p0+p3-p4+p6, p1+p4],
         [p1+p5,       p0-p1+p2+p5]]
    
    Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]  # A entries: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V = [[0]*rank for _ in range(4)]  # B entries: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    W = [[0]*rank for _ in range(4)]  # C entries: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # p0 = (a+d)(e+h)
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    W[0][0] = 1  # C[0,0] += p0
    W[3][0] = 1  # C[1,1] += p0
    
    # p1 = (c+d)*e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    W[1][1] = 1  # C[0,1] += p1
    W[2][1] = 1  # C[1,0] += p1
    
    # p2 = a*(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1 # h
    W[3][2] = 1  # C[1,1] += p2
    
    # p3 = d*(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1 # e
    W[0][3] = 1  # C[0,0] += p3
    
    # p4 = (a+b)*h
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    W[0][4] = -1 # C[0,0] -= p4
    W[1][4] = 1  # C[0,1] += p4
    
    # p5 = (c-d)*(e+f)
    U[2][5] = 1  # c
    U[3][5] = -1 # d
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    W[2][5] = 1  # C[1,0] += p5
    W[3][5] = 1  # C[1,1] += p5
    
    # p6 = (b-d)*(g+h)
    U[1][6] = 1  # b
    U[3][6] = -1 # d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    W[0][6] = 1  # C[0,0] += p6
    
    return U, V, W

def build_1x3x3_decomp():
    """
    Build a rank-3 decomposition for 1x3 x 3x3 -> 3x1.
    This is just a vector-matrix product: y = x * M where x is 1x3, M is 3x3, y is 3x1.
    y[k] = sum_j x[j] * M[j,k]
    
    For a rank-r decomposition, we need:
    sum_{r=1}^{rank} U[j,r] * V[j,k,r] * W[k,r] = x[j]*M[j,k] for all j,k
    
    Actually, for 1x3 x 3x3, the tensor is T[j,k] = x[j]*M[j,k].
    The standard rank is 3*3=9, but we can do better.
    
    Wait, let me reconsider. For multiplying a 1x3 vector by a 3x3 matrix:
    The output is a 1x3 vector.
    y[k] = sum_{j=0}^{2} x[j] * M[j,k]
    
    The tensor T has indices (j, k) with T[j,k] = x[j]*M[j,k].
    But x and M are separate inputs. So the multiplication tensor has:
    Input 1: x (1x3), indices j=0,1,2
    Input 2: M (3x3), indices (j,k) for j=0,1,2 and k=0,1,2
    Output: y (1x3), indices k=0,1,2
    
    The tensor is T[j, j', k] where j indexes x, (j',k) indexes M, and k indexes y.
    T[j, j', k] = 1 if j=j' else 0.
    
    So T[j, j', k] = delta(j, j') for all j, j', k.
    
    This is a 3x9x3 tensor. The rank of this tensor...
    Actually, this is just the identity embedding. T[j, j', k] = delta_{j,j'}.
    
    For a rank-r decomposition:
    sum_{r=1}^r U[j,r] * V[j',k,r] * W[k,r] = delta_{j,j'}
    
    One simple rank-3 decomposition:
    For r=0,1,2 (corresponding to j=0,1,2):
    U[j,r] = delta_{j,r}
    V[j',k,r] = delta_{j',r} (independent of k)
    W[k,r] = 1 (independent of k)
    
    Check: sum_r delta_{j,r} * delta_{j',r} * 1 = delta_{j,j'} ✓
    
    But wait, V[j',k,r] = delta_{j',r} means V doesn't depend on k. That's fine.
    And W[k,r] = 1 means W doesn't depend on k either.
    
    So rank 3 works for 1x3 x 3x3 -> 1x3.
    
    Actually wait, I need to double-check the dimensions.
    n=1, m=3, p=3.
    U has shape (n*m, rank) = (3, rank)
    V has shape (m*p, rank) = (9, rank)
    W has shape (p*n, rank) = (3, rank)
    
    The tensor identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    With n=1: i can only be 0.
    sum_r U[0*3+j, r] * V[j*3+k, r] * W[k*1+0, r] = sum_r U[j,r] * V[j*3+k, r] * W[k,r] = 1
    
    So we need: sum_r U[j,r] * V[j*3+k, r] * W[k,r] = 1 for all j,k in {0,1,2}.
    
    With the decomposition I described:
    U[j,r] = delta_{j,r} for r=0,1,2
    V[j*3+k, r] = delta_{j,r} (independent of k)
    W[k,r] = 1
    
    sum_r delta_{j,r} * delta_{j,r} * 1 = sum_r delta_{j,r}^2 = 1 (since exactly one r equals j)
    
    Yes, this works! Rank 3 for 1x3 x 3x3.
    """
    rank = 3
    U = [[0]*rank for _ in range(3)]   # shape (3, 3), U[j, r]
    V = [[0]*rank for _ in range(9)]   # shape (9, 3), V[j*3+k, r]
    W = [[0]*rank for _ in range(3)]   # shape (3, 3), W[k, r]
    
    for r in range(3):
        U[r][r] = 1
        for k in range(3):
            V[r*3+k][r] = 1
            W[k][r] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    global_start = time.time()
    global_time_limit = 140  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 18
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": U_s}
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try to build a structured rank-15 decomposition
    # Strategy: decompose 2x3 x 3x3 -> 2x3 as a direct sum
    # 
    # The multiplication: C = A * B where A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can split A into rows: A = [A_0; A_1] where A_0, A_1 are 1x3
    # Then C_0 = A_0 * B and C_1 = A_1 * B
    #
    # Each C_i = A_i * B is a 1x3 x 3x3 -> 1x3 multiplication, rank 3 each.
    # Total rank: 3 + 3 = 6? No, that's not right because B is shared.
    #
    # Actually, the tensor for 2x3 x 3x3 has:
    # Input 1: A (2x3), indexed by (i,j)
    # Input 2: B (3x3), indexed by (j,k)  
    # Output: C (2x3), indexed by (i,k)
    #
    # T[i,j, j',k, i',k'] = delta(j,j') * delta(i,i') * delta(k,k')
    # 
    # For a rank-r decomposition:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # 
    # Let me think about this differently. The tensor T has entries T[i,j,k] = 1
    # for all i in {0,1}, j in {0,1,2}, k in {0,1,2}. Wait, that's not right either.
    #
    # The matrix multiplication tensor for (n,m,p) = (2,3,3) is:
    # T in R^{(n*m) x (m*p) x (p*n)}
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and i=i' and k=k', else 0
    #
    # So T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # i.e., the j-index of a matches the j'-index of b, and the i-index of a matches
    # the i'-index of c, and the k-index of b matches the k'-index of c.
    #
    # In other words, T[i*m+j, j*p+k, k*n+i] = 1 and all other entries are 0.
    #
    # For a rank-r decomposition:
    # sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    #
    # And the decomposition should be "exact" meaning it reproduces the tensor.
    #
    # Now, let's think about a structured approach.
    # 
    # Consider splitting by the j-index. For each j in {0,1,2}, we have a "slice"
    # where A[i,j] * B[j,k] contributes to C[i,k].
    #
    # For a fixed j, the contribution is:
    # A[i,j] * B[j,k] for all i,k
    # This is a rank-1 tensor in the (i,k) space for each j.
    #
    # But we need to handle the fact that A[i,j] and B[j,k] are separate factors.
    #
    # Let me try a different decomposition. Consider:
    # For each pair (i,j) in A and each k in B's column:
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to sum to 1.
    #
    # Simple rank-18 (schoolbook): one term per (i,j,k).
    #
    # Can we do better? Let's try rank 15.
    #
    # Idea: Use Strassen-like tricks on the 2x2 submatrix.
    # 
    # Split A into A[:, :2] (2x2) and A[:, 2] (2x1)
    # Split B into B[:2, :] (2x3) and B[2, :] (1x3)
    #
    # C = A[:, :2] * B[:2, :] + A[:, 2] * B[2, :]
    #
    # The first term is 2x2 x 2x3 -> 2x3
    # The second term is 2x1 x 1x3 -> 2x3
    #
    # For 2x2 x 2x3: 
    #   C[i,k] = sum_{j=0,1} A[i,j] * B[j,k]
    #   This can be decomposed. The tensor is 4x6x6.
    #   Rank? At most 4*6=24, but likely less.
    #
    # Actually, let me just try to construct a rank-15 decomposition directly.
    #
    # Alternative approach: use the fact that 2x3 x 3x3 can be viewed as
    # two independent 1x3 x 3x3 multiplications (one for each row of A),
    # but they share the same B.
    #
    # If B were separate for each row, each would be rank 3, total rank 6.
    # But B is shared, so we need to account for that.
    #
    # Let me try a different structured approach.
    #
    # For the 2x3 x 3x3 multiplication, consider the following rank-15 ansatz:
    # 
    # Use 3 "Strassen-like" blocks, each handling a pair of columns of B.
    # 
    # Actually, let me try the most straightforward structured decomposition:
    #
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # For each j, define a "j-slice" contribution. The j-slice involves
    # A[:, j] (a 2-vector) and B[j, :] (a 3-vector), producing a 2x3 outer product.
    #
    # The outer product A[:,j] * B[j,:] has rank 1 as a matrix, but as a tensor
    # in the decomposition, it requires rank 1 if we can factor it.
    #
    # For a single j: we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to
    # reproduce A[i,j]*B[j,k] for all i,k.
    #
    # This is a 2x3 x 3x2 tensor (if we think of it as A_col * B_row).
    # The rank of this tensor... A[i,j]*B[j,k] for fixed j is just
    # (A[0,j]*B[j,0], A[0,j]*B[j,1], A[0,j]*B[j,2],
    #  A[1,j]*B[j,0], A[1,j]*B[j,1], A[1,j]*B[j,2])
    # 
    # This is a rank-1 tensor in the (i,k) space for each j, but since A and B
    # are separate inputs, the tensor T_j[i, j, k] = A[i,j]*B[j,k] has rank...
    #
    # Actually, for the full tensor, each (i,j,k) triple has T=1.
    # 
    # Let me just try to find a rank-15 decomposition by solving a system.
    #
    # For rank r, we have:
    # U: 6 x r, V: 9 x r, W: 6 x r
    # Total variables: 6r + 9r + 6r = 21r
    # Constraints: 2*3*3 = 18 (one per (i,j,k) triple)
    #
    # For r=15: 315 variables, 18 constraints. Very underdetermined.
    #
    # Let me try a structured rank-15 decomposition.
    #
    # Approach: Split the 3 columns of B into groups.
    # Group 1: k=0,1 (2 columns)
    # Group 2: k=2 (1 column)
    #
    # For Group 1 (k in {0,1}):
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    #
    # This is a 2x3 x 2x3 -> 2x2 multiplication (if we restrict to k in {0,1}).
    # Actually no, it's still 2x3 x 3x2 -> 2x2.
    #
    # Let me try yet another approach. Use the fact that:
    # 
    # For 2x3 x 3x3, we can write:
    # C = A * B
    # 
    # Let A = [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # Let B = [B1; B2] where B1 is 2x3 and B2 is 1x3
    # 
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 x 2x3 -> 2x3
    # A2*B2: 2x1 x 1x3 -> 2x3
    #
    # For A2*B2: C[i,k] = A2[i] * B2[k]
    # This is a rank-1 matrix (outer product). As a tensor:
    # T[i, j, k] where j indexes A2 (j=0 only, since A2 is 2x1, so j is the row index)
    # Wait, A2 is 2x1, so it has entries A2[0] and A2[1].
    # B2 is 1x3, so it has entries B2[0], B2[1], B2[2].
    #
    # In the original indexing:
    # A2[i] = A[i, 2] (the third column of A)
    # B2[k] = B[2, k] (the third row of B)
    #
    # So C[i,k] += A[i,2] * B[2,k]
    #
    # In the tensor: T[i*m+2, 2*p+k, k*n+i] = 1 for all i,k.
    # That's 2*3 = 6 entries.
    #
    # For this sub-tensor, we need:
    # sum_r U[i*3+2, r] * V[2*3+k, r] * W[k*2+i, r] = 1 for all i in {0,1}, k in {0,1,2}
    #
    # A rank-3 decomposition:
    # For r=0,1,2:
    # U[i*3+2, r] = 1 for all i (or some pattern)
    # V[2*3+k, r] = 1 for all k
    # W[k*2+i, r] = delta_{i,r} ... but i only goes to 1, and r goes to 2.
    #
    # Hmm, this doesn't quite work. Let me think more carefully.
    #
    # We need: sum_r U[i, r] * V[k, r] * W[i,k, r] = 1 for all i,k
    # where U has 2 rows (for i=0,1), V has 3 rows (for k=0,1,2), W has 6 rows.
    #
    # Actually W is indexed by (k,i) so W[k*2+i, r].
    #
    # For rank 2:
    # r=0: U[0,0]=1, U[1,0]=1, V[k,0]=1 for all k, W[k*2+0,0]=1, W[k*2+1,0]=0
    # r=1: U[0,1]=0, U[1,1]=1, V[k,1]=1 for all k, W[k*2+0,1]=0, W[k*2+1,1]=1
    #
    # Check for i=0: sum_r U[0,r]*V[k,r]*W[k*2+0,r] = 1*1*1 + 0*1*0 = 1 ✓
    # Check for i=1: sum_r U[1,r]*V[k,r]*W[k*2+1,r] = 1*1*0 + 1*1*1 = 1 ✓
    #
    # So rank 2 works for the A2*B2 part!
    #
    # Now for A1*B1: 2x2 x 2x3 -> 2x3
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2}
    #
    # In the tensor: T[i*m+j, j*p+k, k*n+i] = 1 for j in {0,1}, all i,k.
    # That's 2*2*3 = 12 entries.
    #
    # We need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for j in {0,1}, all i,k
    #
    # This is a 2x2 x 2x3 -> 2x3 multiplication.
    # 
    # For 2x2 x 2x2, Strassen gives rank 7.
    # For 2x2 x 2x3, the rank is at most 7*... hmm.
    #
    # Actually, 2x2 x 2x3 can be seen as two independent 2x2 x 2x1 multiplications
    # (one for each column of the 2x3 output), sharing the same A and B.
    # Each 2x2 x 2x1 has rank at most 4 (schoolbook) or possibly less.
    #
    # 2x2 x 2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # T[i, j, k] where i in {0,1}, j in {0,1}, k=0 (only one output)
    # T[i,j,0] = 1 for all i,j.
    #
    # We need: sum_r U[i*2+j, r] * V[j, r] * W[0*2+i, r] = 1 for all i,j.
    # U: 4 x r, V: 2 x r, W: 2 x r.
    #
    # Rank 4 (schoolbook): one term per (i,j).
    # Can we do rank 3? 
    # 
    # For 2x2 x 2x1, the tensor is just the "identity" in some sense.
    # T[i,j,0] = 1 for all i,j.
    #
    # This is a 4x2x2 tensor with all entries 1.
    # The rank of the all-ones tensor of shape (a,b,c) is... it's a rank-1 tensor!
    # T[i,j,k] = 1 = 1*1*1.
    #
    # Wait, that means rank 1 suffices for 2x2 x 2x1?
    # U[i*2+j, 0] = 1 for all i,j
    # V[j, 0] = 1 for all j
    # W[i, 0] = 1 for all i
    #
    # Check: sum_r U[i*2+j,r]*V[j,r]*W[i,r] = 1*1*1 = 1 ✓
    #
    # Yes! Rank 1 for 2x2 x 2x1.
    #
    # So for 2x2 x 2x3 (two columns), we can use rank 1+1 = 2?
    # No, because the two columns share the same A and B factors.
    #
    # For column k=0: sum_r U[i*2+j, r] * V[j, r] * W[i, r] = 1
    # For column k=1: sum_r U[i*2+j, r] * V[j, r] * W'[i, r] = 1
    #
    # But in our problem, W is shared (it's the same W for both columns).
    # So we need a single W that works for both.
    #
    # For 2x2 x 2x3:
    # T[i, j, k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    #
    # In the full tensor indexing:
    # U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
    #
    # For j in {0,1}, i in {0,1}, k in {0,1,2}:
    # sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    #
    # Let's try rank 3:
    # r=0: U[i*3+0, 0] = 1 for all i, V[0*3+k, 0] = 1 for all k, W[k*2+0, 0] = 1, W[k*2+1, 0] = 0
    #        U[i*3+1, 0] = 0 for all i, V[1*3+k, 0] = 1 for all k, (not used since U is 0)
    # r=1: U[i*3+0, 1] = 0, V[0*3+k, 1] = 0, (not used)
    #        U[i*3+1, 1] = 1 for all i, V[1*3+k, 1] = 1 for all k, W[k*2+0, 1] = 0, W[k*2+1, 1] = 1
    # r=2: ...
    #
    # This doesn't work because W is shared.
    #
    # Let me try a different approach. Use the fact that for 2x2 x 2x2, Strassen gives rank 7.
    # For 2x2 x 2x3, we can use two overlapping Strassen decompositions.
    #
    # Actually, let me just try to construct a rank-15 decomposition numerically.
    #
    # For rank r=15, we have 21*15 = 315 variables and 18 constraints.
    # We can set up a system and solve it.
    #
    # But first, let me try a simpler structured approach.
    #
    # Total rank = rank(A1*B1) + rank(A2*B2) = rank(2x2 x 2x3) + 2
    #
    # If rank(2x2 x 2x3) = 13, total = 15.
    #
    # For 2x2 x 2x3, the tensor has 2*2*3 = 12 non-zero entries.
    # Schoolbook rank = 12.
    # Can we do better?
    #
    # 2x2 x 2x2 has rank 7 (Strassen).
    # 2x2 x 2x3 can be seen as 2x2 x 2x2 + 2x2 x 2x1 (the extra column).
    # But they share A and B.
    #
    # Let me try: use Strassen for the first two columns (rank 7) and rank 1 for the third column.
    # Total: 7 + 1 = 8 for 2x2 x 2x3.
    # Total for 2x3 x 3x3: 8 + 2 = 10.
    #
    # But wait, the Strassen decomposition for 2x2 x 2x2 uses specific patterns in U, V, W.
    # Can I extend it to 2x2 x 2x3?
    #
    # For 2x2 x 2x2 (columns 0,1):
    # Strassen gives 7 terms.
    #
    # For column 2:
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # This is a 2x2 x 2x1 multiplication, rank 1.
    #
    # So total rank for 2x2 x 2x3: 7 + 1 = 8.
    # Total for 2x3 x 3x3: 8 + 2 = 10.
    #
    # Let me construct this!
    
    if time.time() - global_start > global_time_limit - 30:
        return best_result
    
    # Construct rank-10 decomposition
    rank_total = 10
    U = [[0]*rank_total for _ in range(n*m)]  # 6 x 10
    V = [[0]*rank_total for _ in range(m*p)]  # 9 x 10
    W = [[0]*rank_total for _ in range(p*n)]  # 6 x 10
    
    # Part 1: A1*B1 (2x2 x 2x3 -> 2x3), using Strassen for cols 0,1 and rank-1 for col 2
    # Strassen for 2x2 x 2x2 (cols 0,1)
    # In the full indexing:
    # A[i,j] for i in {0,1}, j in {0,1} -> U[i*3+j, r]
    # B[j,k] for j in {0,1}, k in {0,1} -> V[j*3+k, r]
    # C[i,k] for i in {0,1}, k in {0,1} -> W[k*2+i, r]
    
    # Strassen 7 terms for 2x2 x 2x2:
    # p0 = (a+d)(e+h)
    U[0*3+0][0] = 1  # A[0,0]
    U[1*3+1][0] = 1  # A[1,1]
    V[0*3+0][0] = 1  # B[0,0]
    V[1*3+1][0] = 1  # B[1,1]
    W[0*2+0][0] = 1  # C[0,0]
    W[1*2+1][0] = 1  # C[1,1]
    
    # p1 = (c+d)*e
    U[1*3+0][1] = 1  # A[1,0]
    U[1*3+1][1] = 1  # A[1,1]
    V[0*3+0][1] = 1  # B[0,0]
    W[0*2+1][1] = 1  # C[1,0]
    W[1*2+0][1] = 1  # C[0,1]
    
    # p2 = a*(f-h)
    U[0*3+0][2] = 1  # A[0,0]
    V[0*3+1][2] = 1  # B[0,1]
    V[1*3+1][2] = -1 # B[1,1]
    W[1*2+1][2] = 1  # C[1,1]
    
    # p3 = d*(g-e)
    U[1*3+1][3] = 1  # A[1,1]
    V[1*3+0][3] = 1  # B[1,0]
    V[0*3+0][3] = -1 # B[0,0]
    W[0*2+0][3] = 1  # C[0,0]
    
    # p4 = (a+b)*h
    U[0*3+0][4] = 1  # A[0,0]
    U[0*3+1][4] = 1  # A[0,1]
    V[1*3+1][4] = 1  # B[1,1]
    W[0*2+0][4] = -1 # C[0,0]
    W[0*2+1][4] = 1  # C[1,0]
    
    # p5 = (c-d)*(e+f)
    U[1*3+0][5] = 1  # A[1,0]
    U[1*3+1][5] = -1 # A[1,1]
    V[0*3+0][5] = 1  # B[0,0]
    V[0*3+1][5] = 1  # B[0,1]
    W[1*2+0][5] = 1  # C[0,1]
    W[1*2+1][5] = 1  # C[1,1]
    
    # p6 = (b-d)*(g+h)
    U[0*3+1][6] = 1  # A[0,1]
    U[1*3+1][6] = -1 # A[1,1]
    V[1*3+0][6] = 1  # B[1,0]
    V[1*3+1][6] = 1  # B[1,1]
    W[0*2+0][6] = 1  # C[0,0]
    
    # Column 2 of A1*B1: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Rank 1:
    U[0*3+0][7] = 1  # A[0,0]
    U[0*3+1][7] = 1  # A[0,1]
    U[1*3+0][7] = 1  # A[1,0]
    U[1*3+1][7] = 1  # A[1,1]
    V[0*3+2][7] = 1  # B[0,2]
    V[1*3+2][7] = 1  # B[1,2]
    W[2*2+0][7] = 1  # C[0,2]
    W[2*2+1][7] = 1  # C[1,2]
    
    # Part 2: A2*B2 (2x1 x 1x3 -> 2x3), rank 2
    # C[i,k] += A[i,2]*B[2,k]
    # r=8: handles i=0
    U[0*3+2][8] = 1  # A[0,2]
    V[2*3+0][8] = 1  # B[2,0]
    V[2*3+1][8] = 1  # B[2,1]
    V[2*3+2][8] = 1  # B[2,2]
    W[0*2+0][8] = 1  # C[0,0]
    W[1*2+0][8] = 1  # C[0,1]
    W[2*2+0][8] = 1  # C[0,2]
    
    # r=9: handles i=1
    U[1*3+2][9] = 1  # A[1,2]
    V[2*3+0][9] = 1  # B[2,0]
    V[2*3+1][9] = 1  # B[2,1]
    V[2*3+2][9] = 1  # B[2,2]
    W[0*2+1][9] = 1  # C[1,0]
    W[1*2+1][9] = 1  # C[1,1]
    W[2*2+1][9] = 1  # C[1,2]
    
    # Verify the decomposition
    if check_decomposition_float(U, V, W, n, m, p, rank_total):
        if check_decomposition(U, V, W, n, m, p, rank_total):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank_total
    
    if time.time() - global_start > global_time_limit - 20:
        return best_result
    
    # Try to find a rank-9 decomposition by combining the column-2 term with Strassen
    # or by finding a more efficient decomposition.
    # 
    # For now, try random optimization from the rank-10 solution to see if we can
    # reduce to rank 9.
    
    # Try rank 9 with random initialization
    for target_rank in [9, 10, 11, 12]:
        if time.time() - global_start > global_time_limit - 10:
            break
        if target_rank >= best_rank:
            break
        
        for attempt in range(10):
            if time.time() - global_start > global_time_limit - 5:
                break
            
            U_r = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V_r = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W_r = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Simple gradient descent
            U_f = np.array(U_r, dtype=float)
            V_f = np.array(V_r, dtype=float)
            W_f = np.array(W_r, dtype=float)
            
            for it in range(200):
                if time.time() - global_start > global_time_limit - 5:
                    break
                
                # Compute residuals
                residuals = np.zeros((n, m, p))
                for r in range(target_rank):
                    u = U_f[:, r].reshape(n, m)
                    v = V_f[:, r].reshape(m, p)
                    w = W_f[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residuals[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                residuals -= 1.0
                
                if np.max(np.abs(residuals)) < 1e-4:
                    break
                
                # Gradient step
                lr = 0.001
                grad_U = np.zeros_like(U_f)
                grad_V = np.zeros_like(V_f)
                grad_W = np.zeros_like(W_f)
                
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res = residuals[i, j, k]
                                grad_U[i*m+j, r] += 2 * res * V_f[j*p+k, r] * W_f[k*n+i, r]
                                grad_V[j*p+k, r] += 2 * res * U_f[i*m+j, r] * W_f[k*n+i, r]
                                grad_W[k*n+i, r] += 2 * res * U_f[i*m+j, r] * V_f[j*p+k, r]
                
                U_f -= lr * grad_U
                V_f -= lr * grad_V
                W_f -= lr * grad_W
            
            U_list = U_f.tolist()
            V_list = V_f.tolist()
            W_list = W_f.tolist()
            
            if check_decomposition_float(U_list, V_list, W_list, n, m, p, target_rank):
                if check_decomposition(U_list, V_list, W_list, n, m, p, target_rank):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
    
    return best_result
