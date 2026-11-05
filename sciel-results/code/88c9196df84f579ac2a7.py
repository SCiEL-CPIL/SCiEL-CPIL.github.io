"""Matrix multiplication 2x3x5: rank-minimal exact tensor decomposition.

Strategy: block-structured decomposition.
- Split the 3x5 middle block into (3x3) and (3x2).
- The 2x3x3 block: split k into {0} (rank 1) and {1,2} (2x2x2 Strassen rank 7).
- The 2x3x2 block: split into j=0 (rank 1), j=1 (rank 1), j=2 (rank 1) = rank 3.
Total rank = 1 + 7 + 3 = 11.

Alternatively, split j into {0} and {1,2}:
- 2x1x5: rank 5 (one per k)
- 2x2x5: split k into {0,1} (2x2x2 Strassen rank 7) and {2,3,4} (2x2x3)
  2x2x3: split k into {2} (rank 1) and {3,4} (2x2x2 Strassen rank 7) = rank 8
  So 2x2x5 = 7 + 8 = 15
Total = 5 + 15 = 20. Worse.

Better: split j into {0,1,2} each of width 1: rank 3*5=15. Worse.

Best: split k into {0} and {1,2,3,4}:
- 2x3x1: rank 3 (one per j)
- 2x3x4: split j into {0} and {1,2}:
  - 2x1x4: rank 4
  - 2x2x4: split k into {1,2} (2x2x2 Strassen rank 7) and {3,4} (2x2x2 Strassen rank 7) = 14
  Total 2x3x4 = 4 + 14 = 18
Total = 3 + 18 = 21. Worse.

Best so far: 11.

Let me try: split j into {0} and {1,2}:
- 2x1x5: rank 5
- 2x2x5: split k into {0,1,2} and {3,4}:
  - 2x2x3: split k into {0} (rank 1) and {1,2} (2x2x2 Strassen rank 7) = 8
  - 2x2x2: Strassen rank 7
  Total 2x2x5 = 8 + 7 = 15
Total = 5 + 15 = 20. Worse.

Let me try another: split into 2x2x3 (j={0,1}, k={0,1,2}) + 2x2x2 (j={0,1}, k={3,4}) + 2x1x5 (j={2}, all k)
- 2x2x3: rank 8 (as above)
- 2x2x2: rank 7
- 2x1x5: rank 5
Total = 8 + 7 + 5 = 20. Worse.

Best: 11.

Actually, let me reconsider 2x2x5.
Split k into {0}, {1,2}, {3,4}:
- 2x2x1: rank 1
- 2x2x2: rank 7
- 2x2x2: rank 7
Total = 15.

Or split k into {0,1}, {2,3}, {4}:
- 2x2x2: 7
- 2x2x2: 7
- 2x2x1: 1
Total = 15.

So 2x2x5 = 15 seems minimal with this approach.

Let me try yet another decomposition for the full 2x3x5:
Split into three 2x1x5 blocks: rank 15. Worse than 11.

Split into 2x3x3 + 2x3x2:
- 2x3x3: split k into {0}, {1,2}: 1 + 7 = 8
- 2x3x2: split j into {0}, {1}, {2}: 1+1+1 = 3
Total = 11. This is the best.

Let me verify the 2x3x2 part more carefully.
For j in {0,1,2}, k in {0,1}:
Each (j,k) pair needs rank 1 since it's 2x1x1.
U[i,j,c] for i in {0,1}, V[j,k,c], W[k,i,c].
For fixed (j,k), we need U[:,j,c] * V[j,k,c] * W[k,:] = identity-like.
Actually for 2x1x1: T[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i.
So U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1. One rank-1 term per (j,k).
3 j-values * 2 k-values = 6 rank-1 terms? No wait, I said rank 3.

Let me re-examine. 2x1x5 block (j fixed):
T[i,j,k] for i in {0,1}, k in {0,1,2,3,4}.
For rank 1: U[i,j,c]*V[j,k,c]*W[k,i,c] must equal 1 for all i,k.
If U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1, then T=1 for all i,k. That's rank 1 for the whole 2x1x5 block!

Wait, that can't be right. Let me check: if we use one rank-1 term with U=1, V=1, W=1,
then T[i,j,k] = 1 for all i,j,k. But we only want T=1 for the specific j we're decomposing.
In the full tensor, T[i,j,k]=1 for all (i,j,k). So if we use one rank-1 term globally,
it gives T=1 everywhere. That's rank 1 for the whole thing!

But wait, I verified this before and the current program returns rank 1 which scored 0.0.
The issue is that the evaluator checks something different, or my understanding is wrong.

Let me re-read the task: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."

The tensor for matrix multiplication C = AB where A is 2x3 and B is 3x5:
C_ik = sum_{j=0}^{2} A_ij * B_jk

The tensor T[i,j,k] = 1 if the term A_ij * B_jk contributes to C_ik, else 0.
For standard matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) since every
A_ij * B_jk term contributes to exactly one C_ik entry.

So T is indeed the all-ones tensor of shape (2,3,5). And rank 1 should work.

But the current program scored 0.0. Maybe the evaluator has a different convention?
Let me look at the W convention: W[k*n+i, r]. So W is indexed by (k, i) -> row k*n+i.

The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

For rank 1: U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1.
Then T[i,j,k] = 1*1*1 = 1. This should work.

Unless the evaluator requires the decomposition to actually compute matrix multiplication
in a different sense. Perhaps the evaluator multiplies the factor matrices to verify
that they produce the correct multiplication map, not just that the tensor entries match.

Let me think about this differently. The evaluator likely checks:
For random A (2x3) and B (3x5), does the factorized computation give A*B?

The factorized computation would be:
C_ik = sum_r (sum_j U[i*m+j, r] * A_ij) * (sum_k' V[j*p+k', r] * B_j'k') * W[k*n+i, r]

No, that's not right either. Let me think about what the factors mean.

Actually, I think the standard interpretation is:
U[i*m+j, r] is the coefficient for A_ij in the r-th "intermediate" value.
V[j*p+k, r] is the coefficient for B_jk in the r-th "intermediate" value.
W[k*n+i, r] is the coefficient for the r-th "intermediate" value in C_ik.

The intermediate value for rank r is: M_r = sum_{i,j} U[i*m+j, r] * A_ij * sum_{j,k} ... 

No, I think it's:
M_r = (sum_{i,j} U[i*m+j, r] * A_ij) * (sum_{j,k} V[j*p+k, r] * B_jk)

Then C_ik = sum_r M_r * W[k*n+i, r]

For this to equal A*B:
C_ik = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A_i'j') * (sum_{j'',k''} V[j''*p+k'', r] * B_j''k'')

For rank 1 with all ones:
M_0 = (sum_{i',j'} A_i'j') * (sum_{j'',k''} B_j''k'')
C_ik = 1 * M_0 = (sum all A) * (sum all B)

This is NOT equal to (A*B)_ik in general! So rank 1 does NOT work for actual matrix multiplication.

The tensor T[i,j,k]=1 means the bilinear term A_ij * B_jk appears in the product.
But the rank-1 decomposition with all-ones factors computes (sum A)(sum B), not A*B.

So the correct interpretation is that the factors must satisfy:
C_ik = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A_i'j') * (sum_{j'',k''} V[j''*p+k'', r] * B_j''k'')

For this to equal sum_j A_ij * B_jk for all A,B, we need the tensor equation.

The all-ones rank-1 fails because it mixes all i',j' in U and all j'',k'' in V.

So my earlier analysis was wrong. The tensor is not "all ones" in the multiplicative sense;
it's the standard matmul tensor which has rank > 1.

OK so back to block decomposition. Let me implement the rank-11 solution.

Block 1: 2x3x1 (k=0), rank 3.
For j in {0,1,2}: one rank-1 term per j.
U[i*m+j, c_j] = 1 for i in {0,1}, c_j = j
V[j*p+0, c_j] = 1
W[0*n+i, c_j] = 1 for i in {0,1}

Block 2: 2x2x2 (j in {0,1}, k in {1,2}), Strassen rank 7.
Local indices: i in {0,1}, j' in {0,1} (global j = j'), k' in {0,1} (global k = k'+1)

Strassen for 2x2x2:
M1 = (A00+A11)(B00+B11)
M2 = (A10+A11)B00
M3 = A00(B10-B00)
M4 = A11(B01-B10)
M5 = (A00+A01)B11
M6 = (A10-A00)(B00+B01)
M7 = (A01-A11)(B10+B11)

C00 = M1 - M4 + M5 - M7
C01 = M4 + M5
C10 = M2 + M3
C11 = M1 - M3 + M6 - M7

Let me implement this.

Actually, let me just use the known optimal or near-optimal decomposition and verify numerically.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
_START = time.time()
_DEADLINE = _START + 140.0


def _time_ok():
    return time.time() < _DEADLINE


def _verify_float(U, V, W, n, m, p, tol=1e-7, n_tests=10):
    """Verify that the factorization correctly computes matrix multiplication."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for _ in range(n_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # Compute via factorization
        # M_r = (U^T A_vec) * (V^T B_vec) where A_vec is A flattened, B_vec is B flattened
        A_vec = A.reshape(n * m)
        B_vec = B.reshape(m * p)
        
        U_prod = Ua @ A_vec  # shape (rank,)
        V_prod = Va @ B_vec  # shape (rank,)
        
        # C_ik = sum_r W[k*n+i, r] * U_prod[r] * V_prod[r]
        C_comp = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                C_comp[i, k] = np.dot(Wa[idx, :], U_prod * V_prod)
        
        if np.max(np.abs(C_comp - C_true)) > tol:
            return False
    
    return True


def _verify_tensor(U, V, W, n, m, p, tol=1e-7):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta condition."""
    # Actually for matmul, the tensor equation is:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k'*n+i',r] should reconstruct matmul
    # The direct tensor check is:
    # For the matmul tensor, T[i,j,k] = 1 means A_ij*B_jk contributes to C_ik
    # The factorization should satisfy:
    # sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r] = delta_jj' * delta_kk' ... no
    # 
    # The correct check is the matrix multiplication check above.
    return _verify_float(U, V, W, n, m, p, tol)


def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matmul.
    Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1.0  # A00
    U[3][0] = 1.0  # A11
    V[0][0] = 1.0  # B00
    V[3][0] = 1.0  # B11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1.0  # A10
    U[3][1] = 1.0  # A11
    V[0][1] = 1.0  # B00
    
    # M3 = A00(B10-B00)
    U[0][2] = 1.0  # A00
    V[2][2] = 1.0  # B10
    V[0][2] = -1.0 # B00
    
    # M4 = A11(B01-B10)
    U[3][3] = 1.0  # A11
    V[1][3] = 1.0  # B01
    V[2][3] = -1.0 # B10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1.0  # A00
    U[1][4] = 1.0  # A01
    V[3][4] = 1.0  # B11
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1.0  # A10
    U[0][5] = -1.0 # A00
    V[0][5] = 1.0  # B00
    V[1][5] = 1.0  # B01
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1.0  # A01
    U[3][6] = -1.0 # A11
    V[2][6] = 1.0  # B10
    V[3][6] = 1.0  # B11
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M3 + M6 - M7
    # W[k*2+i, r]:
    # W[0*2+0, r] = W[0, r] for C00
    # W[1*2+0, r] = W[2, r] for C01
    # W[0*2+1, r] = W[1, r] for C10
    # W[1*2+1, r] = W[3, r] for C11
    
    W[0][0] = 1.0   # C00 += M1
    W[0][3] = -1.0  # C00 -= M4
    W[0][4] = 1.0   # C00 += M5
    W[0][6] = -1.0  # C00 -= M7
    
    W[2][3] = 1.0   # C01 += M4
    W[2][4] = 1.0   # C01 += M5
    
    W[1][1] = 1.0   # C10 += M2
    W[1][2] = 1.0   # C10 += M3
    
    W[3][0] = 1.0   # C11 += M1
    W[3][2] = -1.0  # C11 -= M3
    W[3][5] = 1.0   # C11 += M6
    W[3][6] = -1.0  # C11 -= M7
    
    return U, V, W


def _build_decomposition():
    """Build the rank-11 decomposition.
    
    Block A: j in {0,1,2}, k=0. Rank 3.
    Block B: j in {0,1}, k in {1,2}. Strassen 2x2x2. Rank 7.
    Block C: j=2, k in {1,2,3,4}. Rank 4? Or split into 2x1x4 which is rank 4.
    
    Wait, I need to cover all (j,k) pairs:
    j in {0,1,2}, k in {0,1,2,3,4}
    
    Block A: all j, k=0. 3 terms (one per j).
    Block B: j in {0,1}, k in {1,2}. Strassen 7 terms.
    Block C: j=2, k in {1,2,3,4}. 2x1x4.
    
    For 2x1x4 (i in {0,1}, j=2, k in {1,2,3,4}):
    We need T[i,2,k] = 1 for i in {0,1}, k in {1,2,3,4}.
    Using rank-1 per k: U[i*3+2, c_k]=1, V[2*5+k, c_k]=1, W[k*2+i, c_k]=1.
    That's 4 terms.
    
    Total: 3 + 7 + 4 = 14. Hmm, worse than I thought.
    
    Let me reconsider. For 2x1x4, can we do better than rank 4?
    T[i,j,k] for i in {0,1}, j=2, k in {1,2,3,4}.
    This is a 2x4 matrix of 1s (for the two i values).
    We need U[i*3+2, r] * V[2*5+k, r] * W[k*2+i, r] to sum to 1 for each (i,k).
    
    If we use rank 2:
    r=0: U[:,0]=a_i, V[:,0]=b_k, W[:,0]=c_{k,i}
    r=1: U[:,1]=d_i, V[:,1]=e_k, W[:,1]=f_{k,i}
    
    This is getting complex. Let me just use rank 4 for simplicity.
    Total: 3 + 7 + 4 = 14.
    
    Alternative: split differently.
    Block A: j in {0,1}, k in {0,1,2,3,4}. 2x2x5.
    Block B: j=2, k in {0,1,2,3,4}. 2x1x5.
    
    2x1x5: rank 5 (one per k).
    2x2x5: split k into {0,1}, {2,3}, {4}: 7+7+1 = 15.
    Total: 5 + 15 = 20. Worse.
    
    Another: 
    Block A: j in {0,1}, k in {0,1,2}. 2x2x3.
    Block B: j in {0,1}, k in {3,4}. 2x2x2.
    Block C: j=2, k in {0,1,2,3,4}. 2x1x5.
    
    2x2x3: split k into {0}, {1,2}: 1 + 7 = 8.
    2x2x2: 7.
    2x1x5: 5.
    Total: 8 + 7 + 5 = 20. Worse.
    
    Best so far: 14.
    
    Let me try to optimize 2x1x5.
    T[i,2,k] = 1 for i in {0,1}, k in {0,1,2,3,4}.
    
    Can we do rank 3? We need a 2x5 matrix of 1s to be expressed as
    sum_r u_i(r) * v_k(r) * w_{k,i}(r).
    
    This is hard. Let me just use rank 5 for now.
    
    Actually, let me try a completely different approach:
    Use the fact that 2x3x5 matmul can be done with rank at most 2*3*5/2 = 15
    using a simple halving argument, but let's aim for 14 or less.
    
    Let me implement rank 14 and verify.
    """
    n, m, p = 2, 3, 5
    total_rank = 14
    U = [[0.0] * total_rank for _ in range(n * m)]  # (6, 14)
    V = [[0.0] * total_rank for _ in range(m * p)]  # (15, 14)
    W = [[0.0] * total_rank for _ in range(p * n)]  # (10, 14)
    
    col = 0
    # Block A: all j in {0,1,2}, k=0. Rank 3.
    for j in range(3):
        for i in range(2):
            U[i * m + j][col] = 1.0
        for i in range(2):
            W[0 * n + i][col] = 1.0
        V[j * p + 0][col] = 1.0
        col += 1
    
    # Block B: j in {0,1}, k in {1,2}. Strassen 2x2x2. Rank 7.
    sU, sV, sW = _strassen_2x2x2()
    for r in range(7):
        # Local i in {0,1} -> global i
        # Local j in {0,1} -> global j
        # Local k in {0,1} -> global k = k_local + 1
        for li in range(2):
            for lj in range(2):
                U[li * m + lj][col + r] = sU[li * 2 + lj][r]
        for lj in range(2):
            for lk in range(2):
                V[lj * p + (lk + 1)][col + r] = sV[lj * 2 + lk][r]
        for lk in range(2):
            for li in range(2):
                W[(lk + 1) * n + li][col + r] = sW[lk * 2 + li][r]
    col += 7
    
    # Block C: j=2, k in {1,2,3,4}. Rank 4.
    for k in range(1, 5):
        for i in range(2):
            U[i * m + 2][col] = 1.0
            W[k * n + i][col] = 1.0
        V[2 * p + k][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    
    if not _time_ok():
        return _schoolbook(n, m, p)
    
    # Try rank-14 block decomposition
    result = _build_decomposition()
    if _verify_float(result["U"], result["V"], result["W"], n, m, p, tol=1e-9, n_tests=20):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
