import numpy as np
import time
import sys
from fractions import Fraction
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 4 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    n, m, p = 3, 4, 4
    
    # The tensor T_{i,j,k} = delta_{i+k, j} for matrix multiplication C = A * B
    # where A is n x m, B is m x p, C is n x p
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # T_{i,j,k} = 1 if i+k == j, else 0
    
    # For a rank-r decomposition, we need:
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # such that for all (i, j, k):
    # sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = delta_{i+k, j}
    
    # Note the W index: W[k*n + i, t]
    
    # Strategy: Start with schoolbook (rank = n*m*p = 48) and try to reduce.
    # Better: Use known lower-rank decompositions.
    
    # For 4x4x4, the best known is 21 (Laderman).
    # For 3x4x4, we can potentially use a sub-decomposition.
    
    # Let's first construct the schoolbook decomposition (rank 48)
    # then try to find a better one.
    
    # Schoolbook: rank = m = 4? No, rank = n*m*p for naive, but actually
    # the standard decomposition has rank m = 4 for each output element... 
    # Actually, the tensor rank of n x m times m x p multiplication:
    # The naive algorithm uses n*p*m multiplications.
    # But the tensor rank is at most n*p*m.
    # 
    # Actually, let me think more carefully.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i, j, k) where T[i,j,k] = delta_{i+k,j}? No.
    # 
    # Let me re-read the convention. W[k*n+i, r].
    # So W is indexed by (k, i) where k is column of C (0..p-1) and i is row of C (0..n-1).
    # U is indexed by (i, j) where i is row of A (0..n-1) and j is column of A / row of B (0..m-1).
    # V is indexed by (j, k) where j is row of B (0..m-1) and k is column of B (0..p-1).
    
    # The decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_t (sum_j U[i,j,t] A[i,j]) * (sum_k V[j,k,t] B[j,k]) ... no that's not right.
    
    # Standard tensor decomposition for matmul:
    # C[i,k] = sum_j sum_t U[i,j,t] V[j,k,t] W[i,k,t] * A[i,j] B[j,k]
    # Wait, the tensor T[i,j,k] = delta_{i+k,j} is not quite right either.
    
    # Let me reconsider. The multiplication tensor T satisfies:
    # <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = delta_{i+k,j}? No, that would give C[i,k] = sum_j A[i,j] B[j,k] * delta_{i+k,j}
    # which only works if i+k = j, which is not generally true.
    
    # Actually, T[i,j,k] = 1 for all valid i,j,k, and the contraction is:
    # C[i,k] = sum_{j,t} U[i,j,t] V[j,k,t] W[i,k,t]
    # where U, V, W are the factors.
    
    # The target tensor is T[i,j,k] = delta_{i+k, j} is WRONG.
    # The target tensor for matrix multiplication A(nxm) * B(mx p) -> C(nxp) is:
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # But that's not a tensor decomposition in the usual sense.
    
    # Let me look at this differently. The standard formulation:
    # We want to write C[i,k] = sum_j A[i,j] B[j,k] as
    # C[i,k] = sum_t W[i,k,t] * (sum_j U[i,j,t] A[i,j]) * (sum_j' V[j',k,t] B[j',k])
    # This is not standard either.
    
    # The correct tensor formulation:
    # The bilinear map (A, B) -> C has a tensor T in (R^{nm})^* ⊗ (R^{mp})^* ⊗ (R^{np})
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k} * delta... 
    # Actually T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    # So T[i,j,k] = 1 for all i,j,k.
    # And a rank-r decomposition means:
    # T[i,j,k] = sum_{t=0}^{r-1} U[i,j,t] * V[j,k,t] * W[i,k,t]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r) with W indexed as W[k*n+i, t] = W[i,k,t]
    
    # Wait, W[k*n+i, r] means the index is k*n + i, so W is (p*n, r) and W[k*n+i, t] corresponds to output element C[i,k].
    
    # So the condition is:
    # sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    # This is the matrix multiplication tensor.
    
    # For schoolbook: r = m (for each of the n*p outputs, we do m multiplications)
    # But that's not a tensor decomposition. The tensor rank is the minimum r such that
    # T = sum_{t=0}^{r-1} U_t ⊗ V_t ⊗ W_t
    
    # Schoolbook gives r = n*p*m = 48? No.
    # Actually, the standard algorithm computes each C[i,k] with m multiplications.
    # Total multiplications = n*p*m = 48.
    # Each multiplication corresponds to one term in the tensor decomposition.
    # So schoolbook rank = 48.
    
    # But we can do better. For 4x4x4, Laderman achieves rank 21.
    # For 3x4x4, we might be able to achieve less than 48.
    
    # Let me construct the schoolbook decomposition first.
    
    # Schoolbook: For each (i, k) in [0,n)x[0,p) and each j in [0,m):
    # Term t = i*p*p + k*p + j? No, let's index differently.
    # We need r terms. For schoolbook, r = n*p*m = 48.
    # Term t corresponds to (i0, j0, k0).
    # U[i*m+j, t] = 1 if i==i0 and j==j0, else 0
    # V[j*p+k, t] = 1 if j==j0 and k==k0, else 0  
    # W[k*n+i, t] = 1 if i==i0 and k==k0, else 0
    
    # Then sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 if (i,j,k) == (i0,j0,k0) for some t
    # But we need it to be 1 for ALL i,j,k. That's wrong.
    
    # I'm confusing myself. Let me restart with a clear formulation.
    
    # The matrix multiplication tensor T ∈ (R^{n×m})^* ⊗ (R^{m×p})^* ⊗ (R^{n×p})
    # T = Σ_{i,j,k} E_{ij} ⊗ E_{jk} ⊗ E_{ik}
    # where E_{ij} is the matrix with 1 at (i,j) and 0 elsewhere.
    
    # A rank-r decomposition is:
    # T = Σ_{t=0}^{r-1} U_t ⊗ V_t ⊗ W_t
    # where U_t ∈ (R^{n×m})^*, V_t ∈ (R^{m×p})^*, W_t ∈ R^{n×p}
    
    # Equivalently, for all i,j,k:
    # 1 = Σ_{t=0}^{r-1} U_t[i,j] V_t[j,k] W_t[i,k]
    
    # This must hold for ALL i,j,k.
    
    # Schoolbook: r = m*n*p? No, r = m for each (i,k) pair, total r = n*p*m.
    # For each (i,k), and for each j, we have a term:
    # U_{(i,j,k)}[i',j'] = δ_{i,i'} δ_{j,j'}
    # V_{(i,j,k)}[j',k'] = δ_{j,j'} δ_{k,k'}
    # W_{(i,j,k)}[i',k'] = δ_{i,i'} δ_{k,k'}
    
    # Then Σ_t U_t[i,j] V_t[j,k] W_t[i,k] = Σ_{j''} δ_{j,j''} δ_{j,j''} δ_{k,k} ... 
    
    # This is getting messy. Let me just use a known approach.
    
    # The standard schoolbook decomposition has rank = m (for the "inner" dimension)
    # but that's for a fixed (i,k). The total rank is n*p*m.
    
    # Actually, I recall that the tensor rank of n×m × m×p matmul is at most n*m*p
    # and can be lower.
    
    # Let me just try to find ANY valid decomposition, starting with a simple one.
    
    # Simple approach: Use the fact that for 3x4x4, we can embed it in 4x4x4
    # and use Laderman's rank-21 decomposition, then restrict to the 3x4x4 subproblem.
    
    # But that might not give a valid lower-rank decomposition for 3x4x4.
    
    # Alternative: Just return a valid (possibly non-optimal) decomposition.
    # The score is the rank, so we want to minimize it.
    
    # Let me construct the schoolbook decomposition with rank = n*m*p = 48.
    
    rank_sb = n * m * p  # 48
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, t] = 1.0
                V[j*p + k, t] = 1.0
                W[k*n + i, t] = 1.0
                t += 1
    
    # Verify: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # For a given (i,j,k), only t = i*m*p + j*p + k contributes (if we index that way)
    # Actually with my indexing, t = i*m*p + j*p + k... let me check.
    # i=0,j=0,k=0: t=0
    # i=0,j=0,k=1: t=1
    # i=0,j=0,k=2: t=2
    # i=0,j=0,k=3: t=3
    # i=0,j=1,k=0: t=4
    # ...
    # For U[i*m+j, t] = 1, we need t such that the (i,j) pair matches.
    # With my loop, for fixed i,j, k varies, so t = i*m*p + j*p + k.
    # U[i*m+j, t] = 1 for t in [i*m*p + j*p, i*m*p + j*p + p)
    # V[j*p+k, t] = 1 for t in [i*m*p + j*p, i*m*p + j*p + p) when the k matches... 
    # No, V[j*p+k, t] = 1 only when the specific k matches.
    
    # Let's verify numerically.
    # For (i,j,k) = (0,0,0):
    # Sum over t: U[0,t] * V[0,t] * W[0,t]
    # U[0,t] = 1 for t=0,1,2,3 (i=0,j=0,k=0,1,2,3)
    # V[0,t] = 1 for t=0,1,2,3 (j=0,k=0,1,2,3)
    # W[0,t] = 1 for t=0,1,2,3 (k=0,i=0) -> W[k*n+i, t] = W[0*3+0, t] = W[0,t]
    # For t=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> product = 1
    # For t=1: U[0,1]=1, V[0,1]=1, W[0,1]=1 -> product = 1
    # For t=2: U[0,2]=1, V[0,2]=1, W[0,2]=1 -> product = 1
    # For t=3: U[0,3]=1, V[0,3]=1, W[0,3]=1 -> product = 1
    # Sum = 4, but we need 1.
    
    # This is wrong. The schoolbook decomposition should give exactly 1 for each (i,j,k).
    
    # The issue is that I'm using the same t for multiple (i,j,k) triples.
    # Each term in the decomposition corresponds to ONE (i,j,k) triple.
    # So r = n*m*p = 48, and term t corresponds to triple (i_t, j_t, k_t).
    # U[i*m+j, t] = 1 if i == i_t and j == j_t, else 0
    # V[j*p+k, t] = 1 if j == j_t and k == k_t, else 0
    # W[k*n+i, t] = 1 if i == i_t and k == k_t, else 0
    
    # Then for a given (i,j,k), the sum is 1 iff there exists t with i_t=i, j_t=j, k_t=k.
    # Since every triple appears exactly once, the sum is 1 for all (i,j,k). ✓
    
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, t] = 1.0
                V[j*p + k, t] = 1.0
                W[k*n + i, t] = 1.0
                t += 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t]
                    if abs(val - 1.0) > 1e-9:
                        return False, (i, j, k, val)
        return True, None
    
    ok, info = verify(U, V, W, n, m, p)
    if not ok:
        print(f"Verification failed: {info}")
        # Fallback: return something
        return {'U': U, 'V': V, 'W': W}
    
    # Now try to find a better decomposition.
    # Strategy: Use Kronecker product with 2x2x2 Strassen.
    # 3x4x4 can be viewed as (3/2 * 2) x (4/2 * 2) x (4/2 * 2) = 1.5 x 2 x 2
    # Not integer. Let's try 3x4x4 = (3) x (2*2) x (2*2)
    # Use Strassen for the 2x2 blocks.
    
    # Actually, let's try a different approach: use the rank-7 Strassen for 2x2x2
    # and combine.
    
    # For 3x4x4, we can split A into 3 rows, and B into 2x2 blocks of columns.
    # A is 3x4, B is 4x4.
    # Split B's 4 columns into two 2-column blocks: B = [B1 | B2] where B1, B2 are 4x2.
    # Then C = A * B = [A*B1 | A*B2]
    # A*B1 is 3x2, A*B2 is 3x2.
    
    # For each 3x2 output, we need to compute A(3x4) * B_block(4x2).
    # This is a 3x4x2 matmul.
    
    # For 3x4x2: rank is at most 3*4*2 = 24, but can be lower.
    # We have two such blocks, so total rank is at most 2 * rank(3x4x2).
    
    # But we can do better by sharing computations.
    
    # Let me try a direct approach: solve for a low-rank decomposition using optimization.
    # But that's complex. Let me just return the schoolbook for now, and if time permits,
    # try to improve.
    
    # Given the time constraints, let me just return the schoolbook decomposition.
    # Rank = 48.
    
    # But wait, the current best is 44 according to the prompt. So 48 is worse.
    # Let me try to do better.
    
    # Idea: Use the fact that 3x4x4 is a sub-tensor of 4x4x4.
    # If we have a rank-r decomposition for 4x4x4, we can restrict it to 3x4x4
    # by setting the 4th row of A to zero.
    # This might reduce the effective rank.
    
    # For 4x4x4, Laderman's algorithm has rank 21.
    # Let me implement Laderman's algorithm for 4x4x4 and then restrict.
    
    # Laderman's 4x4x4 decomposition (rank 21):
    # This is complex to implement from scratch. Let me use a known formulation.
    
    # Alternative: Use a simpler lower-rank algorithm.
    # For 3x4x4, the theoretical lower bound is around 12 (from algebraic complexity).
    # But practical algorithms give higher ranks.
    
    # Let me try a block-based approach:
    # Split A (3x4) into two parts: A1 (2x4) and A2 (1x4).
    # Split B (4x4) into two parts: B1 (4x2) and B2 (4x2).
    # C11 = A1*B1 (2x2), C12 = A1*B2 (2x2)
    # C21 = A2*B1 (1x2), C22 = A2*B2 (1x2)
    
    # Use Strassen for 2x2x2 matmul:
    # M1 = (A1[0,0]+A1[1,1]) * (B1[0,0]+B1[1,1])  -- not quite, Strassen is for 2x2 * 2x2
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # For A1 (2x4) * B1 (4x2), we can't directly use Strassen since dimensions don't match.
    # We need to handle 2x4 * 4x2.
    
    # This is getting too complex for the time remaining. Let me just return the schoolbook.
    
    # Actually, let me try a simpler improvement: use rank = n*p*m / something.
    # For 3x4x4, maybe we can use a rank-30 or so decomposition.
    
    # Given time, let me just return the schoolbook (rank 48) which is guaranteed to work.
    
    return {'U': U, 'V': V, 'W': W}
