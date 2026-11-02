"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import itertools
import random
from fractions import Fraction
import numpy as np

def _verify(U, V, W):
    """Verify that the tensor decomposition is exact."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # The multiplication tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(len(U[0])):
                    val = val + U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Faster verification using numpy."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # We can check this by constructing the tensor and comparing
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Let's verify a few random samples plus all if small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :]) # This isn't right
                # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # = (U[i*m+j,:] * V[j*p+k,:]) dot W[k*n+i,:]
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 x 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    where i,j in {0,1} for 2x2, k in {0,1} for 2x2.
    
    Strassen's 7 products for C = A*B where A,B are 2x2:
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b10-b11)
    M4 = a11(b01-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    
    C = [[M1-M4+M5+M6, M4+M5], [M2+M3, M1-M3+M2+M7]]
    
    So C[0,0] = M1 - M4 + M5 + M6
       C[0,1] = M4 + M5
       C[1,0] = M2 + M3
       C[1,1] = M1 - M3 + M2 + M7
    """
    # U[i*2+j, r] where i=row of A, j=col of A
    # V[j*2+k, r] where j=col of A (row of B), k=col of B
    # W[k*2+i, r] where k=col of B, i=row of C (which equals row of A in convention W[k*n+i])
    # Wait, the convention is W[k*n+i, r] where n=2, so W[k*2+i, r]
    # i is the row index of the output (and A), k is the col index of B (and output col)
    
    # Let me re-derive carefully.
    # A is 2x2: A[i,j], i,j in {0,1}
    # B is 2x2: B[j,k], j,k in {0,1}
    # C is 2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # U[i*m+j, r] = U[i*2+j, r]
    # V[j*p+k, r] = V[j*2+k, r]
    # W[k*n+i, r] = W[k*2+i, r]
    #
    # C[i,k] = sum_j sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    #
    # For each Strassen product r, we need:
    # U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] contributes to C[i,k]
    
    # Let's define the 7 factors. For factor r, we have:
    # U_r[i,j] (2x2), V_r[j,k] (2x2), W_r[k,i] (2x2)
    # such that sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta(j) ... no.
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # We want: sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for the "tensor" T[i,j,k] = 1
    # Wait no. The tensor for matrix multiplication 2x2 x 2x2 is T[i,j,k] = 1 for all i,j,k? No.
    # T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k. Yes, it's the all-ones tensor.
    
    # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    #   U1[i,j]: U1[0,0]=1, U1[1,1]=1, rest 0
    #   V1[j,k]: V1[0,0]=1, V1[1,1]=1, rest 0
    #   W1[k,i]: W1[0,0]=1, W1[1,1]=1, rest 0
    #   Contribution to C[i,k]: sum_j U1[i,j]V1[j,k]W1[k,i]
    #   For i=0,k=0: j=0: U1[0,0]V1[0,0]W1[0,0]=1*1*1=1; j=1: U1[0,1]V1[1,0]W1[0,0]=0
    #     So C[0,0] gets 1. But we need the total sum over all r to give 1 for each (i,j,k) in the tensor sense.
    #   Wait, I'm confusing myself. Let me just verify: does sum_r U_r[i,j]V_r[j,k]W_r[k,i] = 1 for all i,j,k?
    
    # Let me just hardcode the known Strassen factors and verify.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    # U1: rows [0,0,0,1] for i*2+j: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   U1[0,0]=1, U1[3,0]=1 (i=1,j=1)
    # V1: V1[0,0]=1, V1[3,0]=1 (j=1,k=1)
    # W1: W1[0,0]=1, W1[3,0]=1 (k=1,i=1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    # U2: U2[2,1]=1 (i=1,j=0), U2[3,1]=1 (i=1,j=1)
    # V2: V2[0,1]=1 (j=0,k=0)
    # W2: W2[0,1]=1 (k=0,i=0)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1
    
    # M3 = a00(b10-b11)
    # U3: U3[0,2]=1 (i=0,j=0)
    # V3: V3[2,2]=1 (j=1,k=0), V3[3,2]=-1 (j=1,k=1)
    # W3: W3[0,2]=1 (k=0,i=0)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1
    
    # M4 = a11(b01-b00)
    # U4: U4[3,3]=1 (i=1,j=1)
    # V4: V4[1,3]=1 (j=0,k=1), V4[0,3]=-1 (j=0,k=0)
    # W4: W4[1,3]=1 (k=1,i=1)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[1][3] = 1
    
    # M5 = (a00+a01)b11
    # U5: U5[0,4]=1 (i=0,j=0), U5[1,4]=1 (i=0,j=1)
    # V5: V5[3,4]=1 (j=1,k=1)
    # W5: W5[3,4]=1 (k=1,i=1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[3][4] = 1
    
    # M6 = (a10-a00)(b00+b10)
    # U6: U6[2,5]=1 (i=1,j=0), U6[0,5]=-1 (i=0,j=0)
    # V6: V6[0,5]=1 (j=0,k=0), V6[2,5]=1 (j=1,k=0)
    # W6: W6[0,5]=1 (k=0,i=0)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = 1
    
    # M7 = (a01-a11)(b01+b11)
    # U7: U7[1,6]=1 (i=0,j=1), U7[3,6]=-1 (i=1,j=1)
    # V7: V7[1,6]=1 (j=0,k=1), V7[3,6]=1 (j=1,k=1)
    # W7: W7[1,6]=1 (k=1,i=1)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[1][6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Start with Strassen-based decomposition: rank = 7 * p = 7 * 16 = 112
    # This is worse than schoolbook 64. So Strassen per-block is not helpful here.
    # 
    # Schoolbook is rank 64. We need to find something better.
    # 
    # The key insight: for 2x2 x 2x16, we can potentially use a lower-rank decomposition
    # by exploiting structure across the 16 columns.
    #
    # Let me think about this differently. The tensor is T[i,j,k] = 1 for all i,j,k
    # with i in {0,1}, j in {0,1}, k in {0,...,15}.
    #
    # We want min rank such that T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r[i*2+j], V_r[j*2+k], W_r[k*2+i]
    #
    # Since T is all-ones, this is equivalent to finding a low-rank decomposition
    # of the "all-ones" tensor of shape (4, 32, 32) under the specific index mappings.
    
    # Actually, let me reconsider. The standard approach for rectangular matrix multiplication
    # 2x2 times 2x16:
    # C = A * B where A is 2x2, B is 2x16, C is 2x16.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can view B as 16 independent 2x1 columns. For each column k, we multiply 2x2 by 2x1.
    # The 2x2 by 2x1 multiplication has tensor of shape (2,2,1) for each k, all identical.
    # So the full tensor is 16 copies of the 2x2x1 tensor.
    #
    # The 2x2x1 tensor T[i,j,k=0] = 1 for all i,j. This has rank 2 (since it's rank-1 in the
    # (i,j) plane replicated... actually T[i,j,0]=1 for all i,j, so it's the outer product of
    # all-ones vectors in i and j, times 1 in k. So rank 1? No, because the indices are coupled.
    #
    # Actually for 2x2 x 2x1: C[i,0] = sum_j A[i,j] B[j,0]
    # T[i,j,0] = 1. The tensor is shape (2,2,1). It equals sum over r of u_r[i] v_r[j] w_r[0].
    # Since w_r[0] is a scalar, this is just a rank-r decomposition of the 2x2 all-ones matrix.
    # The all-ones 2x2 matrix has rank 1! So T for 2x2x2x1 has tensor rank 1.
    #
    # Wait, that can't be right for the full problem. Let me re-examine.
    #
    # For the full 2x2 x 2x16 problem:
    # C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0,...,15}
    #
    # The tensor T[i,j,k] = 1 for all i,j,k. Shape is (2, 2, 16) in terms of (i,j,k).
    # But the factor matrices have shapes:
    #   U: (n*m, rank) = (4, rank) with U[i*m+j, r] = U[i*2+j, r]
    #   V: (m*p, rank) = (32, rank) with V[j*p+k, r] = V[j*16+k, r]
    #   W: (p*n, rank) = (32, rank) with W[k*n+i, r] = W[k*2+i, r]
    #
    # So T[i,j,k] = sum_r U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r]
    #
    # This is a 4x32x32 tensor (in the flattened index sense) that must equal 1 everywhere.
    #
    # Key observation: T[i,j,k] = 1 does not depend on i, j, or k separately in a simple way.
    # It's constant. So we're looking for a rank-r decomposition of the constant-1 tensor
    # under these specific index mappings.
    
    # Let me try a different approach. Since T is all-ones, we can write:
    # 1 = sum_r U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r]
    #
    # One simple decomposition: use r = n*m*p = 64 terms, one for each (i,j,k).
    # That's schoolbook.
    #
    # Can we do better? The tensor is "all ones" which has very low algebraic rank in some sense.
    #
    # Let me try: can we separate the k-dependence?
    # V[j*16+k, r] and W[k*2+i, r] both depend on k.
    #
    # Idea: Use a decomposition where for each "type" of (i,j), we handle all k together.
    # There are 4 types of (i,j). For each type, we need:
    #   sum_r U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r] = 1 for all k.
    #
    # If we could make U[i*2+j, r] depend only on (i,j) and the (V,W) part handle k...
    #
    # Let's try: suppose we use r = 4*s terms, where for each of the 4 (i,j) pairs, we use s terms.
    # For a fixed (i,j), define u_r = U[i*2+j, r] for r in the block for (i,j).
    # Then we need: sum_{r in block} u_r * V[j*16+k, r] * W[k*2+i, r] = 1 for all k.
    #
    # This is like a matrix equation. Let's define for fixed (i,j):
    #   M_{k,r} = V[j*16+k, r] * W[k*2+i, r]
    # Then we need u^T * M = 1^T (row vector of ones of length 16).
    #
    # The minimum s for this is at most 16 (one per k), but could be less if the rows of M are linearly dependent.
    #
    # This is getting complex. Let me just try to find a good decomposition by optimization.
    
    # Actually, let me try a simpler structured approach.
    # 
    # For 2x2 x 2x16, note that:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # We can write this as:
    # C[:,k] = A * B[:,k]
    #
    # One approach: decompose A in a low-rank way. But A is 2x2, rank at most 2.
    #
    # Let me try: use the fact that we can share computation across k.
    #
    # Define for each k:
    #   P0_k = B[0,k], P1_k = B[1,k]
    #   C[0,k] = A[0,0]*P0_k + A[0,1]*P1_k
    #   C[1,k] = A[1,0]*P0_k + A[1,1]*P1_k
    #
    # If we precompute combinations of A entries:
    #   s0 = A[0,0], s1 = A[0,1], s2 = A[1,0], s3 = A[1,1]
    #
    # Then C[0,k] = s0*P0_k + s1*P1_k, C[1,k] = s2*P0_k + s3*P1_k
    #
    # This is just 4 multiplications per k, total 64. No savings.
    #
    # What if we use Strassen-like tricks across the k dimension?
    #
    # Let's try: group k into pairs. For k=2t, 2t+1:
    #   C[0,2t] = s0*P0_{2t} + s1*P1_{2t}
    #   C[0,2t+1] = s0*P0_{2t+1} + s1*P1_{2t+1}
    #
    # Define: P0_sum_t = P0_{2t} + P0_{2t+1}, P0_diff_t = P0_{2t} - P0_{2t+1}
    #         P1_sum_t = P1_{2t} + P1_{2t+1}, P1_diff_t = P1_{2t} - P1_{2t+1}
    #
    # Then: C[0,2t] + C[0,2t+1] = s0*P0_sum_t + s1*P1_sum_t
    #         C[0,2t] - C[0,2t+1] = s0*P0_diff_t + s1*P1_diff_t
    #
    # This doesn't save multiplications directly.
    
    # Let me try a numerical optimization approach to find a low-rank decomposition.
    # Start with rank 48 and try to find a valid decomposition.
    
    def try_rank(rank):
        """Try to find a rank-r decomposition via iterative optimization."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        random.seed(42)
        np.random.seed(42)
        
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Use gradient descent to minimize ||T - sum_r U_r V_r W_r||^2
        # where T[i,j,k] = 1
        
        lr = 0.01
        for iteration in range(2000):
            if time.time() > deadline:
                return None
            
            # Compute current tensor estimate
            # T_est[i,j,k] = sum_r U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r]
            # 
            # Gradient w.r.t. U[i*2+j, r]:
            #   dL/dU[i*2+j, r] = sum_{k} (T_est[i,j,k] - 1) * V[j*16+k, r] * W[k*2+i, r]
            # 
            # This is expensive to compute naively. Let's use a simplified approach.
            
            # Actually, let's use a different strategy: alternating least squares or
            # a simpler heuristic.
            
            # For now, let's just verify if the initial random guess is close.
            # This won't work well. Let me try a structured initialization.
            
            break
        
        return None
    
    # Let me try a more structured approach.
    # 
    # Key insight: The tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_{a=0}^{1} sum_{b=0}^{1} delta(i,a) * delta(j,b) * 1
    # This is rank 4 in a trivial sense, but the index mappings make it more complex.
    #
    # Actually, let's think about it as:
    # T[i,j,k] = 1 = (1) * (1) * (1)
    # So if we could have U[i*2+j, 0] = 1, V[j*16+k, 0] = 1, W[k*2+i, 0] = 1,
    # then T = 1. But U[i*2+j, 0] = 1 for all i,j means U is all-ones in column 0.
    # V[j*16+k, 0] = 1 for all j,k means V is all-ones in column 0.
    # W[k*2+i, 0] = 1 for all k,i means W is all-ones in column 0.
    # Then T[i,j,k] = 1*1*1 = 1. 
    #
    # Wait, that gives rank 1! But that can't be right because the factor matrices
    # have specific shapes and the decomposition must be exact.
    #
    # Let me verify: if U[i*2+j, 0] = 1 for all i in {0,1}, j in {0,1},
    # V[j*16+k, 0] = 1 for all j in {0,1}, k in {0,...,15},
    # W[k*2+i, 0] = 1 for all k in {0,...,15}, i in {0,1},
    # then T[i,j,k] = U[i*2+j,0] * V[j*16+k,0] * W[k*2+i,0] = 1*1*1 = 1.
    #
    # This seems too good to be true. Let me re-read the problem statement.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix 
    # by a 2 by 16 matrix."
    #
    # The tensor for matrix multiplication is NOT all-ones in the way I've been thinking.
    # The tensor T has entries T[a,b,c] where a indexes A, b indexes B, c indexes C,
    # and T[a,b,c] = 1 if c = a*b in the matrix multiplication sense, else 0.
    #
    # For matrix multiplication (n x m) * (m x p) = (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T[i,j,k] = 1 for all valid i,j,k. There's no "if" condition; it's always 1
    # because every (i,j,k) triple contributes to the product.
    #
    # So yes, T[i,j,k] = 1 for all i,j,k. And my rank-1 argument seems valid!
    #
    # But wait, that would mean ALL matrix multiplications have tensor rank 1, which is absurd.
    # The issue is that the factor matrices U, V, W have FIXED shapes and the indices are
    # coupled in a specific way.
    #
    # Let me re-examine. The factor matrices are:
    #   U: shape (n*m, rank) = (4, rank), entry U[i*m+j, r]
    #   V: shape (m*p, rank) = (32, rank), entry V[j*p+k, r]  
    #   W: shape (p*n, rank) = (32, rank), entry W[k*n+i, r]
    #
    # And the condition is:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # For n=2, m=2, p=16:
    #   T[i,j,k] = sum_r U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r] = 1
    #
    # If I set U[i*2+j, 0] = 1, V[j*16+k, 0] = 1, W[k*2+i, 0] = 1,
    # then T[i,j,k] = 1 for all i,j,k. This IS a valid rank-1 decomposition!
    #
    # But this contradicts the known result that matrix multiplication has tensor rank > 1.
    # The resolution: in the standard tensor decomposition of matrix multiplication,
    # the tensor is T[i,j,k] where i indexes rows of A, j indexes cols of A / rows of B,
    # k indexes cols of B / cols of C. The factor matrices are:
    #   U: (n*m, rank) with U[i*m+j, r]
    #   V: (m*p, rank) with V[j*p+k, r]
    #   W: (n*p, rank) with W[i*p+k, r]
    #
    # Note: W is (n*p, rank) with W[i*p+k, r], NOT W[k*n+i, r]!
    #
    # The problem statement says: "The output factor follows the official trace convention W[k*n+i,r]."
    # So W[k*n+i, r] where k indexes cols of B and i indexes rows of A/C.
    # This is the same as W[i*p+k, r] if we swap the order... no, k*n+i vs i*p+k are different.
    #
    # For n=2, p=16: k*n+i = k*2+i, while i*p+k = i*16+k. These are different indexings!
    #
    # With W[k*n+i, r] = W[k*2+i, r], the index is (k,i) with k first.
    # With the standard W[i*p+k, r] = W[i*16+k, r], the index is (i,k) with i first.
    #
    # The tensor condition is the same either way; it's just a different ordering of the W matrix.
    # So my rank-1 argument still holds!
    #
    # I think the issue is that I'm misunderstanding the problem. Let me re-read.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 16 matrix."
    #
    # I think the tensor is NOT all-ones. The tensor for matrix multiplication M_{n,m,p} is defined
    # as: T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j] and B[j,k], which is always true.
    # So T is indeed all-ones.
    #
    # But then the tensor rank would be 1, which is clearly wrong. The resolution must be that
    # the factor matrices have additional constraints or the tensor is defined differently.
    #
    # AH, I think I see the issue. In the standard tensor decomposition, the tensor is:
    #   T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (for the "multiplication" tensor)
    # where the sum is over all i,j,k. This is the all-ones tensor.
    #
    # The tensor rank of the all-ones tensor of shape (n*m, m*p, n*p) is... well, it depends.
    # For n=m=p=2, the all-ones tensor of shape (4,4,4) has tensor rank 4 (I think).
    #
    # Wait, no. The tensor rank of the all-ones tensor 1_{a} ⊗ 1_{b} ⊗ 1_{c} is 1!
    # Because it's the outer product of three all-ones vectors.
    #
    # So the tensor rank of the all-ones tensor is always 1. This means any matrix multiplication
    # would have tensor rank 1, which is absurd.
    #
    # I must be fundamentally misunderstanding the problem. Let me look at this from a different angle.
    #
    # In the context of matrix multiplication algorithms, the "tensor" is not the all-ones tensor.
    # The matrix multiplication tensor is:
    #   M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k  ... no.
    #
    # Actually, the matrix multiplication tensor is defined as:
    #   M_{n,m,p} = sum_{i,j,k} T_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where T_{i,j,k} = 1 if k = i*j (in some sense), but for matrix multiplication,
    # every triple (i,j,k) contributes, so T_{i,j,k} = 1 for all i,j,k.
    #
    # I'm going in circles. Let me just accept that the schoolbook decomposition has rank n*m*p
    # and try to find a better one empirically.
    
    # Given the confusion, let me just return the schoolbook decomposition for now,
    # which is known to be valid.
    
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
# EVOLVE-BLOCK-END
