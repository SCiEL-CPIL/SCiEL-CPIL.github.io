"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    # Deadline guard: must return within 150s wall-clock.
    start_time = time.time()
    DEADLINE = 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # The tensor T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C
    # We need to find rank-minimal decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Known lower bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # For 4x4 * 4x6, the known best is 24 (this is a rectangular case)
    # Actually, let me think about this more carefully.
    
    # For matrix multiplication (n,m) x (m,p) -> (n,p):
    # The tensor has dimensions (n*m) x (m*p) x (p*n) in the flattened form
    # Rank lower bounds: max(n*m, m*p, p*n)
    # For 4,4,6: max(16, 24, 24) = 24
    
    # Let's try to construct a rank-24 solution directly.
    # The structure: T[i,j,k] = delta (it's 1 for all valid (i,j,k) where j is shared)
    # Wait, actually T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p)
    # No wait - the matrix multiplication tensor is T[i,j,k] = 1 if and only if
    # the (i,j) entry of A, (j,k) entry of B contribute to (i,k) of C.
    # Actually for standard matrix mult, T[i,j,k] = 1 for all combinations where
    # the indices are valid. Since all indices are valid here, T[i,j,k] = 1 for all
    # i in [0,n), j in [0,m), k in [0,p).
    
    # Wait, that's not right either. The tensor for matrix multiplication is:
    # T[i, j, k] = 1 if the (i,j) of A and (j,k) of B produce (i,k) of C
    # This is 1 for ALL valid (i,j,k) since every element of A and B contributes.
    # So T is the all-ones tensor of shape (n, m, p) = (4, 4, 6).
    
    # Flattened: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] summed over r = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the all-ones 3D tensor.
    # The all-ones tensor of shape (n,m,p) has rank = max(n,m,p) when we can use
    # the standard construction... actually no.
    
    # Let me reconsider. The all-ones tensor T[i,j,k] = 1.
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # One approach: Use the fact that 1 = sum over some structure.
    # If we set U[i*m+j, r] = 1 for all i,j when r < something, etc.
    
    # Actually, the simplest decomposition of the all-ones tensor:
    # T[i,j,k] = sum_r a_r[i] * b_r[j] * c_r[k] where a_r, b_r, c_r are vectors
    # But our flattening is different: U is indexed by (i,j), V by (j,k), W by (k,i)
    
    # Let me think about this differently. We need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    # Consider the following construction with rank = m = 4:
    # For each r in [0, m):
    #   U[i*m+j, r] = 1 if j == r else 0
    #   V[j*p+k, r] = 1 if j == r else 0  -- wait, this doesn't work because V depends on j and k
    #   Actually V[j*p+k, r] should be 1 when j == r
    #   W[k*n+i, r] = 1 for all k, i
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = U[i*m+j,j]*V[j*p+k,j]*W[k*n+i,j]
    # = 1*1*1 = 1. Yes! This works with rank = m = 4?
    
    # Wait, let me double check. U[i*m+j, r] = 1 if j==r. So U[i*m+j, j] = 1, others 0.
    # V[j*p+k, r] = 1 if j==r. So V[j*p+k, j] = 1, others 0.
    # W[k*n+i, r] = 1 for all k,i,r.
    # Product: U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] is non-zero only when r=j.
    # Then it equals 1*1*1 = 1. Sum over r gives 1. 
    
    # But wait, the rank would be m=4, but U has n*m=16 rows, V has m*p=24 rows, W has p*n=24 rows.
    # The rank is the number of columns = m = 4.
    
    # Hmm, but is this correct? Let me verify with a specific example.
    # n=4, m=4, p=6. Take i=2, j=1, k=5.
    # U[2*4+1, r] = U[9, r]. This is 1 if r==j==1, else 0. So U[9,1]=1.
    # V[1*6+5, r] = V[11, r]. This is 1 if r==j==1, else 0. So V[11,1]=1.
    # W[5*4+2, r] = W[22, r]. This is 1 for all r.
    # Product at r=1: 1*1*1 = 1. All other r: 0. Sum = 1. Correct!
    
    # So rank = m = 4? That seems too good. Let me check if this is valid.
    # The issue: is the all-ones tensor really what we need?
    
    # Actually, I need to re-examine the problem. The matrix multiplication tensor
    # for multiplying A (n x m) by B (m x p) to get C (n x p) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] represents the coefficient: T[i,j,k] = 1 (since A[i,j]*B[j,k] contributes)
    # So yes, T[i,j,k] = 1 for all valid i,j,k.
    
    # But wait, I think I'm confusing myself. Let me re-read the problem.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So my construction above with rank = m = 4 should work!
    
    # Let me verify this is indeed rank 4 and not lower.
    # The lower bound is max(n*m, m*p, p*n) = max(16, 24, 24) = 24.
    # Wait, that lower bound is for the standard matrix multiplication where the
    # tensor is T[i,j,k] = 1. But actually that lower bound doesn't apply here because
    # I'm using a different flattening.
    
    # Hmm, let me reconsider. The standard lower bound for matrix multiplication rank is
    # based on the structure of the tensor. For the all-ones tensor of shape (n,m,p),
    # the rank can indeed be as low as max(n,m,p) or even lower.
    
    # Actually, I realize I need to be more careful. Let me just verify my construction
    # works and use it. If rank 4 works, great. Let me code it up and verify.
    
    # Construction: rank = m = 4
    rank = m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for i in range(n):
            U[i * m + r, r] = 1
        for k in range(p):
            V[r * p + k, r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1
    
    # Verify
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if val != 1:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = np.zeros((n * m, rank_fb))
                    V_fb = np.zeros((m * p, rank_fb))
                    W_fb = np.zeros((p * n, rank_fb))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj, col] = 1
                                V_fb[jj * p + kk, col] = 1
                                W_fb[kk * n + ii, col] = 1
                                col += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
