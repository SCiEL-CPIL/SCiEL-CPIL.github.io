"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Use a block decomposition.
    - For the first 3 output columns (3x3x3 part), use the known rank-19 Coppersmith-Winograd decomposition.
    - For the remaining 4 output columns, use a shared structure with the A-factors.
    
    The total rank will be 19 + 4*3*3 = 55 initially, but we can potentially share more.
    
    Actually, let me use a cleaner approach:
    - Split B into B1 (3x3) and B2 (3x4)
    - C1 = A @ B1 uses rank-19
    - C2 = A @ B2: for each column of B2, we need 3x3 mult, but we can share A-factors
    
    For C2 = A @ B2 where B2 is 3x4:
    Each output column C2[:,k] = A @ B2[:,k] is a 3x3 x 3x1 mult with rank 9.
    But we can share the 9 A-side terms across all 4 columns.
    So total for C2: 9 (shared A) + 4*3 (B-side) + 4*3 (C-side) ... no, that's not how tensor rank works.
    
    Let me just use the schoolbook for the 4 extra columns: 4*9 = 36 terms.
    Total: 19 + 36 = 55.
    
    But wait, I can do better. For the 4 extra columns, I can use a 3x3 x 3x4 decomposition.
    rank(3,3,4) <= 3*4 + 3*3 + 3*4 - 3 = 12+9+12-3 = 30? No, that's not a valid bound.
    
    Actually, rank(3,3,4) <= 3*rank(3,3,1) = 3*9 = 27? No.
    
    Let me just use a simple construction and verify it works.
    """
    start_time = time.time()
    
    n, m, p = 3, 3, 7
    
    # Use schoolbook decomposition as baseline (rank 63)
    # Then try to improve
    
    # Schoolbook: rank = n*m*p = 63
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify the decomposition
    # For random A, B, check that sum_r U[:,r] * A_flat * V[:,r] * B_flat * W[:,r] == C_flat
    # Actually, the condition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # Wait, that's not right. Let me re-derive.
    
    # The correct condition: 
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give the "multiplication" pattern
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-1 term: s = (U_term . A) * (V_term . B), then C += s * W_term
    # So: C[i,k] = sum_r (sum_{i',j'} U[i',j',r] A[i',j']) * (sum_{j',k'} V[j',k',r] B[j',k']) * W[i,k,r]
    
    # For this to equal sum_j A[i,j] B[j,k] for all A,B:
    # sum_r U[i1,j1,r] V[j2,k2,r] W[i3,k3,r] = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3}
    
    # Let me verify with the schoolbook:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for col = i*m*p + j*p + k
    # sum_r U[i1*m+j1, r] V[j2*p+k2, r] W[k3*n+i3, r]
    # = 1 if there exists r such that (i1,j1,k) = (i1',j1',k') and (j2,k2) = (j1',k') and (k3,i3) = (k',i1')
    # = delta_{i1,i3} delta_{j1,j2} delta_{k2,k3} ✓
    
    # Good, schoolbook works. Now let me try to reduce rank.
    
    # Use Coppersmith-Winograd for 3x3x3 (rank 19) + 36 for remaining 4 columns = 55
    # But let me first check if I can find a better decomposition.
    
    # For now, return schoolbook (rank 63) as a safe fallback
    # TODO: implement rank-19 CW decomposition for 3x3x3
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
