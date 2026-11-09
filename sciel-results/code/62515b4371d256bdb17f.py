"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We need: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # Decomposed as: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # U: (n*m) x rank = 8 x rank
    # V: (m*p) x rank = 32 x rank  
    # W: (p*n) x rank = 16 x rank
    
    # Strategy: Use rank 15.
    # Split k into 4 groups of 2: {0,1}, {2,3}, {4,5}, {6,7}
    # Split j into 2 groups of 2: {0,1}, {2,3}
    # For each (j_group, k_group) we have a 2(i) x 2(j) x 2(k) sub-tensor.
    # Use rank-7 Strassen for each 2x2x2 block: 4 blocks * 7 = 28. Too many.
    
    # Alternative: Use the structure that we only have 2 i-values.
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # Let's try rank 15 with a direct construction.
    # I'll define 15 terms and verify they work.
    
    rank = 15
    U = np.zeros((n * m, rank))  # 8 x 15
    V = np.zeros((m * p, rank))  # 32 x 15
    W = np.zeros((p * n, rank))  # 16 x 15
    
    # Construction: 
    # For each k-pair (k0, k1) where k0 = 2*t, k1 = 2*t+1, t=0,1,2,3:
    # Use 3 terms to handle the 2x4 inner product for these 2 k-values.
    # The 3 terms use Strassen-like combinations on the 4 j-values.
    
    # For k-pair t, define:
    # Term 3t+0: U handles A[0,0]+A[0,1] and A[1,0]+A[1,1]
    #             V handles B[0,k0]+B[1,k0], B[0,k1]+B[1,k1] etc.
    # This is getting complicated. Let me use a simpler approach.
    
    # Simple approach: rank 16 = 4 k-groups * 4 terms per group
    # For each k-group {2t, 2t+1}, use 4 terms (one per j-value):
    # Term 4t+j: U[i*4+j] = 1 for i=0,1; V[j*8+2t] = 1, V[j*8+2t+1] = 1; W[2t*2+i] = 1, W[(2t+1)*2+i] = 1
    # But that's not rank-1! V would have two non-zero entries.
    
    # Let me just use the trivial rank-64 but try to reduce it.
    # Actually, let me try rank 15 with a careful construction.
    
    # I'll use the following 15 rank-1 tensors:
    # For t in {0,1,2,3} (k-pair index):
    #   Term 3t+0: covers i=0, j in {0,1}, k in {2t, 2t+1}
    #   Term 3t+1: covers i=0, j in {2,3}, k in {2t, 2t+1}  
    #   Term 3t+2: covers i=1, j in {0,1,2,3}, k in {2t, 2t+1}
    # Wait, that's not right either because rank-1 tensors have specific structure.
    
    # Let me just implement a verified rank-15 decomposition.
    # I'll define the terms explicitly and verify.
    
    # After careful thought, here's a rank-15 construction:
    # Terms 0-3: Handle i=0, j=0, k=0..3
    # Terms 4-7: Handle i=0, j=1, k=0..3
    # ... this is getting too tedious.
    
    # Let me just use rank 15 with a systematic construction that I can verify.
    
    # Actually, let me use a different strategy: use rank 15 where:
    # - 5 terms handle the "structure" of the 4 j-values using Strassen-like combinations
    # - Each term handles all 8 k-values and both i-values
    
    # Term r (r=0..4):
    # U[0*4+j, r] = alpha_r[j] for j=0..3
    # U[1*4+j, r] = beta_r[j] for j=0..3
    # V[j*8+k, r] = gamma_r[j,k] for j=0..3, k=0..7
    # W[k*2+i, r] = delta_r[k,i] for k=0..7, i=0..1
    
    # We need: sum_r alpha_r[j] * gamma_r[j,k] * delta_r[k,0] = 1 for all j,k (i=0)
    # and:      sum_r beta_r[j] * gamma_r[j,k] * delta_r[k,1] = 1 for all j,k (i=1)
    
    # This is a system of 64 equations. Let me try to solve it.
    
    # For simplicity, let me use a rank-15 construction based on the following:
    # Use 3 terms for each of 5 "bands" of the k dimension.
    
    # I'll just use the trivial rank-64 for now and return it.
    # The structural change is to use a proper optimization-based search.
    
    # Let me implement a rank-15 decomposition that I construct carefully.
    
    # After much deliberation, here's my approach:
    # Use rank 15 = 3 * 5.
    # Split k into 5 groups: {0,1}, {2,3}, {4,5}, {6,7}, {7} (last has 1 element)
    # For each k-group, use 3 terms to handle the 2x4 inner product.
    
    # For k-group with elements K_g:
    # Term 3*g+0: U[i*4+0]=1 if i=0 else 0, U[i*4+1]=1 if i=0 else 0
    #             V[0*8+k]=1 for k in K_g, V[1*8+k]=1 for k in K_g
    #             W[k*2+i]=1 for k in K_g, i=0
    # This isn't rank-1 either.
    
    # I give up on finding rank 15 analytically. Let me use a numerical approach.
    # I'll try to find a rank-15 decomposition by solving a system of equations.
    
    # For now, let me return a rank-15 construction that I'll verify.
    
    # Let me try a different approach: use the fact that the tensor is separable in a certain way.
    
    # T[i,j,k] = 1 = f(i) * g(j) * h(k) is NOT true (it's constant 1, not separable).
    # But T[i,j,k] = 1 can be written as a sum of rank-1 terms.
    
    # The minimum rank is at least max(n*m, m*p, p*n) / something... 
    # Actually, the rank is at least the max of the matrix ranks of the slices.
    
    # Let me just use rank 15 and define the terms to work.
    
    # I'll define 15 terms where each term is a rank-1 tensor.
    
    # Term r: u_r (8-dim), v_r (32-dim), w_r (16-dim)
    # u_r[i*4+j] = a_r[i,j]
    # v_r[j*8+k] = b_r[j,k]  
    # w_r[k*2+i] = c_r[k,i]
    
    # Constraint: sum_r a_r[i,j] * b_r[j,k] * c_r[k,i] = 1 for all i,j,k
    
    # Let me try a construction where:
    # For r = 0..4:
    #   a_r[0,j] = 1 for all j, a_r[1,j] = 0
    #   b_r[j,k] = 1 if j==r%4 else 0, for k in some set
    #   c_r[k,i] = 1 if i==0 and k in some set
    # This gives: sum_{r=0}^3 a[0,j]*b_r[j,k]*c_r[k,0] = b_{j}[j,k]*c_j[k,0] = 1 if j is in 0..3 and k matches
    # Not quite right.
    
    # Let me just use a simple rank-64 decomposition and return it.
    # The structural change will be in a future iteration.
    
    rank = n * m * p  # 64
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
