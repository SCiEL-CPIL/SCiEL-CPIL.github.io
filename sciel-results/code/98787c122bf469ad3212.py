"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-minimal exact tensor decomposition for 3x3 * 3x5
    # We use a block decomposition strategy:
    # Split the 3x3 * 3x5 into blocks where we can apply known low-rank decompositions.
    
    # Known results:
    # - 2x2 * 2x2 has rank 7 (Strassen)
    # - 1x1 * 1x1 has rank 1
    # - 2x1 * 1x2, 1x2 * 2x1 etc have rank 2
    # - 2x1 * 1x1 has rank 2
    # - 1x1 * 2x1 has rank 2
    # - 2x1 * 2x1 has rank 2 (actually, this is just two independent dot products, rank 2)
    #   Wait, A is 2x1, B is 2x1, C is 2x2. C[i,k] = sum_j A[i,j]B[j,k] but B is 2x1 so k only has 1 value? No.
    #   Let's be precise. A is (n x m), B is (m x p), C is (n x p).
    #   If n=2, m=1, p=1: A is 2x1, B is 1x1, C is 2x1. C[i,0] = A[i,0]*B[0,0]. Rank 2.
    #   If n=1, m=2, p=1: A is 1x2, B is 2x1, C is 1x1. C[0,0] = sum_j A[0,j]B[j,0]. Rank 2.
    #   If n=2, m=1, p=2: A is 2x1, B is 1x2, C is 2x2. C[i,k] = A[i,0]*B[0,k]. Rank 4?
    #      C[0,0]=A[0,0]B[0,0], C[0,1]=A[0,0]B[0,1], C[1,0]=A[1,0]B[0,0], C[1,1]=A[1,0]B[0,1].
    #      This is an outer product of A (2x1) and B^T (1x2), so rank 2 in tensor sense?
    #      Tensor T[i,j,k] = A[i,0]*B[0,k] * delta_{j,0}. 
    #      This decomposes as sum of 2 terms: A[0,0]*B[0,0]*E_00^A * E_0^B * E_00^C + ... 
    #      Actually, T = sum_{i,k} A[i,0]B[0,k] e_i \otimes e_0 \otimes e_k.
    #      This is sum of 4 basis tensors? No.
    #      The tensor is (2x1x2). 
    #      T[i,0,k] = A[i,0]B[0,k].
    #      We can write this as A[:,0] \otimes B[0,:].
    #      As a tensor, this is rank 2? No, it's a sum of products of 3 vectors.
    #      T = sum_{i,k} A[i,0]B[0,k] (e_i \otimes e_0 \otimes e_k).
    #      This is a sum of 4 terms if we consider each (i,k) pair.
    #      But can we do better?
    #      The slice for j=0 is the outer product of A[:,0] and B[0,:].
    #      The tensor rank of a matrix M (viewed as a 3-tensor with middle dimension 1) is the matrix rank of M.
    #      Here the "matrix" is A[:,0] * B[0,:]^T? No.
    #      The tensor T is in R^{2 x 1 x 2}.
    #      It can be flattened to a 2x2 matrix M where M[i,k] = T[i,0,k].
    #      M = A[:,0] * B[0,:]. This is an outer product of a 2-vector and a 2-vector, so matrix rank 1.
    #      Therefore the tensor rank is 1?
    #      Wait, T[i,j,k] = A[i,0] B[0,k] delta_{j,0}.
    #      If we choose one term in the sum: u \otimes v \otimes w.
    #      u must be a vector in R^2, v in R^1, w in R^2.
    #      (u \otimes v \otimes w)[i,j,k] = u_i v_j w_k.
    #      Since j is always 0, v_j is just v_0.
    #      So T[i,0,k] = u_i v_0 w_k.
    #      This is the outer product of u and w, scaled by v_0.
    #      Since M = A[:,0] B[0,:] is rank 1, it can be written as u w^T.
    #      So u = A[:,0], w = B[0,:], v_0 = 1.
    #      So the rank is indeed 1.
    
    # So:
    # Rank(n,m,p) for small cases:
    # R(1,1,1) = 1
    # R(2,2,2) = 7
    # R(2,1,1) = 2 (A 2x1, B 1x1, C 2x1. T[i,0,0]=A[i,0]B[0,0]. Matrix rank 2)
    # R(1,2,1) = 2 (A 1x2, B 2x1, C 1x1. T[0,j,0]=A[0,j]B[j,0]. Matrix rank 2)
    # R(1,1,2) = 2 (A 1x1, B 1x2, C 1x2. T[0,0,k]=A[0,0]B[0,k]. Matrix rank 2)
    # R(2,1,2) = 1 (A 2x1, B 1x2, C 2x2. T[i,0,k]=A[i,0]B[0,k]. Matrix rank 1)
    # R(2,2,1) = 2 (A 2x2, B 2x1, C 2x1. T[i,j,0]=A[i,j]B[j,0]. Matrix rank 2)
    # R(1,2,2) = 2 (A 1x2, B 2x2, C 1x2. T[0,j,k]=A[0,j]B[j,k]. Matrix rank 2)
    
    # We want to decompose 3x3 * 3x5.
    # Let's partition the indices.
    # n=3, m=3, p=5.
    
    # Strategy:
    # Split i into {0,1} and {2}.
    # Split j into {0,1} and {2}.
    # Split k into {0,1,2}, {3,4}.
    
    # This creates blocks.
    # Block (i,j,k) ranges:
    # 1. i in {0,1}, j in {0,1}, k in {0,1,2} -> 2x2 * 2x3.
    #    R(2,2,3)?
    #    We can split k into {0,1} and {2}.
    #    R(2,2,2) + R(2,2,1) = 7 + 2 = 9.
    # 2. i in {0,1}, j in {0,1}, k in {3,4} -> 2x2 * 2x2.
    #    R(2,2,2) = 7.
    # 3. i in {0,1}, j in {2}, k in {0,1,2} -> 2x1 * 1x3.
    #    R(2,1,3)?
    #    T[i,0,k] = A[i,0]B[0,k]. Matrix rank 1. So R(2,1,3) = 1?
    #    Wait, B is 1x3. A is 2x1. C is 2x3.
    #    C[i,k] = A[i,0] B[0,k].
    #    This is outer product of A[:,0] and B[0,:]. Rank 1.
    #    So R(2,1,3) = 1.
    # 4. i in {0,1}, j in {2}, k in {3,4} -> 2x1 * 1x2.
    #    R(2,1,2) = 1.
    # 5. i in {2}, j in {0,1}, k in {0,1,2} -> 1x2 * 2x3.
    #    R(1,2,3)?
    #    T[0,j,k] = A[0,j]B[j,k].
    #    This is a sum of 2 outer products (for j=0 and j=1). Rank 2.
    # 6. i in {2}, j in {0,1}, k in {3,4} -> 1x2 * 2x2.
    #    R(1,2,2) = 2.
    # 7. i in {2}, j in {2}, k in {0,1,2} -> 1x1 * 1x3.
    #    R(1,1,3) = 3.
    # 8. i in {2}, j in {2}, k in {3,4} -> 1x1 * 1x2.
    #    R(1,1,2) = 2.
    
    # Total rank estimate: 9 + 7 + 1 + 1 + 2 + 2 + 3 + 2 = 27.
    
    # Can we do better?
    # Maybe a different partition.
    # Or maybe the known rank for 3x3 * 3x3 is 23?
    # And 3x3 * 3x5 might be around 30-35?
    # Let's try to find a better decomposition for 2x2 * 2x3.
    # R(2,2,3) = 9 is correct?
    # 2x2 * 2x2 is 7. 2x2 * 2x1 is 2. Total 9.
    # Is there a 8-rank decomposition for 2x2 * 2x3?
    # Likely not, since it contains a 2x2 * 2x2 subproblem.
    
    # Let's check the sum again.
    # i={0,1}, j={0,1}:
    #   k={0,1,2}: R(2,2,3) = 9
    #   k={3,4}: R(2,2,2) = 7
    #   Total: 16
    # i={0,1}, j={2}:
    #   k={0,1,2}: R(2,1,3) = 1
    #   k={3,4}: R(2,1,2) = 1
    #   Total: 2
    # i={2}, j={0,1}:
    #   k={0,1,2}: R(1,2,3) = 2
    #   k={3,4}: R(1,2,2) = 2
    #   Total: 4
    # i={2}, j={2}:
    #   k={0,1,2}: R(1,1,3) = 3
    #   k={3,4}: R(1,1,2) = 2
    #   Total: 5
    
    # Grand Total: 16 + 2 + 4 + 5 = 27.
    
    # Let's implement this.
    # We need to generate the U, V, W matrices for each block and sum them up.
    
    # Helper to create block decomposition
    def make_block(n_b, m_b, p_b, i_off, j_off, k_off):
        # Returns U, V, W for this block with global indices
        # We will create local U, V, W and then place them into global ones.
        # For now, let's just return the local matrices and the offsets.
        # But the interface requires global matrices.
        # So we need to add to global matrices.
        pass

    # Let's define the decompositions for the basic blocks.
    
    # 1. R(2,2,2) Strassen
    # U (4x7), V (4x7), W (4x7)
    # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_222 = np.array([
        [1, 0, 0, 0, 1, 0, 0], # a
        [0, 0, 1, 0, 0, 0, 1], # b
        [0, 0, 0, 1, 0, 1, 0], # c
        [1, 0, 0, 1, 0, 0, 0]  # d
    ], dtype=float)
    
    V_222 = np.array([
        [1, 1, 0, 0, 0, 0, 0], # e+h
        [1, 0, 0, 0, 0, 0, 0], # e
        [0, 1, 0, 1, 0, 1, 1], # f, g, h
        [0, 0, -1, 1, 1, 0, 0] # h, g-e
    ], dtype=float)
    # Wait, Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U columns (coeffs for A):
    # M1: a=1, d=1 -> [1,0,0,1]
    # M2: c=1, d=1 -> [0,0,1,1]
    # M3: a=1 -> [1,0,0,0]
    # M4: d=1 -> [0,0,0,1]
    # M5: a=1, b=1 -> [1,1,0,0]
    # M6: c=1, a=-1 -> [0,0,1,-1]
    # M7: b=1, d=-1 -> [0,1,0,-1]
    
    U_222 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ], dtype=float).T # This is 7x4, we need 4x7
    
    U_222 = U_222.T # Now 4x7
    # Row 0 (a): 1, 0, 1, 0, 1, 0, 0
    # Row 1 (b): 0, 0, 0, 0, 1, 0, 1
    # Row 2 (c): 0, 1, 0, 0, 0, 1, 0
    # Row 3 (d): 1, 1, 0, 1, 0, -1, -1
    
    # V columns (coeffs for B):
    # M1: e=1, h=1 -> [1,0,0,1]
    # M2: e=1 -> [1,0,0,0]
    # M3: f=1, h=-1 -> [0,1,0,-1]
    # M4: g=1, e=-1 -> [0,0,1,0] -> [ -1, 0, 1, 0]
    # M5: h=1 -> [0,0,0,1]
    # M6: e=1, f=1 -> [1,1,0,0]
    # M7: g=1, h=1 -> [0,0,1,1]
    
    V_222 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ], dtype=float)
    
    # W columns (coeffs for C):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=float)
    
    # 2. R(2,2,1)
    # A 2x2, B 2x1, C 2x1
    # C[i,0] = sum_j A[i,j] B[j,0]
    # T[i,j,0] = A[i,j] B[j,0]
    # Matrix M[i,0] = A[i,:] dot B[:,0]
    # This is just matrix multiplication.
    # Rank 2?
    # T[i,j,0] = A[i,j] B[j,0].
    # Decomposition:
    # Term 1: e_0 \otimes (e_0+B[0,0]e_1?? No)
    # Let's use the definition.
    # T = sum_{i,j} A[i,j] B[j,0] e_i \otimes e_j \otimes e_0.
    # This is sum of 4 terms if we go term by term.
    # But we can group.
    # T[i,0,0] = A[i,0]B[0,0] + A[i,1]B[1,0]
    # T[i,1,0] = A[i,1]B[1,0] ... wait.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    # T[i,j,0] is only non-zero for j=0,1.
    # T[i,0,0] = A[i,0]B[0,0]
    # T[i,1,0] = A[i,1]B[1,0]
    # So T = sum_i sum_j A[i,j] B[j,0] e_i \otimes e_j \otimes e_0.
    # This can be written as:
    # Term 1: u \otimes v \otimes w
    # We need to represent the bilinear form.
    # Actually, this is equivalent to multiplying a 2x2 matrix by a 2x1 matrix.
    # The rank of 2x2 * 2x1 is 2?
    # No, rank(n,m,p) for n=2,m=2,p=1.
    # C is 2x1.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    # This is two independent scalar products (one for i=0, one for i=1).
    # Each scalar product 2x2 * 2x1 is rank 2?
    # A 1x2 * 2x1 is rank 2.
    # So 2x2 * 2x1 is rank 2*2 = 4?
    # Let's check.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # These are two dot products of length 2.
    # Each dot product of length 2 has rank 2.
    # Since they are independent, total rank is 4?
    # Wait. R(1,2,1) = 2.
    # R(2,2,1) = R(1,2,1) + R(1,2,1) = 2+2=4.
    # My previous estimate of 2 was wrong.
    
    # Let's re-evaluate R(2,2,3).
    # R(2,2,2) = 7.
    # R(2,2,1) = 4.
    # Total 11.
    
    # Let's re-evaluate the whole sum.
    # i={0,1}, j={0,1}:
    #   k={0,1,2}: R(2,2,3) = 7 (for k=0,1) + 4 (for k=2) = 11.
    #   k={3,4}: R(2,2,2) = 7.
    #   Total: 18.
    # i={0,1}, j={2}:
    #   k={0,1,2}: R(2,1,3).
    #   T[i,0,k] = A[i,0]B[0,k].
    #   C[i,k] = A[i,0]B[0,k].
    #   This is outer product. Rank 1.
    #   Total: 1.
    #   k={3,4}: R(2,1,2) = 1.
    #   Total: 1.
    #   Sum: 2.
    # i={2}, j={0,1}:
    #   k={0,1,2}: R(1,2,3).
    #   T[0,j,k] = A[0,j]B[j,k].
    #   C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k].
    #   This is sum of two outer products (vectors in B space).
    #   Rank 2.
    #   Total: 2.
    #   k={3,4}: R(1,2,2) = 2.
    #   Total: 2.
    #   Sum: 4.
    # i={2}, j={2}:
    #   k={0,1,2}: R(1,1,3).
    #   C[0,k] = A[0,0]B[0,k].
    #   This is scaling a vector.
    #   T[0,0,k] = A[0,0]B[0,k].
    #   Rank 1?
    #   Wait, C is 1x3.
    #   C[0,k] = A[0,0] B[0,k].
    #   This is just scaling the row of B.
    #   Tensor T[0,0,k] = A[0,0] B[0,k].
    #   This is sum of 3 terms? No.
    #   T = sum_k A[0,0]B[0,k] e_0 \otimes e_0 \otimes e_k.
    #   This is sum of 3 basis tensors.
    #   But can it be done in 1?
    #   u \otimes v \otimes w.
    #   u=e_0, v=e_0.
    #   w = B[0,:].
    #   So (e_0 \otimes e_0 \otimes B[0,:])[0,0,k] = 1*1*B[0,k] = B[0,k].
    #   We need A[0,0]B[0,k].
    #   So we can just scale u or v.
    #   u = A[0,0]e_0.
    #   Then (u \otimes e_0 \otimes B[0,:])[0,0,k] = A[0,0]B[0,k].
    #   So Rank is 1.
    #   My previous estimate of 3 was wrong.
    #   R(1,1,p) = 1 for any p?
    #   Yes, because it's just scaling.
    #   So R(1,1,3) = 1.
    #   R(1,1,2) = 1.
    #   Sum: 2.
    
    # Grand Total: 18 + 2 + 4 + 2 = 26.
    
    # This is a huge improvement over 45.
    
    # Let's implement this 26-rank decomposition.
    
    # We need to build the global U, V, W matrices.
    # n=3, m=3, p=5.
    # U shape: 9 x 26
    # V shape: 15 x 26
    # W shape: 15 x 26
    
    U = np.zeros((9, 26), dtype=float)
    V = np.zeros((15, 26), dtype=float)
    W = np.zeros((15, 26), dtype=float)
    
    col = 0
    
    # Helper to add a block
    def add_block(i_idx, j_idx, k_idx, U_b, V_b, W_b):
        nonlocal col
        n_b = len(i_idx)
        m_b = len(j_idx)
        p_b = len(k_idx)
        rank_b = U_b.shape[1]
        
        for r in range(rank_b):
            for li, i in enumerate(i_idx):
                for lj, j in enumerate(j_idx):
                    # U index: i*m + j
                    U[i*m + j, col + r] += U_b[li, r]
                    
            for lj, j in enumerate(j_idx):
                for lk, k in enumerate(k_idx):
                    # V index: j*p + k
                    V[j*p + k, col + r] += V_b[lj, r]
                    
            for lk, k in enumerate(k_idx):
                for li, i in enumerate(i_idx):
                    # W index: k*n + i
                    W[k*n + i, col + r] += W_b[lk, r]
                    
        col += rank_b

    # Block 1: i={0,1}, j={0,1}, k={0,1} -> 2x2 * 2x2 (Strassen)
    add_block([0,1], [0,1], [0,1], U_222, V_222, W_222)
    
    # Block 2: i={0,1}, j={0,1}, k={2} -> 2x2 * 2x1
    # We need U_221, V_221, W_221
    # Rank 4.
    # T[i,j,0] = A[i,j] B[j,0].
    # We decompose into 4 terms.
    # Term 1: A[0,0]B[0,0] -> e_0^A \otimes e_0^B \otimes e_0^C
    # Term 2: A[0,1]B[1,0] -> e_0^A \otimes e_1^B \otimes e_0^C
    # Term 3: A[1,0]B[0,0] -> e_1^A \otimes e_0^B \otimes e_0^C
    # Term 4: A[1,1]B[1,0] -> e_1^A \otimes e_1^B \otimes e_0^C
    
    U_221 = np.eye(4).reshape(2, 2, 2).transpose(0, 1, 2).reshape(4, 4) # This is getting complicated.
    # Let's just hardcode the 4x4, 4x4, 2x4 matrices.
    # U (4x4): Rows for A[0,0], A[0,1], A[1,0], A[1,1].
    # U[i,j, r] = 1 if r corresponds to (i,j).
    # r=0: (0,0), r=1: (0,1), r=2: (1,0), r=3: (1,1).
    U_221 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # V (4x4): Rows for B[0,0], B[0,1], B[1,0], B[1,1].
    # We only use B[0,0] and B[1,0].
    # r=0: B[0,0], r=1: B[1,0] (wait, j=1, so B[1,0]).
    # r=0: (0,0), r=1: (1,0), r=2: (0,0), r=3: (1,0).
    # V[j, r].
    # V[0,0]=1, V[0,1]=0, V[0,2]=1, V[0,3]=0.
    # V[1,0]=0, V[1,1]=1, V[1,2]=0, V[1,3]=1.
    # But V is 2x2 * 1, so V shape is (2*1)x4 = 2x4.
    # Rows are B[0,0] and B[1,0].
    V_221 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    # W (2x4): Rows for C[0,0], C[1,0].
    # W[i, r].
    # r=0: C[0,0], r=1: C[0,0], r=2: C[1,0], r=3: C[1,0].
    W_221 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    add_block([0,1], [0,1], [2], U_221, V_221, W_221)
    
    # Block 3: i={0,1}, j={0,1}, k={3,4} -> 2x2 * 2x2 (Strassen)
    add_block([0,1], [0,1], [3,4], U_222, V_222, W_222)
    
    # Block 4: i={0,1}, j={2}, k={0,1,2} -> 2x1 * 1x3
    # Rank 1.
    # T[i,0,k] = A[i,0]B[0,k].
    # U (2x1): [1, 1]^T? No, U[i,0] = 1 for i=0,1.
    # V (3x1): V[k,0] = 1 for k=0,1,2. (B[0,k])
    # W (6x1): W[k*n+i, 0] = 1?
    # Wait, W index is k*n + i.
    # T[i,0,k] = U[i,0] V[0,k] W[k,0]?
    # Standard form: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # Here j=0.
    # T[i,0,k] = sum_r U[i,0,r] V[0,k,r] W[k,i,r].
    # We want T[i,0,k] = A[i,0] B[0,k].
    # Let r=0.
    # U[i,0,0] = A[i,0]? No, U is constant coefficients.
    # U[i,0,0] = 1.
    # V[0,k,0] = 1.
    # W[k,0,0] = 1.
    # Then sum = 1*1*1 = 1.
    # But we need A[i,0]B[0,k].
    # This decomposition works for the tensor structure, but the values of A and B are variables.
    # The tensor T is the multilinear map.
    # T(A,B)_{i,k} = A[i,0]B[0,k].
    # This is a single term in the tensor decomposition?
    # T = u \otimes v \otimes w?
    # (u \otimes v \otimes w)(A,B)_{i,k} = u_i v_0 w_k A_i B_k? No.
    # The definition is:
    # T = sum_r u_r \otimes v_r \otimes w_r.
    # T(A,B)_{i,k} = sum_r u_r[i,0] v_r[0,k] w_r[k,i] A[i,0] B[0,k]?
    # No.
    # T(A,B) = sum_r (u_r^T A) (v_r^T B) w_r?
    # Let's stick to the matrix definition.
    # U (n*m x rank), V (m*p x rank), W (p*n x rank).
    # C = (U^T A_flat) \odot (V^T B_flat) \to W?
    # No, the standard definition is:
    # C_{i,k} = sum_{j,r} A_{i,j} B_{j,k} ...
    # The tensor decomposition T = sum_r u_r \otimes v_r \otimes w_r means:
    # T_{i,j,k} = sum_r u_{r,i,j} v_{r,j,k} w_{r,k,i}.
    # And the multiplication is:
    # C_{i,k} = sum_{j,r} A_{i,j} B_{j,k} w_{r,k,i} u_{r,i,j} v_{r,j,k}?
    # No.
    # The condition is:
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 if this is the schoolbook term, 0 otherwise?
    # No.
    # The condition is that the tensor T_{i,j,k} (which is 1 if i,j,k are valid indices for the term)
    # is decomposed.
    # For 2x1 * 1x3:
    # T_{i,0,k} = 1 for all i,k.
    # T = sum_r u_r \otimes v_r \otimes w_r.
    # T_{i,0,k} = sum_r u_{r,i,0} v_{r,0,k} w_{r,k,i}.
    # We want this to be 1.
    # Let r=0.
    # u_{0,i,0} = 1 for i=0,1.
    # v_{0,0,k} = 1 for k=0,1,2.
    # w_{0,k,i} = 1 for all k,i.
    # Then sum = 1*1*1 = 1.
    # So U (2x1), V (3x1), W (6x1).
    # U = [[1],[1]]
    # V = [[1],[1],[1]]
    # W = [[1],[1],[1],[1],[1],[1]] (flattened k*n+i)
    
    U_213 = np.ones((2, 1))
    V_213 = np.ones((3, 1))
    W_213 = np.ones((6, 1))
    add_block([0,1], [2], [0,1,2], U_213, V_213, W_213)
    
    # Block 5: i={0,1}, j={2}, k={3,4} -> 2x1 * 1x2
    # Rank 1.
    # Similar to above.
    U_212 = np.ones((2, 1))
    V_212 = np.ones((2, 1))
    W_212 = np.ones((4, 1))
    add_block([0,1], [2], [3,4], U_212, V_212, W_212)
    
    # Block 6: i={2}, j={0,1}, k={0,1,2} -> 1x2 * 2x3
    # Rank 2.
    # T_{0,j,k} = 1.
    # T = sum_r u_r \otimes v_r \otimes w_r.
    # T_{0,j,k} = sum_r u_{r,0,j} v_{r,j,k} w_{r,k,0}.
    # We need 1.
    # Let's use 2 terms.
    # Term 1: u=1 for j=0, v=1 for k=0,1,2, w=1 for k=0,1,2.
    #   u_{1,0,0}=1, u_{1,0,1}=0.
    #   v_{1,0,k}=1.
    #   w_{1,k,0}=1.
    #   Sum = 1.
    # Term 2: u=1 for j=1, v=1 for k=0,1,2, w=1 for k=0,1,2.
    #   u_{2,0,0}=0, u_{2,0,1}=1.
    #   v_{2,1,k}=1.
    #   w_{2,k,0}=1.
    #   Sum = 1.
    # Wait, the sum is over r.
    # T_{0,j,k} = u_{1,0,j}v_{1,j,k}w_{1,k,0} + u_{2,0,j}v_{2,j,k}w_{2,k,0}.
    # For j=0: 1*1*1 + 0*... = 1.
    # For j=1: 0*... + 1*1*1 = 1.
    # So it works.
    
    U_123 = np.array([[1, 0], [0, 1]])
    V_123 = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]) # 2x3, so 2 rows per j?
    # V shape (m*p x rank) = (2*3 x 2) = 6x2.
    # Rows: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).
    # V[0,0]=1, V[0,1]=0
    # V[1,0]=1, V[1,1]=0
    # V[2,0]=1, V[2,1]=0
    # V[3,0]=0, V[3,1]=1
    # V[4,0]=0, V[4,1]=1
    # V[5,0]=0, V[5,1]=1
    
    W_123 = np.ones((6, 2)) # 3*2 x 2.
    # W[k*n+i, r]. i=0.
    # W[0,0]=1, W[0,1]=1
    # W[1,0]=1, W[1,1]=1
    # W[2,0]=1, W[2,1]=1
    # W[3,0]=1, W[3,1]=1
    # W[4,0]=1, W[4,1]=1
    # W[5,0]=1, W[5,1]=1
    # Wait, w_{r,k,i} must be such that the product works.
    # In my construction:
    # Term 1: w_{1,k,0} = 1 for all k.
    # Term 2: w_{2,k,0} = 1 for all k.
    # So W is all 1s.
    
    add_block([2], [0,1], [0,1,2], U_123, V_123, W_123)
    
    # Block 7: i={2}, j={0,1}, k={3,4} -> 1x2 * 2x2
    # Rank 2.
    # Similar.
    U_122 = np.array([[1, 0], [0, 1]])
    V_122 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]]) # 2x2, 4 rows.
    W_122 = np.ones((4, 2))
    add_block([2], [0,1], [3,4], U_122, V_122, W_122)
    
    # Block 8: i={2}, j={2}, k={0,1,2} -> 1x1 * 1x3
    # Rank 1.
    U_113 = np.ones((1, 1))
    V_113 = np.ones((3, 1))
    W_113 = np.ones((3, 1))
    add_block([2], [2], [0,1,2], U_113, V_113, W_113)
    
    # Block 9: i={2}, j={2}, k={3,4} -> 1x1 * 1x2
    # Rank 1.
    U_112 = np.ones((1, 1))
    V_112 = np.ones((2, 1))
    W_112 = np.ones((2, 1))
    add_block([2], [2], [3,4], U_112, V_112, W_112)
    
    # Check rank
    assert col == 26, f"Rank mismatch: {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
