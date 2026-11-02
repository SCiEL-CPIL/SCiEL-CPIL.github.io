"""
Rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication.
Uses Strassen's algorithm on 2x2x2 blocks with a (2+1) split of the inner dimension,
reducing rank from schoolbook 54 to 50.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _strassen_2x2_decomposition():
    """
    Returns the 7 rank-1 terms for 2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-vectors (flattened 2x2).
    The convention: C = sum_r u_r * (v_r * w_r) in outer product sense.
    
    For Strassen:
    M1 = (A11 - A22)(B12 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B11 - B22)
    M4 = A22(B22 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Flatten 2x2 matrices in row-major: [00, 01, 10, 11]
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    
    terms = []
    
    # M1 = (A11 - A22)(B12 + B22)
    # A11=a00, A22=a11, B12=b01, B22=b11
    # u for M1: contributes to C11 (+1), C22 (+1)
    # C11 = C[0,0], C22 = C[1,1]
    # In flattened C: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
    # u1: [1, 0, 0, 1] (C11 + C22)
    # v1 (A part): a00 - a11 -> [1, 0, 0, -1]
    # w1 (B part): b01 + b11 -> [0, 1, 0, 1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 0, 1]))
    
    # M2 = (A21 + A22)B11
    # A21=a10, A22=a11, B11=b00
    # C21 = M2 -> C[1,0] = idx2
    # u2: [0, 0, 1, 0]
    # v2: [0, 0, 1, 1]
    # w2: [1, 0, 0, 0]
    terms.append(([0, 0, 1, 0], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    # M3 = A11(B11 - B22)
    # A11=a00, B11=b00, B22=b11
    # C12 = M3, C22 = M3 -> C[0,1]=idx1, C[1,1]=idx3
    # u3: [0, 1, 0, 1]
    # v3: [1, 0, 0, 0]
    # w3: [1, 0, 0, -1]
    terms.append(([0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, -1]))
    
    # M4 = A22(B22 - B11)
    # A22=a11, B22=b11, B11=b00
    # C11 = M4, C21 = M4 -> C[0,0]=idx0, C[1,0]=idx2
    # u4: [1, 0, 1, 0]
    # v4: [0, 0, 0, 1]
    # w4: [-1, 0, 0, 1]
    terms.append(([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 1]))
    
    # M5 = (A11 + A12)B22
    # A11=a00, A12=a01, B22=b11
    # C11 = -M5, C12 = M5 -> C[0,0]=-1, C[0,1]=1
    # u5: [-1, 1, 0, 0]
    # v5: [1, 1, 0, 0]
    # w5: [0, 0, 0, 1]
    terms.append(([-1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21=a10, A11=a00, B11=b00, B12=b01
    # C22 = M6 -> C[1,1]=idx3
    # u6: [0, 0, 0, 1]
    # v6: [-1, 0, 1, 0]
    # w6: [1, 1, 0, 0]
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]))
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12=a01, A22=a11, B21=b10, B22=b11
    # C11 = M7 -> C[0,0]=idx0
    # u7: [1, 0, 0, 0]
    # v7: [0, 1, 0, -1]
    # w7: [0, 0, 1, 1]
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]))
    
    return terms

def construct():
    """
    Construct rank-50 decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Split the inner dimension (size 3) into 2+1.
    - For the 2x2 part: use Strassen's rank-7 algorithm.
      A is 2x2 (rows 0,1; inner cols 0,1), B is 2x9 (inner cols 0,1; all 9 output cols).
      This gives 7 * 9 = 63... wait, that's not right.
      
    Actually, let me reconsider. The tensor is T[i,j,k] = delta_{i,j,k} for A(2x3)*B(3x9)->C(2x9).
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into {0,1} and {2}:
    C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
    
    For the first part (j in {0,1}), we have A_sub(2x2) * B_sub(2x9).
    This is NOT a square matrix multiplication, so Strassen doesn't directly apply in the standard way.
    
    Let me think differently. The standard approach for rectangular:
    
    Actually, the knowledge says "Rank-50 decomposition using Strassen's algorithm on 2x2x2 blocks for the (2+1) split of the inner dimension, reducing from schoolbook rank 54 to 50."
    
    Schoolbook: 2*3*9 = 54.
    
    The (2+1) split: 
    - Part 1: A(2x2) * B(2x9) -> contributes to C(2x9). This is a 2x2x9 tensor.
    - Part 2: A(2x1) * B(1x9) -> contributes to C(2x9). This is a 2x1x9 tensor (rank 18).
    
    Part 2 is rank 18 (schoolbook).
    For Part 1: 2x2x9 tensor. We can think of B as 9 columns, each being a 2x1 vector.
    The 2x2x1 tensor (multiplying 2x2 by 2x1) has rank... 
    
    Actually, let's think of it as: for each output column k (0..8), we compute C[:,k] = A_sub * B_sub[:,k] where A_sub is 2x2 and B_sub[:,k] is 2x1.
    
    The tensor for 2x2 * 2x1 -> 2x1 has dimension 2*2*1 = 4, but it's really a 2x2 matrix times a 2-vector.
    Rank of 2x2x1 matrix multiplication tensor: this is the same as matrix-vector multiplication 2x2.
    
    Hmm, let me reconsider. The known result is rank 50. Let me just construct it.
    
    For the 2+1 split:
    - Terms for j=2 (the "1" part): 2*1*9 = 18 terms (schoolbook for 2x1x9)
    - Terms for j in {0,1} (the "2" part): need to compute 2x2 * 2x9 -> 2x9
    
    For 2x2 * 2x9: Think of B as having 9 columns. For each column, we do 2x2 * 2x1.
    The 2x2 * 2x1 multiplication can be done with rank... 
    
    Actually, I think the approach is:
    - Use Strassen-like decomposition for the 2x2 block multiplication.
    - The 2x2 * 2x2 -> 2x2 has rank 7 (Strassen).
    - For 2x2 * 2x9, we can view it as 2x2 * (2x2 + 2x2 + 2x2 + 2x2 + 2x1) but that's messy.
    
    Alternative: View 2x2 * 2x9 as 9 independent 2x2 * 2x1 problems? No, that's not right either because B shares the same 2x2 structure per column.
    
    Let me just use the direct construction. For 2x2 * 2x9:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1 and k=0..8.
    
    This is a 2x2x9 tensor. Its rank: we can use the fact that this is equivalent to 9 copies of 2x2x1, but they share A.
    
    Actually, I'll use a different approach. Let me just do:
    - For the 2x2 part (j=0,1): use a rank-7 decomposition per "2x2 block" of B, but B has 9 columns.
    
    I think the cleanest way: 
    - Part 2 (j=2): 18 terms, schoolbook.
    - Part 1 (j=0,1): The tensor T1[i,j,k] for i in {0,1}, j in {0,1}, k in {0..8}.
      This can be decomposed by viewing B_sub as a 2x9 matrix. 
      For each k, C[:,k] = A_sub @ B_sub[:,k].
      
      We can use: for 2x2 * 2x1, the rank is 4 (schoolbook) but can we do better?
      Actually 2x2 * 2x1 is just matrix-vector, which has rank 4 (it's just 4 multiplications).
      
      But we have 9 columns sharing the same A. So:
      C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
      
      We can group: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
      
      This is a linear combination of rows of B with coefficients from A.
      The tensor has rank at most 2*9 = 18 (schoolbook for this part), or we can do:
      C[i,k] = sum_j A[i,j] B[j,k]
      
      Using Strassen idea: define
      M1 = (A[0,0]-A[1,1])*(B[0,:] + B[1,:])  -> gives a vector
      M2 = (A[1,0]+A[1,1])*B[0,:]
      M3 = A[0,0]*(B[0,:] - B[1,:])
      M4 = A[1,1]*(B[1,:] - B[0,:])
      M5 = (A[0,0]+A[0,1])*B[1,:]
      M6 = (A[1,0]-A[0,0])*(B[0,:] + B[0,:])... 
      
      This is getting complicated. Let me just use the known rank-50 result directly.
      
      The construction: 
      - 18 terms for the j=2 part (schoolbook)
      - 32 terms for the j in {0,1} part using a rank-32 decomposition
      
      Wait, 18 + 32 = 50. So the 2x2x9 part has rank 32? That seems high.
      
      Actually, I recall now: for m x n * n x p where one dimension is small, there are known bounds.
      
      Let me just go with a simpler approach that's guaranteed to work: use the schoolbook for the j=2 part (18 terms) and for the j in {0,1} part, use 2*2*9 = 36 terms schoolbook, total 54. That's the same as full schoolbook.
      
      To get 50, I need to save 4 terms. The Strassen trick on the 2x2 part saves 1 term per 2x2x2 block. If I can apply this to 4 blocks of the 2x2x9... 
      
      Actually, 2x2x9: if I split the 9 into 4+4+1, I get 4 blocks of 2x2x2 (each rank 7 = 28) + 1 block of 2x2x1 (rank 4) = 32. Plus 18 for j=2. Total 50!
      
      Yes! That's it.
    """
    deadline = time.time() + 120
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Split inner dimension m=3 into 2+1
    # Part A: j in {0,1}, A_sub is 2x2, B_sub is 2x9
    # Part B: j in {2}, A_sub is 2x1, B_sub is 1x9
    
    # Part B: schoolbook, 2*1*9 = 18 terms
    U_B = []
    V_B = []
    W_B = []
    for i in range(n):
        for k in range(p):
            U_B.append([1] + [0]*(n*m-1))  # A[i,2] = 1
            V_B.append([0]*(m*p-1) + [1])   # B[2,k]... wait, need to index properly
            W_B.append([0]*(p*n-1) + [1])   # C[k*n+i]
    
    # Let me be more careful with indexing.
    # U has shape (n*m, rank) = (6, rank). Index: i*m + j
    # V has shape (m*p, rank) = (27, rank). Index: j*p + k
    # W has shape (p*n, rank) = (18, rank). Index: k*n + i
    
    U = []
    V = []
    W = []
    
    # Part B: j=2, schoolbook
    for i in range(n):
        for k in range(p):
            u_vec = [0] * (n*m)
            u_vec[i*m + 2] = 1  # A[i,2]
            
            v_vec = [0] * (m*p)
            v_vec[2*p + k] = 1  # B[2,k]
            
            w_vec = [0] * (p*n)
            w_vec[k*n + i] = 1  # C[i,k]
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
    
    # Part A: j in {0,1}, use Strassen on 2x2x2 blocks
    # Split p=9 into 4+4+1
    block_sizes = [4, 4, 1]
    strassen_terms = _strassen_2x2_decomposition()
    
    for block_start, block_size in [(0, 4), (4, 4), (8, 1)]:
        if block_size == 4:
            # 2x2x4: split into two 2x2x2 blocks
            for sub_start in [0, 2]:
                k_start = block_start + sub_start
                # 2x2x2 block: A is 2x2 (rows 0,1; cols 0,1), B is 2x2 (rows 0,1; cols k_start..k_start+1)
                # C is 2x2 (rows 0,1; cols k_start..k_start+1)
                for (u_2x2, v_2x2, w_2x2) in strassen_terms:
                    u_vec = [0] * (n*m)
                    v_vec = [0] * (m*p)
                    w_vec = [0] * (p*n)
                    
                    # u_2x2: [a00, a01, a10, a11] -> A[i,j] for i,j in {0,1}
                    for idx, val in enumerate(u_2x2):
                        if val != 0:
                            r, c = divmod(idx, 2)  # row, col in 2x2
                            u_vec[r*m + c] += val
                        # Actually u_2x2 is the coefficient for C, not A
                        # Let me re-read my Strassen terms...
                    
                    # I defined terms as (u, v, w) where:
                    # u is for C (2x2 flattened), v is for A (2x2 flattened), w is for B (2x2 flattened)
                    # C = sum_r u_r * (v_r outer w_r) ... no, that's not right.
                    # 
                    # In tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Wait, the convention is: C[i,k] = sum_j A[i,j] B[j,k]
                    # Tensor: T[i,j,k] = 1 if this term contributes
                    # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
                    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
                    # And C[i,k] = sum_r U[i*m+j, r] * ... 
                    
                    # Actually the standard is: 
                    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # No wait, that would give sum over j and r.
                    
                    # The correct convention for matrix multiplication tensor:
                    # T[i,j,k] = delta (it's 1 for all valid i,j,k in the multiplication)
                    # Actually T[i,j,k] = 1 always (for the multiplication tensor)
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # = sum_j sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
                    
                    # Hmm, I think the convention is:
                    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k] = 1
                    # where i indexes A rows, j indexes A cols / B rows, k indexes B cols / C cols
                    # And C[i,k] = sum_j A[i,j] B[j,k]
                    
                    # Let me just use: for each rank-1 term r:
                    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # and the sum over r should give 1 for the multiplication tensor.
                    
                    # For Strassen 2x2x2:
                    # A is 2x2, B is 2x2, C is 2x2
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # The 7 terms give: C = sum_{r=1}^{7} (linear form in A) * (linear form in B) * (position in C)
                    
                    # In my terms above:
                    # term (u, v, w): 
                    #   v is the linear form in A (2x2 flattened)
                    #   w is the linear form in B (2x2 flattened)  
                    #   u is where it lands in C (2x2 flattened)
                    
                    # So for the full problem:
                    # U[i*m+j, r] = v[j + ...] mapped to A[i,j]
                    # V[j*p+k, r] = w[k + ...] mapped to B[j,k]
                    # W[k*n+i, r] = u[...] mapped to C[i,k]
                    
                    pass  # I'll redo this properly below
    
    # Let me restart the construction more carefully.
    U = []
    V = []
    W = []
    
    # Part B: j=2, schoolbook (18 terms)
    for i in range(n):
        for k in range(p):
            u_vec = [0] * (n*m)
            u_vec[i*m + 2] = 1
            
            v_vec = [0] * (m*p)
            v_vec[2*p + k] = 1
            
            w_vec = [0] * (p*n)
            w_vec[k*n + i] = 1
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
    
    # Part A: j in {0,1}
    # Split k into blocks: [0,1], [2,3], [4,5], [6,7], [8]
    # For blocks of size 2: use Strassen (7 terms each)
    # For block of size 1: schoolbook (4 terms)
    
    strassen_terms = _strassen_2x2_decomposition()
    
    # Blocks: (k_start, k_size)
    blocks = [(0, 2), (2, 2), (4, 2), (6, 2), (8, 1)]
    
    for k_start, k_size in blocks:
        if k_size == 2:
            # 2x2x2 block: A(2x2) * B(2x2) -> C(2x2)
            # A indices: i in {0,1}, j in {0,1} -> A[i*m+j] = A[i*3+j]
            # B indices: j in {0,1}, k in {k_start, k_start+1} -> B[j*p+k]
            # C indices: i in {0,1}, k in {k_start, k_start+1} -> C[k*n+i]
            for (u_c, v_a, w_b) in strassen_terms:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                # v_a is 2x2 flattened [a00, a01, a10, a11] for A[i,j], i,j in {0,1}
                for idx, val in enumerate(v_a):
                    if val != 0:
                        r, c = divmod(idx, 2)
                        u_vec[r*m + c] = val  # A[r, c] where r in {0,1}, c in {0,1}
                
                # w_b is 2x2 flattened [b00, b01, b10, b11] for B[j,k], j in {0,1}, k in {k_start, k_start+1}
                for idx, val in enumerate(w_b):
                    if val != 0:
                        r, c = divmod(idx, 2)
                        v_vec[r*p + (k_start + c)] = val  # B[r, k_start+c]
                
                # u_c is 2x2 flattened [c00, c01, c10, c11] for C[i,k], i in {0,1}, k in {k_start, k_start+1}
                for idx, val in enumerate(u_c):
                    if val != 0:
                        r, c = divmod(idx, 2)
                        w_vec[(k_start + c)*n + r] = val  # C[k_start+c, r]
                
                U.append(u_vec)
                V.append(v_vec)
                W.append(w_vec)
        else:
            # k_size == 1: schoolbook 2x2x1 (4 terms)
            for i in range(2):
                for j in range(2):
                    u_vec = [0] * (n*m)
                    u_vec[i*m + j] = 1
                    
                    v_vec = [0] * (m*p)
                    v_vec[j*p + k_start] = 1
                    
                    w_vec = [0] * (p*n)
                    w_vec[k_start*n + i] = 1
                    
                    U.append(u_vec)
                    V.append(v_vec)
                    W.append(w_vec)
    
    rank = len(U)
    # Convert to lists of lists (already are)
    return {"U": U, "V": V, "W": W}
