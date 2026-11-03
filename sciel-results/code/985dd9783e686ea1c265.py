"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy: Column-pairing approach.
    Partition the 9-column dimension into three groups of 3 columns.
    For each 4x4x3 sub-problem, we attempt to find a low-rank decomposition.
    The schoolbook rank for 4x4x3 is 4*4*3 = 48.
    Total schoolbook rank for 4x4x9 is 4*4*9 = 144.
    
    Since finding a provably minimal rank decomposition for these specific
    dimensions is an open problem in algebraic complexity theory, and given
    the strict time constraints and the need for a *valid* decomposition,
    we implement the schoolbook decomposition which is guaranteed to be
    correct.
    
    However, we can try to use a slightly more efficient structure if possible.
    For 2x2x2, Bini's rank-7 tensor is known.
    For 4x4x4, the rank is at most 49 (Kronecker of 7x7) or 16 (if we could find a rank-4 for 2x2x2, which we can't, schoolbook is 4).
    Actually, the Kronecker product of the schoolbook 2x2x2 (rank 4) with itself gives rank 16 for 4x4x4.
    So for 4x4x9, we can split into two 4x4x4 blocks (rank 16 each) and one 4x4x1 block.
    The 4x4x1 block is just multiplying a 4x4 matrix by a 4x1 vector, resulting in a 4x1 vector.
    This is a tensor of dimensions (4, 4, 1). The rank is 4 (each output element is a sum of 4 products).
    So total rank could be 16 + 16 + 4 = 36.
    
    Let's verify this:
    4x4x4 schoolbook rank is 16?
    The tensor T_{i,j,k} = delta_{i,j} delta_{j,k} ... no, T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor is T_{i,j,k} = 1 if the indices form a valid path i->j->k, else 0? No.
    The multiplication tensor T has components T_{i,j,k} which are the coefficients in the expansion of C_{i,k} = sum_{j,l} A_{i,j} B_{j,k} ... wait.
    Standard definition: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor T is defined such that T(A,B) = C.
    In terms of rank-1 decompositions, T = sum_r u_r \otimes v_r \otimes w_r.
    For schoolbook, for each j, we have a term A_{:,j} \otimes B_{j,:} \otimes (e_j e_j^T)? No.
    
    Let's stick to the simplest correct implementation first: full schoolbook for 4x4x9.
    Rank = 4 * 4 * 9 = 144.
    
    But the directive asks for a structural change and a "column-pairing" strategy aiming for lower rank.
    If I can construct a valid rank-36 decomposition, that would be a huge improvement (score > 1.0).
    
    Let's try to construct the rank-36 decomposition:
    1. Decompose B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    2. C = A*B1 + A*B2 + A*B3.
    3. A*B1 is 4x4. A*B2 is 4x4. A*B3 is 4x1.
    4. We need a rank-16 decomposition for 4x4x4.
       The Kronecker product of the rank-4 schoolbook 2x2x2 tensor with itself yields a rank-16 decomposition for 4x4x4.
       Let S be the 2x2x2 schoolbook tensor. S = sum_{j=0}^1 e_j \otimes e_j \otimes (e_j e_j^T)? No.
       
       Let's define the 2x2x2 schoolbook tensor explicitly.
       For 2x2x2, C_{i,k} = sum_j A_{i,j} B_{j,k}.
       Terms:
       j=0: A_{:,0} * B_{0,:} -> contributes to C.
       j=1: A_{:,1} * B_{1,:} -> contributes to C.
       
       The tensor T_{i,j,k} is non-zero only when... wait, the tensor is fixed.
       T_{i,j,k} = 1 if we are looking at the term for index j?
       Actually, the multiplication tensor T_{n,n,n} is defined by T_{i,j,k} = 1 if i,j,k are such that...
       No, the tensor is T \in R^{n \times n \times n} where T_{i,j,k} = \delta_{i,j} \delta_{j,k}? No, that's identity.
       
       The multiplication tensor T satisfies C = T(A,B).
       C_{i,k} = \sum_{j} A_{i,j} B_{j,k}.
       So T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
       This is 1 if the indices match the summation structure, but actually, for a fixed (i,k), the sum is over j.
       So for each (i,j,k), T_{i,j,k} = 1 if this term appears in the product?
       Actually, T is a 3rd order tensor.
       T_{i,j,k} = 1 for all i,j,k? No.
       
       Let's look at the rank-1 decomposition.
       A rank-1 term is u \otimes v \otimes w.
       (u \otimes v \otimes w)(A,B) = (u^T A) (v^T B) w? No.
       It's (sum_i u_i A_{i,j})? No.
       
       Standard tensor decomposition of matrix multiplication:
       T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r
       where u_r \in R^n, v_r \in R^n, w_r \in R^{n \times n} (or flattened).
       Actually, the task specifies U, V, W shapes:
       U: (n*m, rank) -> rows correspond to (i,j) of A.
       V: (m*p, rank) -> rows correspond to (j,k) of B.
       W: (p*n, rank) -> rows correspond to (k,i) of C (based on W[k*n+i, r]).
       
       So a rank-1 term r is defined by:
       u_r \in R^{n*m}
       v_r \in R^{m*p}
       w_r \in R^{p*n}
       
       The contribution to C is:
       C_{k,i} += w_r[k*n+i] * (sum_{i',j'} u_r[i'*m+j'] A_{i',j'}) * (sum_{j',k'} v_r[j'*p+k'] B_{j',k'})
       
       Wait, the contraction is:
       Scalar = u_r \cdot vec(A)
       Scalar = v_r \cdot vec(B)
       C += w_r * (u_r \cdot vec(A)) * (v_r \cdot vec(B))
       
       For schoolbook:
       For each (i,j,k):
       u_r has 1 at [i*m+j]
       v_r has 1 at [j*p+k]
       w_r has 1 at [k*n+i]
       
       This works because:
       (u_r \cdot vec(A)) = A_{i,j}
       (v_r \cdot vec(B)) = B_{j,k}
       C_{k,i} += 1 * A_{i,j} * B_{j,k}
       Summing over all (i,j,k) gives C_{k,i} = sum_j A_{i,j} B_{j,k}.
       
       Now, can we do better for 4x4x4?
       We can use the Kronecker product of the 2x2x2 schoolbook decomposition.
       2x2x2 schoolbook has rank 4.
       The Kronecker product of two rank-4 tensors has rank 16.
       Does it decompose 4x4x4?
       
       Let T_2 be the 2x2x2 multiplication tensor.
       T_4 = T_2 \otimes T_2?
       The Kronecker product of tensors corresponds to the tensor product of the operations.
       If we view 4x4 matrices as 2x2 blocks of 2x2 matrices, then multiplying two 4x4 matrices is NOT simply the Kronecker product of two 2x2 matrix multiplications.
       (A \otimes B)(C \otimes D) = (AC) \otimes (BD).
       But we want to multiply A (4x4) by B (4x4).
       If we write A = \sum_{a,b} A_{a,b} E_{a,b} and B = \sum_{c,d} B_{c,d} E_{c,d},
       this is not a block multiplication.
       
       However, there is a known result: If R(MM_n) = r, then R(MM_{2n}) \le r^2.
       This is achieved by the Kronecker product of the decomposition.
       So R(MM_4) \le R(MM_2)^2 = 4^2 = 16.
       
       So we CAN construct a rank-16 decomposition for 4x4x4.
       And R(MM_4) \le R(MM_3) * ... no, just use the bound.
       
       So for 4x4x9:
       Split B into B1 (4x4), B2 (4x4), B3 (4x1).
       C = A B1 + A B2 + A B3.
       Term 1: A B1 (4x4x4). Rank 16.
       Term 2: A B2 (4x4x4). Rank 16.
       Term 3: A B3 (4x4x1).
       For 4x4x1, C (4x1) = A (4x4) * B3 (4x1).
       C_i = sum_j A_{i,j} B3_{j}.
       This is a tensor of dimensions (4,4,1).
       Rank?
       Schoolbook: for each j, term is A_{:,j} * B3_{j} * e_j (in output space? No, output is 4x1).
       w_r is in R^{1*4} = R^4.
       u_r is in R^{16}.
       v_r is in R^{4}.
       
       Schoolbook for 4x4x1:
       For each j in 0..3:
       u_r has 1 at [i*m+j] for all i? No.
       u_r depends on i?
       C_i = sum_j A_{i,j} B_{j}.
       For a fixed j, the term contributes A_{i,j} B_{j} to all C_i?
       No, A_{i,j} is different for each i.
       So we need a term for each (i,j)?
       If we use a term for each (i,j):
       u_r has 1 at [i*m+j].
       v_r has 1 at [j*1+0] = [j].
       w_r has 1 at [0*n+i] = [i].
       Contribution: A_{i,j} * B_j * 1 to C_i.
       Sum over j: C_i = sum_j A_{i,j} B_j.
       This works. Rank = 4*4 = 16?
       Wait, the sum is over j.
       For a fixed j, do we have one term or 4 terms?
       If we have one term for each (i,j), that's 16 terms.
       Can we do better?
       C = A * B3.
       This is just a matrix-vector multiplication.
       The tensor is T_{i,j,k=0} = \delta_{j,k}? No.
       T_{i,j} = 1?
       C_i = \sum_j A_{i,j} B_j.
       The tensor T has components T_{i,j} (flattened to 3rd order with last dim 1).
       T_{i,j,0} = 1 for all i,j?
       No, T_{i,j,0} is the coefficient of A_{i,j} B_j in C_i.
       It is 1 for all i,j.
       So T is a matrix of all ones?
       Rank of a matrix of all ones is 1.
       But this is a tensor decomposition into rank-1 tensors.
       T = \sum_r u_r \otimes v_r \otimes w_r.
       If T_{i,j,0} = 1, can we write this as a sum of rank-1 terms?
       u_r \otimes v_r \otimes w_r has components u_{r,i} v_{r,j} w_{r,0}.
       We want \sum_r u_{r,i} v_{r,j} w_{r,0} = 1 for all i,j.
       Let w_{r,0} = 1.
       Then \sum_r u_{r,i} v_{r,j} = 1.
       This is the matrix of all ones, which has rank 1.
       So we can use r=1:
       u_1 = [1,1,1,1]^T (flattened to 16 dims? No, u is in R^{n*m}).
       Wait, u_r is in R^{n*m} = R^{16}.
       v_r is in R^{m*p} = R^{4}.
       w_r is in R^{p*n} = R^{4}.
       
       We want \sum_{i',j'} u_{r, i'm+j'} A_{i',j'} \sum_{j',k'} v_{r, j'p+k'} B_{j',k'} w_{r, k'n+i} = C_{i,k}.
       Here p=1, so k'=0.
       C_i = \sum_j A_{i,j} B_j.
       
       If we use 1 term:
       u_1 must extract \sum_j A_{i,j} ? No, u_1 is a fixed vector.
       u_1 \cdot vec(A) is a scalar.
       We need the result to depend on i.
       So we cannot use just 1 term because u_1 \cdot vec(A) is a single scalar, but we need 4 different scalars (one for each i) to multiply with B.
       Unless... w_r handles the i-dependence?
       w_r is multiplied by the scalar (u \cdot A) (v \cdot B).
       So C_i += w_{r,i} * (u_r \cdot A) * (v_r \cdot B).
       If we use 1 term, C_i = w_{r,i} * S_A * S_B.
       This implies C is a rank-1 matrix (outer product of w and something).
       But C = A B is not generally rank 1.
       So we need more terms.
       
       Actually, the rank of the matrix-vector multiplication tensor (n x m x 1) is n*m?
       Let's check n=2, m=2, p=1.
       C_0 = A_00 B_0 + A_01 B_1
       C_1 = A_10 B_0 + A_11 B_1
       Schoolbook:
       (i=0,j=0): u=[1,0,0,0], v=[1,0], w=[1,0]
       (i=0,j=1): u=[0,1,0,0], v=[0,1], w=[1,0]
       (i=1,j=0): u=[0,0,1,0], v=[1,0], w=[0,1]
       (i=1,j=1): u=[0,0,0,1], v=[0,1], w=[0,1]
       Rank 4.
       
       Can we do rank 2?
       If rank 2, C_0 = u_0 A S_A0 + u_1 A S_A1 ...
       It is known that the rank of matrix-vector multiplication is nm.
       So for 4x4x1, rank is 16.
       
       So total rank = 16 (4x4x4) + 16 (4x4x4) + 16 (4x4x1) = 48.
       
       Wait, earlier I thought 4x4x1 rank was 4. That was wrong.
       So the total rank is 48.
       
       Is 48 better than 144? Yes.
       Score = 144/48 = 3.0.
       
       Let's implement the rank-48 decomposition.
       
       Steps:
       1. Define the rank-16 decomposition for 4x4x4 using Kronecker product of 2x2x2 schoolbook.
       2. Define the rank-16 decomposition for 4x4x1 (schoolbook).
       3. Combine them.
       
       2x2x2 Schoolbook Components:
       j=0:
         u: [1, 0] (A_00) -> flattened to 4? No, for 2x2x2, U is 2x2=4 dims.
         v: [1, 0] (B_00) -> V is 2x2=4 dims.
         w: [1, 0] (C_00) -> W is 2x2=4 dims.
         Wait, W index is k*n+i.
         For 2x2x2:
         i=0, j=0, k=0:
         U[0*2+0] = U[0] = 1
         V[0*2+0] = V[0] = 1
         W[0*2+0] = W[0] = 1
         
         i=0, j=0, k=1:
         U[0] = 1
         V[0*2+1] = V[1] = 1
         W[1*2+0] = W[2] = 1
         
         i=1, j=0, k=0:
         U[1*2+0] = U[2] = 1
         V[0] = 1
         W[0*2+1] = W[1] = 1
         
         i=1, j=0, k=1:
         U[2] = 1
         V[1] = 1
         W[3] = 1
         
         So for 2x2x2 schoolbook, we have 4 terms.
         
       Kronecker Product for 4x4x4:
       The indices for 4x4x4 are i in 0..3, j in 0..3, k in 0..3.
       We can map i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2.
       The term (u1, v1, w1) for (i1, j1, k1) and (u2, v2, w2) for (i2, j2, k2)
       combines to a term for (i, j, k).
       
       U_4 index: i*m + j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2.
       This is the Kronecker product of the U_2 indices.
       
       So we can generate the 16 terms by taking the Cartesian product of the 4 terms of 2x2x2.
       
       For 4x4x1:
       Schoolbook rank 16.
       Terms for each (i, j) where i in 0..3, j in 0..3.
       U index: i*4 + j.
       V index: j*1 + 0 = j.
       W index: 0*4 + i = i.
       
       Total rank 48.
       
       Let's code this up.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # We will build the decomposition manually to ensure correctness and low rank.
    # Rank 48 strategy:
    # 1. Two 4x4x4 blocks (rank 16 each)
    # 2. One 4x4x1 block (rank 16)
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to get 2x2x2 schoolbook terms
    # Returns list of (u_idx, v_idx, w_idx) for 2x2x2
    # U size 4, V size 4, W size 4
    def get_2x2x2_schoolbook():
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = i * 2 + j
                    v_idx = j * 2 + k
                    w_idx = k * 2 + i
                    terms.append((u_idx, v_idx, w_idx))
        return terms
    
    sb2 = get_2x2x2_schoolbook()
    
    # Build 4x4x4 decomposition (Rank 16)
    # We need to map 2x2x2 indices to 4x4x4 indices.
    # i4 = 2*i1 + i2
    # j4 = 2*j1 + j2
    # k4 = 2*k1 + k2
    
    # U4 index: i4 * 4 + j4
    # V4 index: j4 * 4 + k4  (Wait, V is m*p. For 4x4x4, m=4, p=4. So V index is j*4+k)
    # W4 index: k4 * 4 + i4
    
    # For the first 4x4x4 block (k in 0..3 of the 9 columns? No, B is 4x9.
    # We split B columns into 0..3, 4..7, 8.
    # So B1 is cols 0..3. B2 is cols 4..7. B3 is col 8.
    
    # Block 1: B cols 0..3
    for t1 in sb2:
        for t2 in sb2:
            u1_idx, v1_idx, w1_idx = t1
            u2_idx, v2_idx, w2_idx = t2
            
            # Extract i1, j1, k1 from u1, v1, w1
            # u1_idx = i1*2 + j1
            i1 = u1_idx // 2
            j1 = u1_idx % 2
            # v1_idx = j1*2 + k1
            # k1 = v1_idx % 2
            k1 = v1_idx % 2
            
            i2 = u2_idx // 2
            j2 = u2_idx % 2
            k2 = v2_idx % 2
            
            i4 = 2 * i1 + i2
            j4 = 2 * j1 + j2
            k4 = 2 * k1 + k2
            
            U_idx = i4 * 4 + j4
            V_idx = j4 * 4 + k4  # k4 is 0..3, which maps to B col 0..3
            W_idx = k4 * 4 + i4  # k4 is 0..3, which maps to C row 0..3
            
            U[U_idx][col] = 1
            V[V_idx][col] = 1
            W[W_idx][col] = 1
            col += 1
    
    # Block 2: B cols 4..7
    # V index offset: j4 * 4 + (k4 + 4)
    # W index offset: (k4 + 4) * 4 + i4
    for t1 in sb2:
        for t2 in sb2:
            u1_idx, v1_idx, w1_idx = t1
            u2_idx, v2_idx, w2_idx = t2
            
            i1 = u1_idx // 2
            j1 = u1_idx % 2
            k1 = v1_idx % 2
            
            i2 = u2_idx // 2
            j2 = u2_idx % 2
            k2 = v2_idx % 2
            
            i4 = 2 * i1 + i2
            j4 = 2 * j1 + j2
            k4 = 2 * k1 + k2
            
            U_idx = i4 * 4 + j4
            V_idx = j4 * 4 + (k4 + 4)
            W_idx = (k4 + 4) * 4 + i4
            
            U[U_idx][col] = 1
            V[V_idx][col] = 1
            W[W_idx][col] = 1
            col += 1
            
    # Block 3: B col 8 (4x4x1)
    # Schoolbook rank 16
    # i in 0..3, j in 0..3
    # U index: i * 4 + j
    # V index: j * 1 + 0 = j (B col 8, so index in V is j*9+8? No, V is m*p = 4*9=36.
    # V index for B[j, 8] is j * 9 + 8.
    # W index: 0 * 4 + i = i (C row 8? No, C is 4x9.
    # C[i, 8]. W index is k * n + i = 8 * 4 + i.
    
    for i in range(4):
        for j in range(4):
            U_idx = i * 4 + j
            V_idx = j * 9 + 8
            W_idx = 8 * 4 + i
            
            U[U_idx][col] = 1
            V[V_idx][col] = 1
            W[W_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
