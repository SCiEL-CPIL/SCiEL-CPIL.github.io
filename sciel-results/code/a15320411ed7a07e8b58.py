"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The tensor T_{i,j,k} = delta_{i,k} for matrix multiplication A(2x4) * B(4x4) = C(2x4).
    Here i indexes rows of A and C (0..1), j indexes columns of A and rows of B (0..3),
    k indexes columns of B and C (0..3).
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use a greedy rank-1 decomposition approach, but also exploit structure.
    For 2x4x4, the schoolbook rank is 2*4*4 = 32.
    
    Let's think about whether we can do better. The tensor is:
    T[i,j,k] = 1 if i == k (where i is in {0,1}, k is in {0,3}), else 0.
    Wait, that's not right. Let me re-excarefully.
    
    Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor is T[i,j,k] which gives the coefficient of A[i,j]*B[j,k] in C[i,k].
    T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which is always true
    when we're computing C[i,k] = sum_j A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid i,j,k?
    No, that's not right either.
    
    Actually, the tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    T[i, j, k] where i is row of first matrix (0..n-1), j is the "inner" index (0..m-1),
    k is column of second matrix (0..p-1). The tensor value T[i,j,k] = 1 if this term
    appears in the product, which it does for C[i,k] = sum_j A[i,j]*B[j,k].
    
    So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    This is just the all-ones tensor of size n*m*p!
    
    Wait, that can't be right for rank minimization. Let me reconsider.
    
    Actually, the multiplication tensor M_{n,m,p} has entries:
    M_{(i,j),(j,k),(i,k)} ... no, let me use the standard formulation.
    
    The tensor is in U ⊗ V ⊗ W where:
    - U has basis e_{i,j} for i in [0,n), j in [0,m) (entries of A)
    - V has basis f_{j,k} for j in [0,m), k in [0,p) (entries of B)  
    - W has basis g_{i,k} for i in [0,n), k in [0,p) (entries of C)
    
    The tensor T = sum_{i,j,k} e_{i,j} ⊗ f_{j,k} ⊗ g_{i,k}
    
    So T[i*m+j, j*p+k, i*p+k] = 1 (using flattened indices).
    
    For n=2, m=4, p=4:
    - U has 8 dimensions (i=0,1; j=0,1,2,3)
    - V has 16 dimensions (j=0,1,2,3; k=0,1,2,3)
    - W has 8 dimensions (i=0,1; k=0,1,2,3)
    
    The tensor has 2*4*4 = 32 non-zero entries, each being 1.
    The schoolbook decomposition uses rank 32.
    
    Can we do better? The tensor rank of matrix multiplication tensors is known to be
    less than n*m*p for some cases, but for small dimensions it's often equal.
    
    For 2x4x4, let's think: this can be seen as two independent 1x4x4 multiplications
    (one for each row i of A). Each 1x4x4 multiplication has tensor rank 4 (since it's
    just sum_k e_{j,k} ⊗ f_{j,k} ⊗ g_k for a fixed i, which is a rank-4 tensor).
    
    Wait, for 1xm*p: T[j,k] = sum_j A[j]*B[j,k] = C[k]. The tensor is sum_j f_{j} ⊗ g_{j,k} ⊗ h_k.
    This is a rank-m tensor (one term per j). So 1x4x4 has rank 4.
    
    For 2x4x4, we have two such blocks (i=0 and i=1), so the rank is at most 4+4=8?
    No wait, that's not right either. Let me re-examine.
    
    For i=0: terms are e_{0,j} ⊗ f_{j,k} ⊗ g_{0,k} for j=0..3, k=0..3. That's 16 terms.
    For i=1: terms are e_{1,j} ⊗ f_{j,k} ⊗ g_{1,k} for j=0..3, k=0..3. That's 16 terms.
    
    Total: 32 terms, matching schoolbook.
    
    But can we combine terms across i? The issue is that e_{0,j} and e_{1,j} are different
    basis vectors, and g_{0,k} and g_{1,k} are different. So we can't simply combine them.
    
    However, maybe we can use a different decomposition. Let's think of it as:
    For each j, we have a 2x4 sub-problem: A[:,j] (a 2-vector) times B[j,:] (a 4-vector)
    gives a 2x4 matrix contribution to C. This is an outer product of a 2-vector and a 4-vector,
    which has rank 1 as a matrix, but as a tensor term it's e_{:,j} ⊗ f_{j,:} ⊗ g_{:,:}.
    
    Actually, I think the rank is indeed 32 for 2x4x4. Let me verify with known results.
    
    The tensor rank of matrix multiplication:
    - 1x1x1: rank 1
    - 1xmxp: rank m (since C[k] = sum_j A[j]B[j,k], need m terms)
    - 2x1x1: rank 2
    - 2x2x2: rank 7 (Strassen)
    - 2x2x1: rank 2
    - 2x4x4: ?
    
    For 2x4x4, we can view it as block multiplication. If we partition the 4x4 matrix into
    2x2 blocks, we get a 2x2 block matrix times a 2x2 block matrix (with 2x2 blocks).
    This is like multiplying two 2x2 matrices where each entry is a 2x2 matrix.
    The rank would be at most 7 * (rank of 2x2x2 with 2x2 entries)... this gets complicated.
    
    Actually, let me just try the greedy approach and see if we can find a decomposition
    with rank < 32. If not, we'll use 32.
    
    Let me implement a greedy rank-1 decomposition.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 16
    W_dim = p * n  # 8
    
    # Build the full tensor with trace convention W[k*n+i, r]
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Structured rank-8 decomposition: 2 terms per inner index j
    # For each j, use one term for i=0 (all k) and one term for i=1 (all k)
    rank = 2 * m  # 8
    
    U_mat = np.zeros((U_dim, rank))
    V_mat = np.zeros((V_dim, rank))
    W_mat = np.zeros((W_dim, rank))
    
    for j in range(m):
        r0 = 2 * j      # term for i=0
        r1 = 2 * j + 1  # term for i=1
        
        for k in range(p):
            # i=0 term
            U_mat[0 * m + j, r0] = 1.0
            V_mat[j * p + k, r0] = 1.0
            W_mat[k * n + 0, r0] = 1.0
            
            # i=1 term
            U_mat[1 * m + j, r1] = 1.0
            V_mat[j * p + k, r1] = 1.0
            W_mat[k * n + 1, r1] = 1.0
    
    # Verify the decomposition
    recon = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
    if not np.allclose(recon, T, atol=1e-10):
        # Fallback to schoolbook rank 32
        rank = n * m * p
        U_mat = np.zeros((U_dim, rank))
        V_mat = np.zeros((V_dim, rank))
        W_mat = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_mat[i * m + j, col] = 1
                    V_mat[j * p + k, col] = 1
                    W_mat[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[float(U_mat[i, r]) for r in range(rank)] for i in range(U_dim)]
    V_list = [[float(V_mat[i, r]) for r in range(rank)] for i in range(V_dim)]
    W_list = [[float(W_mat[i, r]) for r in range(rank)] for i in range(W_dim)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
