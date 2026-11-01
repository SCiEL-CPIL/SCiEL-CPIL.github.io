import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Use Kronecker product: (2x2x2 Strassen) ⊗ (1x2x3)
    # T1: 2x2x2 with rank 7 (Strassen)
    # T2: 1x2x3 with rank 3 (each output is independent dot product)
    # Total: 2x4x6 with rank 7*3 = 21
    
    rank = 21
    
    # Strassen factors for 2x2x2
    # Local indices for 2x2: 0:00, 1:01, 2:10, 3:11
    sU = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    sV = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    sW = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    # Factors for 1x2x3: rank 3
    # A is 1x2, B is 2x3, C is 1x3
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # For each k in {0,1,2}: independent dot product of length 2
    # So rank 3, with:
    # U_t2[i, k] = 1 if i==k else 0 (but i is always 0 for 1x2, so U_t2[0,k]=1)
    # Wait, let me think again.
    # For 1x2x3: U has shape (1*2, 3) = (2,3), V has shape (2*3, 3) = (6,3), W has shape (3*1, 3) = (3,3)
    # Actually, the convention is U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For 1x2x3: n=1, m=2, p=3, so U is (2, rank), V is (6, rank), W is (3, rank)
    # C[0,k] = sum_j A[0,j]*B[j,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # This can be decomposed as: for each k, one rank-1 term
    # Term k: U[:,k] = [1, 0] if k corresponds to B[0,k], but that's not right
    # Actually: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # This is sum of 2 terms, but they share the same U and V structure
    # Let me use: for each k in {0,1,2}:
    #   U[:,k] = [1, 1]  (both A[0,0] and A[0,1] contribute)
    #   V[0*3+k, k] = 1, V[1*3+k, k] = 1  (B[0,k] and B[1,k])
    #   W[k*1+0, k] = 1  (C[0,k])
    # Wait, that gives: sum_k U[0,k]*V[0*3+k,k]*W[k,0] + U[1,k]*V[1*3+k,k]*W[k,0]
    # = sum_k (1*B[0,k]*1 + 1*B[1,k]*1) = sum_k (B[0,k] + B[1,k]) which is wrong
    # 
    # Let me reconsider. The tensor T[i,j,k] = delta_{i,0} * delta_{j,k} (since C[0,k] = sum_j A[0,j]*B[j,k])
    # Actually no, T[i,j,k] represents the coefficient in C[i,k] = sum_j A[i,j]*B[j,k]
    # For 1x2x3: T[0,j,k] = 1 for all j,k (since C[0,k] = sum_j A[0,j]*B[j,k])
    # Wait, that's not right either. The multiplication tensor has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # For 1x2x3: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # So T[0,0,k] = 1 and T[0,1,k] = 1 for each k
    # This tensor has rank 3 (one for each k)
    # Decomposition: for each k in {0,1,2}:
    #   U[:,k] = [1, 1]  (A[0,0] and A[0,1] both have coefficient 1)
    #   V[j*3+k, k] = 1 for j in {0,1}  (B[j,k] has coefficient 1)
    #   W[k, 0] = 1  (C[0,k] has coefficient 1)
    # Check: sum_k U[0,k]*V[0*3+k,k]*W[k,0] + U[1,k]*V[1*3+k,k]*W[k,0]
    # = sum_k (1*B[0,k]*1 + 1*B[1,k]*1) = sum_k (B[0,k] + B[1,k])
    # But we want C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for each k separately
    # The issue is that the sum over k mixes different output columns
    # 
    # I think I'm confusing myself. Let me use the standard definition.
    # The tensor for matrix multiplication (n,m,p) has entries T[i,j,k] where:
    # i in {0,...,n-1}, j in {0,...,m-1}, k in {0,...,p-1}
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], else 0
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, that's not right. Each product A[i,j]*B[j,k] contributes to C[i,k], but C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in a sense, but with the constraint that j is shared)
    # Actually, the multiplication tensor is defined as: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the index j is "contracted"
    # More precisely: T[i,j,k] = 1 if and only if the triple (i,j,k) is valid, which it always is for standard matrix mult
    # But that would give rank n*m*p, which is the naive rank
    # 
    # I think the correct definition is: the tensor has entries T[i,j,k] = 1 for all i,j,k, and the rank is the minimum number of rank-1 tensors that sum to T
    # For 1x2x3: T is a 1x2x3 tensor with all entries 1
    # T[0,j,k] = 1 for j in {0,1}, k in {0,1,2}
    # This can be written as: sum_{k=0}^{2} (e_0 ⊗ (e_0 + e_1) ⊗ e_k)
    # So rank = 3
    # Decomposition:
    #   For k=0: U[:,0] = [1,1], V[0*3+0,0]=1, V[1*3+0,0]=1, W[0*1+0,0]=1
    #   For k=1: U[:,1] = [1,1], V[0*3+1,1]=1, V[1*3+1,1]=1, W[1*1+0,1]=1
    #   For k=2: U[:,2] = [1,1], V[0*3+2,2]=1, V[1*3+2,2]=1, W[2*1+0,2]=1
    # Check: C[0,0] = sum_k U[0,k]*V[0, k]*W[0,k] + U[1,k]*V[3+k, k]*W[0,k]
    # = U[0,0]*V[0,0]*W[0,0] + U[1,0]*V[3,0]*W[0,0] = 1*1*1 + 1*1*1 = 2
    # But we want C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # The issue is that U and V are multiplied, so we get A[0,0]*A[0,1]*B[0,0]*B[1,0] which is wrong
    # 
    # I see the problem. The Kronecker product structure means that if T = T1 ⊗ T2, then:
    # T[i1,i2, j1,j2, k1,k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
    # And the decomposition factors are:
    # U[i1*m2 + i2, r1*R2 + r2] = U1[i1, r1] * U2[i2, r2]
    # V[j1*p2 + j2, r1*R2 + r2] = V1[j1, r1] * V2[j2, r2]
    # W[k1*n2 + k2, r1*R2 + r2] = W1[k1, r1] * W2[k2, r2]
    # where T1 is (n1,m1,p1) and T2 is (n2,m2,p2)
    # 
    # So for T = T1 ⊗ T2 where T1 is 2x2x2 (Strassen, rank 7) and T2 is 1x2x3 (rank 3):
    # n1=2, m1=2, p1=2, n2=1, m2=2, p2=3
    # Total: n=n1*n2=2, m=m1*m2=4, p=p1*p2=6
    # U[i1*2 + i2, r1*3 + r2] = U1[i1, r1] * U2[i2, r2]
    # V[j1*3 + j2, r1*3 + r2] = V1[j1, r1] * V2[j2, r2]
    # W[k1*1 + k2, r1*3 + r2] = W1[k1, r1] * W2[k2, r2]
    # 
    # Now I need the factors for T2 (1x2x3):
    # T2[i2,j2,k2] = 1 for all i2 in {0}, j2 in {0,1}, k2 in {0,1,2}
    # Rank 3 decomposition:
    # For r2 in {0,1,2} (corresponding to k2=r2):
    #   U2[i2, r2] = 1 for all i2 (since i2 is always 0)
    #   V2[j2*3 + k2, r2] = 1 if k2 == r2 else 0
    #   W2[k2*1 + i2, r2] = 1 if k2 == r2 else 0
    # Wait, W2 has shape (p2*n2, rank) = (3*1, 3) = (3,3)
    # W2[k2, r2] = 1 if k2 == r2 else 0
    # 
    # Let me verify: C2[i2,k2] = sum_{j2,r2} U2[i2,r2]*V2[j2*3+k2,r2]*W2[k2,r2]
    # = sum_{j2,r2} 1 * (1 if k2==r2 else 0) * B2[j2,k2] * (1 if k2==r2 else 0)
    # = sum_{j2} B2[j2,k2]  (since only r2=k2 contributes)
    # = B2[0,k2] + B2[1,k2]
    # But we want C2[0,k2] = A2[0,0]*B2[0,k2] + A2[0,1]*B2[1,k2]
    # The problem is that U2 should depend on A2, but in the Kronecker product, U2 is fixed
    # 
    # I think I'm misunderstanding the Kronecker product structure. Let me look at this differently.
    # 
    # Actually, I think the issue is that T2 (1x2x3) is not the standard multiplication tensor.
    # For 1x2x3 matrix multiplication: C[0,k] = sum_j A[0,j]*B[j,k]
    # The tensor T2[i,j,k] should be such that C[i,k] = sum_j A[i,j]*B[j,k] = sum_{i,j,k} T2[i,j,k] A[i,j] B[j,k]
    # But this doesn't work because the indices don't match up
    # 
    # Let me restart. The multiplication tensor for (n,m,p) is defined as:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i is the i-th basis vector in R^n, etc.
    # Wait, that's not right either.
    # 
    # The correct definition: T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k]
    # For standard matrix multiplication, every A[i,j]*B[j,k] appears in C[i,k] exactly once
    # So T[i,j,k] = 1 for all valid i,j,k
    # But that would make T the all-ones tensor, which has rank min(n*m, m*p, p*n) = min(n*m, m*p, p*n)
    # For 1x2x3: rank = min(2, 6, 3) = 2? But we know the rank should be 3
    # 
    # I'm clearly confused about the definition. Let me just use the fact that for 1xm p, the rank is p
    # (each output column is an independent dot product)
    # So T2 (1x2x3) has rank 3
    # 
    # Given the confusion, let me just keep the current rank 42 solution and verify it's correct,
    # then try to find a better decomposition through other means.
    
    # Actually, let me just return the current solution which is known to work
    n, m, p = 2, 4, 6
    rank = 42

    # Pre-define Strassen factors
    # Local indices for 2x2: 0:00, 1:01, 2:10, 3:11
    sU = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    sV = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    sW = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for r in range(rank):
        # r = j_block * 21 + k_block * 7 + prod_idx
        # 2 j_blocks, 3 k_blocks, 7 prods
        rem = r
        prod_idx = rem % 7
        rem = rem // 7
        k_block = rem % 3
        j_block = rem // 3

        u_factors = sU[prod_idx]
        v_factors = sV[prod_idx]
        w_factors = sW[prod_idx]

        for li, val in u_factors:
            gi = li // 2
            gj = li % 2
            U[gi * m + (j_block * 2 + gj), r] = val

        for li, val in v_factors:
            gj = li // 2
            gk = li % 2
            V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = val

        for li, val in w_factors:
            gi = li // 2
            gk = li % 2
            W[(k_block * 2 + gk) * n + gi, r] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
