import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    # Rank-20 decomposition for 3x4x4 core, then extend to 3x4x8 with rank 40
    # The 3x4x4 tensor T_{i,j,k} = delta_{i,0}*delta_{j,k} for i in 0..2, j in 0..3, k in 0..3
    # We partition: for each of 4 "blocks" in the k-direction (two 4-blocks for 8),
    # and for each block, we use a rank-10 decomposition of the 3x4x4 slice.
    # Actually, the tensor is sum over j of e_i_j (in U) * e_jk (in V) * e_ki (in W)
    # For 3x4x8: rank = 96 schoolbook. 
    # Better: use the fact that the 4x4 multiplication has rank 16 (schoolbook) or can be lower.
    # Actually for 3x4x4, the tensor rank is 12 (schoolbook) but can be reduced.
    # Let me just use a direct approach: the tensor is separable.
    # T[i,j,k] = 1 if the multiplication is correct.
    # Actually, the matrix multiplication tensor for (n x m) * (m x p) has:
    # U indexed by (i, j) for i in [0,n), j in [0,m)
    # V indexed by (j, k) for j in [0,m), k in [0,p)  
    # W indexed by (k, i) for k in [0,p), i in [0,n)
    # And the tensor T = sum_{c} u_c * v_c * w_c where
    # (u_c * v_c * w_c)[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This must equal delta_{i,j} * delta_{j,k} ... no, it equals 1 when the 
    # "multiplication" contributes. Actually T[i,j,k] = delta_{i,j}*delta_{j,k} is wrong.
    # The correct tensor: for matrix multiply C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i, j, k] = 1 (it's the elementary tensor)
    # And we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k] = 1 for all valid i,j,k
    # Wait, that's not right either. The tensor is:
    # T[i,j,k] should be such that contracting U, V, W gives the multiplication.
    # Actually the standard form: the multiplication tensor mu_{n,m,p} has:
    # mu_{n,m,p} = sum_{i,j,k} e_i^{(n)} * e_j^{(m)} * e_k^{(p)} 
    # No, that's not right.
    
    # Let me re-read the task: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) with the combined index k*n + i.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The decomposition means: for each (i, j, k):
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = delta_{i,j} * delta_{j,k}... 
    # No. The matrix multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    # So the "tensor" we're decomposing is the bilinear map (A, B) -> C
    # The tensor T has components T[i, j, k] where i indexes row of A/col of C,
    # j indexes col of A / row of B, k indexes col of B / col of C.
    # T[i,j,k] = 1 for all valid (i,j,k) - it's the full tensor of 1s in the 
    # appropriate subspace? No, that's not right either.
    
    # Actually, the matrix multiplication tensor is:
    # mu = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C
    # where e_i^A picks out A[i,*], e_j^B picks out B[*,j], e_k^C picks out C[*,k]
    # But in the bilinear form: <mu, (A,B)> = sum_{i,j,k} A[i,j] B[j,k] C^*[i,k]
    # So the tensor is mu[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the rank decomposition is: mu = sum_r u_r * v_r * w_r where
    # u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # and u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] summed over r gives 1 for all valid (i,j,k)
    
    # Schoolbook: for each (i,j,k), u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1
    # That's rank n*m*p = 96.
    
    # For a better decomposition, we can use the fact that the tensor is 
    # "separable" in some sense. Actually, the 3x4x8 tensor has a known 
    # lower rank. Let me think about this differently.
    
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,8)
    # We want to write T as sum_r u_r * v_r * w_r (outer product)
    # This is equivalent to finding the tensor rank of the all-ones tensor 
    # of shape (3,4,8) in the "multiplication" format.
    
    # Wait, I think I'm confusing myself. Let me re-derive.
    # The matrix multiplication bilinear map: (A, B) -> A*B
    # where A is 3x4, B is 4x8, A*B is 3x8.
    # The tensor T is in R^{3*4} * R^{4*8} * R^{3*8} (or with the trace convention)
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # No, that's not right.
    
    # OK let me just use the standard: the multiplication tensor for m x n * n x p
    # has rank at most m*n*p (schoolbook) and at least something lower.
    # For 3x4x8, I'll try a rank-40 decomposition.
    
    # Strategy: Use the 4x4x4 tensor structure.
    # Actually, let me just return the schoolbook for now and verify correctness.
    # Then I can try to improve.
    
    deadline = time.time() + 140  # safety margin
    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
