"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5
    
    # Use a block-based construction.
    # Split the middle dimension m=5 into m1=2 and m2=3.
    # (n x m1) * (m1 x p) + (n x m2) * (m2 x p)
    # For each sub-problem, use schoolbook but with sharing.
    
    # Actually, let me use a direct construction with explicit rank reduction.
    # The key insight: for (i,j,k), the term A[i,j]*B[j,k] contributes to (AB)[i,k].
    # We need: 1 = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] for all i,j,k.
    
    # Schoolbook uses one term per (i,j,k) triple, giving rank 75.
    # To reduce rank, we need multiple (i,j,k) triples to share a rank-1 term.
    # This requires U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] to be non-zero for multiple (i,j,k).
    
    # Let me try: for each i, use a rank-1 term that covers all j,k for that i.
    # U[i*m+j, c_i] = 1 for all j, V[j*p+k, c_i] = 1 for all j,k, W[k*n+i, c_i] = 1 for all k.
    # Then for fixed i, sum_c U V W = sum over c where c=c_i: 1*1*1 = 1 for all j,k. 
    # But we need it to be 1 for each (i,j,k), and the sum is over c, so for (i,j,k) we get 1.
    # But different i use different c, so this works! Rank = n = 3?
    # Wait, no. For (i,j,k), we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    # If for a given i, we set U[i*m+j, c_i] = 1 for all j, V[j*p+k, c_i] = 1 for all j,k, 
    # W[k*n+i, c_i] = 1 for all k, then for (i,j,k): 
    # sum_c = U[i*m+j,c_i] * V[j*p+k,c_i] * W[k*n+i,c_i] = 1*1*1 = 1. 
    # But V[j*p+k, c_i] = 1 for ALL j,k, which means V has 1s in all rows for column c_i.
    # And W[k*n+i, c_i] = 1 for all k.
    # And U[i*m+j, c_i] = 1 for all j.
    # This works! Rank = 3!
    
    # Wait, that can't be right. Let me check: if rank=3, then R(3,5,5)=3, which is absurd.
    # The error: V[j*p+k, c] must be the same for all (j,k) that share the same c.
    # If V[j*p+k, c_i] = 1 for all j,k, and U[i*m+j, c_i] = 1 for all j, 
    # W[k*n+i, c_i] = 1 for all k, then for (i',j,k') where i' != i:
    # U[i'*m+j, c_i] = 0 (since only rows for i=i have 1s in column c_i for U)
    # So it's 0, which is fine.
    # For (i,j,k): U[i*m+j,c_i]=1, V[j*p+k,c_i]=1, W[k*n+i,c_i]=1, product=1. Sum=1. ✓
    # For (i,j,k') where k' != k: same, W[k'*n+i,c_i]=1, so still 1. ✓
    # For (i',j,k) where i' != i: U[i'*m+j,c_i]=0, so 0. ✓
    # 
    # So rank = 3 works?! That means R(3,5,5) <= 3, which is clearly wrong because
    # R(1,1,1)=1 and R(n,1,p) >= n*p... wait no.
    
    # I'm making an error. Let me re-examine. The condition is:
    # for ALL i in [0,n), j in [0,m), k in [0,p): 
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # 
    # With my construction: U[i*m+j, c_i] = 1 iff i' = i (for the row index i*m+j).
    # So U[row, col] = 1 only when row corresponds to the same i as col's i.
    # V[j*p+k, c_i] = 1 for ALL j,k. So V[row, col] = 1 for all rows when col=c_i.
    # W[k*n+i, c_i] = 1 for all k. So W[row, col] = 1 for all rows when col=c_i.
    # 
    # For (i,j,k): sum_c = U[i*m+j, c_i] * V[j*p+k, c_i] * W[k*n+i, c_i] = 1*1*1 = 1. ✓
    # For (i',j,k) with i' != i: U[i'*m+j, c_i] = 0, so contribution 0. 
    #   And for c_{i'}, U[i'*m+j, c_{i'}] = 1, V[j*p+k, c_{i'}] = 1, W[k*n+i', c_{i'}] = 1, 
    #   but we need W[k*n+i, c_{i'}] for the (i',j,k) tuple, which is W[k*n+i', c_{i'}] = 1. 
    #   So sum = 0 + 1 = 1. ✓
    # 
    # This seems to work! But R(3,5,5) <= 3 is impossible. Where's the flaw?
    # 
    # OH WAIT. The issue is that V[j*p+k, c] is shared across different i values.
    # V has shape (m*p, rank) = (25, rank). V[j*p+k, c_i] = 1 for all j,k means 
    # column c_i of V is all 1s. Similarly W column c_i is all 1s (for rows k*n+i, 
    # which is all rows since i is fixed... no, W has shape (p*n, rank) = (15, rank), 
    # and W[k*n+i, c_i] = 1 for all k means rows i, i+n, i+2n, ... which is 5 rows).
    # 
    # Let me recheck: for (i,j,k), the W index is k*n+i. For fixed i, as k ranges 0..4, 
    # k*n+i ranges i, i+3, i+6, i+9, i+12. So 5 specific rows.
    # V index j*p+k: for fixed i (via c_i), j ranges 0..4, k ranges 0..4, so all 25 rows.
    # 
    # So the construction is valid and gives rank 3. But this contradicts known results.
    # 
    # I think I'm misunderstanding the tensor decomposition. Let me re-read the problem.
    # 
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]" with "exactly decompose the matrix multiplication tensor"
    # 
    # I believe the correct condition is:
    # For all i,j,k: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # where T[i,j,k] = 1 for all i,j,k (the matmul tensor is all-ones in this indexing).
    # 
    # And my construction satisfies this. So R(3,5,5) <= 3? That can't be.
    # 
    # Let me check with n=m=p=1: R(1,1,1) should be 1. My construction: rank=1, 
    # U[0,0]=1, V[0,0]=1, W[0,0]=1. 1=1*1*1. ✓
    # 
    # n=1,m=2,p=1: R(1,2,1) should be 2 (since (1x2)(2x1) needs 2 mults).
    # My construction: rank = n = 1. U[0*2+0, c_0]=1, U[0*2+1, c_0]=1, V[0*1+0, c_0]=1, 
    # V[1*1+0, c_0]=1, W[0*1+0, c_0]=1. 
    # For (0,0,0): U[0,c0]*V[0,c0]*W[0,c0] = 1. ✓
    # For (0,1,0): U[1,c0]*V[1,c0]*W[0,c0] = 1. ✓
    # So rank 1 works for (1,2,1)? But (1x2)(2x1) = A[0,0]*B[0,0] + A[0,1]*B[1,0], 
    # which needs 2 multiplications. How can rank 1 work?
    # 
    # AH I see the issue. The rank-1 term U[0,c]*V[0,c]*W[0,c] + U[1,c]*V[1,c]*W[0,c] 
    # is NOT a single rank-1 term in the tensor sense. A rank-1 term in the 3-tensor 
    # T[i,j,k] is of the form u[i]*v[j]*w[k]. Here, for fixed c, the term is 
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]. 
    # 
    # For n=1,m=2,p=1, c=0: 
    #   i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0]
    #   i=0,j=1,k=0: U[1,0]*V[1,0]*W[0,0]
    # These are two separate tensor entries, not a single rank-1 tensor.
    # 
    # A rank-1 tensor T[i,j,k] = u[i]*v[j]*w[k] has the property that T[i,j,k] factors 
    # as a product of three vectors. In our case, for fixed c, the "tensor" is 
    # T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
    # This does NOT factor as u[i]*v[j]*w[k] in general!
    # 
    # So my construction is NOT a valid rank decomposition in the tensor sense.
    # The tensor rank requires each rank-1 component to be of the form u[i]*v[j]*w[k].
    # 
    # OK so I need to be more careful. Let me just use schoolbook.
    
    rank = n * m * p
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
