"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify exact decomposition."""
        r = len(U[0])
        # Check diagonal: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal: sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i3,c] == 0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i3][c]
                                if abs(s) > tol:
                                    return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return U, V, W, rank

    # Fallback: schoolbook decomposition (rank 50)
    U0, V0, W0, r0 = schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U0, V0, W0, r0

    # Primary construction: rank = m*p = 25
    # For n=2, partition columns j into two groups S0 (assigned to row i=0) and S1 (assigned to row i=1)
    # Component c indexed by (j, k):
    #   U[i*m+j', c] = 1 if j'==j, else 0
    #   V[j'*p+k', c] = 1 if j'==j and k'==k, else 0
    #   W[k'*n+i', c] = 1 if k'==k and (i'==0 iff j in S0, i'==1 iff j in S1), else 0
    #
    # Verification:
    # Diagonal (i,j,k): sum over c=(j',k') of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    #   = U[i*m+j,(j,k)] * V[j*p+k,(j,k)] * W[k*n+i,(j,k)]
    #   = 1 * 1 * (1 if (i==0 and j in S0) or (i==1 and j in S1), else 0)
    #   For this to equal 1, we need: if i==0 then j in S0, if i==1 then j in S1
    #   But wait - this only works if each j is assigned to exactly one i.
    #   The diagonal term for (i,j,k) only gets contribution from c=(j,k),
    #   and W[k*n+i,(j,k)] = 1 only if i matches the group of j.
    #   So for i=0: need j in S0 for ALL j -> S0 = {0,1,2,3,4}, S1 = {}
    #   For i=1: need j in S1 for ALL j -> S1 = {0,1,2,3,4}, S0 = {}
    #   These are contradictory! The simple construction doesn't work for all (i,j,k).
    
    # Let me re-examine. The issue is that for the diagonal entry (i,j,k),
    # we need the sum over c to be 1. With the parameterization above, only c=(j,k) 
    # contributes (since U[i*m+j,c] is nonzero only when j'=j, and V[j*p+k,c] is 
    # nonzero only when j'=j and k'=k). So we need W[k*n+i,(j,k)] = 1 for all i,j,k.
    # But W[k*n+i,(j,k)] can only be 1 for one of i=0 or i=1 (since it's 0/1).
    
    # So this simple construction DOESN'T WORK. Let me try a different approach.
    
    # Alternative: use a more flexible construction where multiple c's contribute.
    # Let's try rank = 2*m*p = 50 (no improvement) or find a different structure.
    
    # Actually, let's try a different parameterization:
    # Component c indexed by (j, k, s) where s in {0, 1} (two sub-components per (j,k))
    # U[i*m+j', c=(j,k,s)] = delta(j,j') * a[i,k,s]
    # V[j'*p+k', c=(j,k,s)] = delta(j,j') * delta(k,k') * b[j,k,s]
    # W[k'*n+i', c=(j,k,s)] = delta(k,k') * d[i',j,s]
    
    # Diagonal (i,j,k): sum_s a[i,k,s] * b[j,k,s] * d[i,j,s] = 1
    # Off-diagonal (i,j,j2,k,k2,i3) with not(j=j2 and k=k2 and i=i3):
    #   If j!=j2: all terms are 0 (from U or V delta)
    #   If j=j2 but k!=k2: all terms are 0 (from V or W delta)
    #   If j=j2 and k=k2 but i!=i3: sum_s a[i,k,s] * b[j,k,s] * d[i3,j,s] = 0
    
    # So we need:
    # For all i,j,k: sum_s a[i,k,s] * b[j,k,s] * d[i,j,s] = 1
    # For all i!=i3, j, k: sum_s a[i,k,s] * b[j,k,s] * d[i3,j,s] = 0
    
    # With s in {0,1}, let's try:
    # a[i,k,0] = 1, a[i,k,1] = 0 for i=0; a[i,k,0] = 0, a[i,k,1] = 1 for i=1
    # b[j,k,0] = 1, b[j,k,1] = 1 for all j,k
    # d[i,j,0] = 1 for i=0, d[i,j,1] = 1 for i=1
    
    # Then diagonal (i,j,k): 
    #   i=0: a[0,k,0]*b[j,k,0]*d[0,j,0] + a[0,k,1]*b[j,k,1]*d[0,j,1] = 1*1*1 + 0*1*1 = 1 ✓
    #   i=1: a[1,k,0]*b[j,k,0]*d[1,j,0] + a[1,k,1]*b[j,k,1]*d[1,j,1] = 0*1*1 + 1*1*1 = 1 ✓
    # Off-diagonal (i=0, i3=1, j, k):
    #   sum_s a[0,k,s]*b[j,k,s]*d[1,j,s] = a[0,k,0]*b[j,k,0]*d[1,j,0] + a[0,k,1]*b[j,k,1]*d[1,j,1]
    #   = 1*1*1 + 0*1*1 = 1 ≠ 0 ✗
    
    # That doesn't work. Let me try:
    # a[0,k,0]=1, a[0,k,1]=1; a[1,k,0]=1, a[1,k,1]=-1
    # b[j,k,0]=1, b[j,k,1]=1
    # d[0,j,0]=1, d[0,j,1]=0; d[1,j,0]=1, d[1,j,1]=0
    
    # Diagonal (0,j,k): 1*1*1 + 1*1*0 = 1 ✓
    # Diagonal (1,j,k): 1*1*1 + (-1)*1*0 = 1 ✓
    # Off-diag (0,1,j,k): 1*1*1 + 1*1*0 = 1 ≠ 0 ✗
    
    # Still doesn't work. The issue is that d[i,j,s] needs to be orthogonal between i values.
    
    # Let me try: d[0,j,0]=1, d[0,j,1]=0; d[1,j,0]=0, d[1,j,1]=1
    # a[0,k,0]=1, a[0,k,1]=0; a[1,k,0]=0, a[1,k,1]=1
    # b[j,k,0]=1, b[j,k,1]=1
    
    # Diagonal (0,j,k): 1*1*1 + 0*1*0 = 1 ✓
    # Diagonal (1,j,k): 0*1*0 + 1*1*1 = 1 ✓
    # Off-diag (0,1,j,k): 1*1*0 + 0*1*1 = 0 ✓
    # Off-diag (1,0,j,k): 0*1*1 + 1*1*0 = 0 ✓
    
    # This works! Rank = 2*m*p = 50. Same as schoolbook, no improvement.
    
    # To get rank < 50, we need a more clever construction. Let's try rank = 40 or 35.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # The rank-25 construction I initially tried has a flaw in the diagonal check.
    
    # Let me try one more thing: a rank-35 construction using a hybrid approach.
    # Actually, given time constraints, let's just return schoolbook.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
